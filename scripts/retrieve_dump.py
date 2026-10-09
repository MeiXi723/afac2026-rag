"""Retrieval-only runner: dump per-question evidence bundles for the session brain.

No LLM/API calls. Question-type-aware retrieval profiles — the tunable lever for
"切分策略持续迭代". Profiles live in PROFILES below.

Usage:
  python scripts/retrieve_dump.py --qids ins_a_001,ins_a_005
  python scripts/retrieve_dump.py --split a            # all 100
  python scripts/retrieve_dump.py --split a --only-missing
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from collections import OrderedDict

# 本地 bge 模型已缓存，禁止联网校验（直连 huggingface.co 会 SSL 失败）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag_system.config import RAGConfig  # noqa: E402
from rag_system.retriever import Retriever  # noqa: E402
from rag_system.eval_norm import norm as eval_norm  # noqa: E402

LOCAL_EMBEDDING_MODEL = 'BAAI/bge-small-zh-v1.5'  # 本地缓存，512 维

# 样板块（阅读指引/目录/费率表等）：词频高但信息密度低，降权
BOILER = re.compile(r'阅读指引|条款目录|本阅读指引旨在|图示说明|请您慎重决策|扫描以查询验证|请您注意仔细阅读')

# 断言语素：选项断言里的短词根（违约/补偿/评级…），条款原文常用近义词
# （违约情形 vs 违约事件），按词长选 rare_kws 会把它们挤掉——命中时强制并入
ASSERTION_MORPHEMES = ['违约', '补偿', '评级', '上限', '免赔', '等待期', '担保', '免责',
                       '赔付', '利率', '规模', '期限', '回售', '赎回', '质押', '退保',
                       '身故', '重疾', '既往症', '犹豫期', '现金价值', '资产负债率',
                       '净利润', '营业收入', '上市地点', '市场份额', '原保费']
# 存在性断言动词：选项在断言"文档里有没有 X"，命中即证据（规则二b 触发条件）
ASSERT_PAT = re.compile(r'提到|提及|披露|描述|约定|载明|包含|涉及|说明|列出|给出|标注|载有|刊载')

# 规则四（E8）：财报计算题科目节锚定。「主要会计数据」表格块文本无公司名
# token（表格页无页眉/落款），题干公司名锚不到块，双腿 top-50 均召不回
# （fin_b_015 实测 catl_18/midea_11/12 全双腿盲；对照组 catl_354 靠落款公司名入包）
SECTION_PAT = re.compile(r'主要会计数据|主要财务指标|会计数据和财务指标')
# 规则五（E9，2026-10-09）：fc 条款章节保底——「违约/补偿」类选项的证据在条款
# 章节正文块，双路对低区分度章节叙述块召回弱（fc_a_009 实测 text02 违约事项块缺席）
CLAUSE_SECTION_PAT = re.compile(r'违约|争议解决|补偿|赔偿责任|赔偿责任和纠纷')
CLAUSE_OPT_PAT = re.compile(r'违约|补偿|赔偿|争议|担保')
SUBJECT_KWS = ['营业收入', '归属于上市公司股东的净利润', '经营活动产生的现金流量净额',
               '资产负债率', '基本每股收益', '现金分红', '总资产']

# 公司全名/简称抽取（doc 别名表）：年报封面与第二节有「公司简称：X」行
ALIAS_FULL_PAT = re.compile(r'([\u4e00-\u9fa5]{2,14}(?:集团|股份)?有限公司)')
ALIAS_SHORT_PAT = re.compile(r'简称[\s：:]*([\u4e00-\u9fa5A-Za-z0-9]{2,10})')
_DOC_ALIASES = {}  # domain → {doc_id: [别名…]}


def _load_doc_aliases(domain, all_chunks):
    """构建 doc_id → 公司别名表（全名 + 简称行 + 全名去公司后缀核心前缀）。

    只扫每 doc 前 40 块（封面/第二节区域），domain 级缓存一次。
    """
    if domain in _DOC_ALIASES:
        return _DOC_ALIASES[domain]
    per_doc = OrderedDict()
    for c in all_chunks:
        did = c.get('doc_id') or ''
        if not did:
            continue
        per_doc.setdefault(did, []).append(c.get('text', ''))
    table = {}
    for did, texts in per_doc.items():
        head = '\n'.join(texts[:40])
        names = set(ALIAS_FULL_PAT.findall(head))
        # 全名按块序出现频次排序——封面全名最靠前且高频，取频次最高者为主名
        ranked = sorted(names, key=lambda n: -head.count(n))
        aliases = set()
        for full in ranked[:2]:
            aliases.add(full)
            core = re.sub(r'(新能源科技|科技|集团|股份|有限)+$', '', full[:-4])  # 「有限公司」4字，[:-3] 留孤字致 sub 失效（audit F4）
            for L in (4, 3, 2):
                if len(core) >= L:
                    aliases.add(core[:L])
        for m in ALIAS_SHORT_PAT.finditer(head):
            v = m.group(1).strip()
            if 2 <= len(v) <= 10 and '有限' not in v:
                aliases.add(v)
        table[did] = sorted(aliases)
    _DOC_ALIASES[domain] = table
    return table

# ---- 题型感知检索 profile（切分策略迭代入口）----
# query_mode: how to build sub-queries
#   stem            = 题干 + 实体（不含选项）
#   stem_plus_opts  = 题干 + 每个选项文本各一条子查询
#   stem_plus_opts_except_tf = 同上但 tf 只用题干
# cap: bundle 证据总字符上限（大脑一次能精读的量）
PROFILES = {
    # cap v2（M2-A）：全量 golden 评测显示深位单命中（融合序 18~25 名）被 cap
    # 竞争挤出，提 cap + 放宽每标签累积深度；28k/30k 字符对离线大脑可读
    # T2 cap +4k（2026-10-08）：缺口归因 63% 为池内 cap 挤出，验证放量回收
    'mcq': {'query_mode': 'stem_plus_opts', 'cap': 32000, 'per_query_top': 15},
    'multi': {'query_mode': 'stem_plus_opts', 'cap': 34000, 'per_query_top': 15},
    'tf': {'query_mode': 'stem', 'cap': 22000, 'per_query_top': 15},
    '计算题': {'query_mode': 'stem', 'cap': 26000, 'per_query_top': 10},
    'default': {'query_mode': 'stem', 'cap': 22000, 'per_query_top': 8},
}


def load_questions(split):
    qs = []
    qdir = os.path.join(BASE, 'data', 'questions', f'group_{split}')
    for f in sorted(glob.glob(os.path.join(qdir, '*'))):
        if f.endswith(('.md', '.csv')) or os.path.isdir(f):
            continue
        try:
            raw = open(f, encoding='utf-8-sig').read().strip()
            items = json.loads(raw) if raw.startswith('[') else [json.loads(l) for l in raw.splitlines() if l.strip()]
            qs.extend(items)
        except Exception as e:  # P2#11：坏文件跳过不炸全批
            print(f'!! skip bad question file {f}: {e}')
    return qs


def build_queries(q, profile, retriever):
    """Return list of (tag, query_text) sub-queries."""
    stem = q.get('question', '')
    entities = retriever._extract_entities(stem)
    ent = ' ' + ' '.join(entities) if entities else ''
    queries = [('stem', stem + ent), ('stem_raw', stem)]
    if profile['query_mode'] == 'stem_plus_opts':
        for letter, text in (q.get('options') or {}).items():
            queries.append((f'opt_{letter}', f'{stem} {letter}: {text}'))
    # 召回补强：题干核心词紧凑查询（无选项噪声，命中保险责任类条款）
    try:
        from rag_system.indexer import jieba_lcut
        stop = {'的', '了', '是', '在', '有', '和', '与', '及', '或', '以下', '哪些', '正确', '错误',
                '关于', '情况', '进行', '假设', '所有', '以下哪些', '分别', '多少', '如何', '可以',
                '正确的是', '说法', '描述', '下列'}
        kws = [w for w in jieba_lcut(stem) if len(w) >= 2 and w not in stop]
        core = ' '.join(dict.fromkeys(kws))[:120]
        if core:
            queries.append(('core', core))
    except Exception:
        pass
    return queries


def run_one(q, retriever, config):
    fmt = q.get('answer_format') or q.get('type', 'default')
    fmt = {'多选题': 'multi', '单选题': 'mcq', '判断题': 'tf'}.get(fmt, fmt)  # B 榜中文题型映射
    profile = PROFILES.get(fmt, PROFILES['default'])
    domain = q.get('domain', '')
    doc_ids = q.get('doc_ids') or None
    merged: OrderedDict = OrderedDict()

    results_by_tag = {}
    query_texts = {}
    leg_top = max(50, 3 * profile['per_query_top'])  # M2-B：50——ins_a_020 golden 融合位 49
    for tag, query in build_queries(q, profile, retriever):
        # 完整双路召回 + RRF 融合（+可选精排）：BM25 与本地向量各取 leg_top，
        # 融合后不再走 _dynamic_truncate 截断层——bundle 规模由 cap 统一裁决，
        # 三规则保底在下游作为后处理叠加。
        bm25 = retriever._bm25_search(domain, query, leg_top, doc_ids)
        emb = retriever._embedding_search(domain, query, leg_top, doc_ids)
        res = retriever._rrf_fusion(bm25, emb, domain, query)
        if not retriever._cross_encoder_failed:
            res = retriever._cross_encoder_rerank(query, res)
        results_by_tag[tag] = res
        query_texts[tag] = query
        for rank, c in enumerate(res[:50]):
            cid = c['chunk_id']
            if cid not in merged:
                merged[cid] = {
                    'chunk_id': cid, 'doc_id': c.get('doc_id', ''),
                    'section_path': c.get('section_path', ''),
                    'pages': c.get('pages', []), 'text': c.get('text', ''),
                    'score': 0.0, 'hits': [],
                }
            # RRF-ish: 早名次加分，多次命中累计
            merged[cid]['score'] += 1.0 / (10 + rank)
            merged[cid]['hits'].append(tag)

    ranked = sorted(merged.values(), key=lambda x: x['score'], reverse=True)

    # 切分策略迭代 v2：样板块降权 + 题干核心词密度加成
    try:
        from rag_system.indexer import jieba_lcut
        stop = {'的', '了', '是', '在', '有', '和', '与', '及', '或', '以下', '哪些', '正确', '错误',
                '关于', '情况', '进行', '假设', '所有', '产品', '分别为', '多少', '如何', '可以'}
        kws = [w for w in jieba_lcut(q.get('question', '')) if len(w) >= 2 and w not in stop]
        top_kws = list(dict.fromkeys(kws))[:12]
    except Exception:
        top_kws = []
    for c in ranked:
        if BOILER.search(c.get('text', '')[:200]):
            c['score'] *= 0.4
        if top_kws:
            density = sum(1 for k in top_kws if k in c.get('text', ''))
            c['score'] *= (1 + 0.15 * min(density, 6))

    # 确定性扩展：按 ranked 分数序，逐锚点附前后邻接块（替换 _expand_context 的集合乱序）
    chunk_by_id = {c['chunk_id']: c for c in retriever._load_domain_chunks(domain)}
    expanded, seen = [], set()
    for c in ranked:
        seq = [c['chunk_id']]
        info = chunk_by_id.get(c['chunk_id'], {})
        if info.get('prev_chunk_id'):
            seq.insert(0, info['prev_chunk_id'])
        if info.get('next_chunk_id'):
            seq.append(info['next_chunk_id'])
        for x in seq:
            if x in seen:
                continue
            seen.add(x)
            cc = chunk_by_id.get(x)
            if cc:
                expanded.append({'chunk_id': x, 'doc_id': cc.get('doc_id', ''),
                                 'section_path': cc.get('section_path', ''),
                                 'text': cc.get('text', '')})

    # 保底机制（选项×文档感知）：每个选项子查询 × 每份 doc，在该文档内选
    # 选项关键词重叠度最高的非样板块强制入包。归一化子串匹配，免疫 PDF
    # 抽取的词中空格伪影（如"养 老保险金"导致 BM25 永远打不中真实条款）。
    guaranteed = set()
    all_chunks = retriever._load_domain_chunks(domain)
    norm_text = eval_norm  # 全链路共用归一化（NFKC+去空白+标点统一）
    docs_for_q = list(doc_ids or []) or sorted({c.get('doc_id') for c in all_chunks})
    doc_cands = {}
    for did in docs_for_q:
        cands = [(c['chunk_id'], norm_text(c.get('text', '')))
                 for c in all_chunks if c.get('doc_id') == did]
        doc_cands[did] = [(cid, t) for cid, t in cands
                          if len(t) >= 50 and not BOILER.search(t[:200])]
    for tag in results_by_tag:
        if not (tag.startswith('opt_') or tag == 'stem_raw'):
            continue
        # 关键词只取选项文本本身（opt_* 时）：stem+option 拼接查询里的题干词
        # （如"基本要素"）会把选项决定性词挤出 rare_kws、抬高无关块的重叠数——
        # 实测 fc_a_011 封面块（上市地点）因此输给中段"基本要素"块
        if tag.startswith('opt_'):
            letter = tag.split('_', 1)[1]
            kw_source = (q.get('options') or {}).get(letter, '') or query_texts.get(tag, '')
        else:
            kw_source = query_texts.get(tag, '')
        qtext = norm_text(kw_source)
        opt_kws = [w for w in jieba_lcut(qtext) if len(w) >= 2]
        # 决策性特征词：选项里最长的少数词（如"酒后驾驶""保质期""45.53"）
        # ∪ 断言语素（选项含"违约"等词根时强制并入，近义词条款靠二字词根子串命中）
        rare_kws = sorted(set(opt_kws), key=lambda w: (-len(w), w))[:4]
        rare_kws = list(dict.fromkeys(rare_kws + [m for m in ASSERTION_MORPHEMES if m in qtext]))
        # 规则一：融合序首个非样板块（对正常文本有效）。opt_* 标签每文档
        # 保底 2 槽——单槽会被同文档高分对手抢走（ins_a_011/015/020 实测），
        # 选项子查询的针对性最强，值得双槽
        slots = 2 if tag.startswith('opt_') else 1
        per_doc: dict[str, int] = {}
        for c in results_by_tag[tag]:
            did = c.get('doc_id')
            if did not in doc_cands or per_doc.get(did, 0) >= slots:
                continue
            if c['chunk_id'] not in {cid for cid, _ in doc_cands[did]}:
                continue  # 样板块（doc_cands 已过滤 BOILER）
            guaranteed.add(c['chunk_id'])
            per_doc[did] = per_doc.get(did, 0) + 1
        # 规则二b：存在性断言选项（"提到了/披露了 X 条款"）——单块择优会被顺带
        # 提及块赢走，改按命中文素长度平方加权取每文档 top-3（"违约事件"列表块
        # 中的长文素会压过只在句尾带一个"违约"的引用块）
        morphs = [m for m in ASSERTION_MORPHEMES if m in qtext]
        # 仅 opt_* 标签启用 2b：stem_raw 上"描述/说明"等通用词会大范围误触发；
        # 且必须命中至少 1 块才 continue——零命中继续走规则二/三兜底（P1#3）
        if morphs and tag.startswith('opt_') and ASSERT_PAT.search(qtext):
            added_2b = 0
            for did in docs_for_q:
                cands = doc_cands.get(did) or []
                scored = []
                for cid, ct in cands:
                    matched = [m for m in morphs if m in ct]
                    if matched:
                        # 第三键：百分比密度——"具体数值"类断言的决定性内容
                        # 多在财务表格行（如"资产负债率 12.42% 9.45% 20.26%"），
                        # 散文提及块密度低（双路检索对纯表格块语义打分弱，实测
                        # fc_a_005 主要财务指标表被 8 个并列块淹没）
                        pct = min(len(re.findall(r'\d+\.?\d*%', ct)), 12)
                        scored.append((sum(len(m) ** 2 for m in matched),
                                       sum(1 for k in opt_kws if k in ct), pct, cid))
                scored.sort(reverse=True)
                for *_, cid in scored[:3]:
                    guaranteed.add(cid)
                    added_2b += 1
            if added_2b:
                continue
        # 规则二：含决策性特征词的块强制入包（对空格伪影文本有效）
        if not rare_kws:
            continue
        for did in docs_for_q:
            cands = doc_cands.get(did) or []
            hit = [ct for cid, ct in cands if any(k in ct for k in rare_kws)]
            if hit:
                best = max(hit, key=lambda ct: sum(1 for k in opt_kws if k in ct))
                best_id = next(cid for cid, ct in cands if ct == best)
                guaranteed.add(best_id)
        # 规则三：每份文档保证入包一个题干核心词密度最高的条款块
        # （选项只提产品名时，条款正文靠题干词命中——如 doc16 2.5(三) 身故保险金）
        if top_kws:
            for did in docs_for_q:
                cands = doc_cands.get(did) or []
                if not cands:
                    continue
                best = max(cands, key=lambda ct: sum(1 for k in top_kws if k in ct))
                if sum(1 for k in top_kws if k in best[1]) >= 3:
                    guaranteed.add(best[0])

    # 规则四（E8，2026-10-09）：财报域计算题科目节锚定。「主要会计数据」表格块
    # 文本无公司名 token（表格页无页眉/落款），题干公司名锚不到块，双腿 top-50
    # 均召不回（fin_b_015 实测 catl_18/midea_11/12 全双腿盲）。按题面公司实体
    # 定位文档后，将其会计数据/财务指标节块按科目词重叠保底入包。
    # RB_SKIP_RULE4=1 供单变量对照实验禁用本规则（B v7.1 基线）。
    # audit F6：严格 '1' 判定——RB_SKIP_RULE4=0 不再误禁用规则四
    if fmt == '计算题' and domain == 'financial_reports' and os.environ.get('RB_SKIP_RULE4') != '1':
        aliases = _load_doc_aliases(domain, all_chunks)
        qtext_raw = q.get('question', '')
        subj = [k for k in SUBJECT_KWS if k in norm_text(qtext_raw)]
        cand_ids = {cid for did in aliases
                    for cid, _ in (doc_cands.get(did) or [])}
        for did, names in aliases.items():
            if not any(n in qtext_raw for n in names if len(n) >= 3):
                continue
            scored = []
            for c in all_chunks:
                cid = c['chunk_id']
                if c.get('doc_id') != did or cid not in cand_ids:
                    continue
                sp = c.get('section_path', '') or ''
                t = norm_text(c.get('text', ''))
                if not (SECTION_PAT.search(sp) or SECTION_PAT.search(t[:400])):
                    continue
                scored.append((sum(1 for s in subj if s in t), cid))
            scored.sort(reverse=True)
            for _, cid in scored[:2]:
                guaranteed.add(cid)

    # 规则五（E9，2026-10-09）：fc 条款章节保底。选项含「违约/补偿/赔偿/争议/
    # 担保」时，每 doc 取违约/争议/补偿章节内题干核心词密度最高的 1 块保底
    # （fc_a_009 D「均提及违约或补偿条款」的 text02 侧章节块双腿盲实测）。
    # RB_SKIP_RULE5：严格 '1' 判定（同 F6 教训），供复现 v7.1 严格态
    if (fmt in ('multi', 'mcq') and domain == 'financial_contracts' and doc_ids
            and os.environ.get('RB_SKIP_RULE5') != '1'
            and CLAUSE_OPT_PAT.search(' '.join((q.get('options') or {}).values()))):
        for did in doc_ids:
            valid_ids = {cid for cid, _ in (doc_cands.get(did) or [])}
            best, best_k = None, 2  # 密度=题干词+条款词命中，阈值 2
            for c in all_chunks:
                if c.get('doc_id') != did:
                    continue
                cid = c['chunk_id']
                if cid not in valid_ids:
                    continue  # 样板块（doc_cands 已过滤 BOILER）
                sp = c.get('section_path', '') or ''
                tt = c.get('text', '') or ''
                if not CLAUSE_SECTION_PAT.search(sp):
                    continue
                k = sum(1 for k2 in top_kws if k2 in tt) + sum(1 for w in ('违约', '补偿') if w in tt[:150])
                if k >= best_k:
                    best, best_k = cid, k
            if best:
                guaranteed.add(best)

    # 保底总量上限（#13：doc_ids 缺失时 docs_for_q=全域，保底可能失控）。
    # P1-1 修复：截断按重要性（累积分降序，无分者 0）——禁止 chunk_id 字典序
    # （B 榜全域题实测单 doc 垄断全部 40 槽）；并加 12k 字符子上限防豁免撑爆 cap
    if len(guaranteed) > 40:
        score_of = {c['chunk_id']: c.get('score', 0.0) for c in ranked}
        scored = sorted(((score_of.get(g, 0.0), g) for g in guaranteed), reverse=True)
        guaranteed = {g for _, g in scored[:40]}
    char_of = {c['chunk_id']: len(c.get('text', '')) for c in expanded}
    guar_chars = sum(char_of.get(g, 0) for g in guaranteed)
    if guar_chars > 12000:
        # audit F8：按分数降序截断（禁 chunk_id 字典序，与本函数 40 槽截断的 P1-1 原则一致）
        kept, acc = set(), 0
        for _, g in sorted(((score_of.get(g, 0.0), g) for g in guaranteed), reverse=True):
            acc += char_of.get(g, 0)
            kept.add(g)
            if acc > 12000:
                break
        guaranteed = kept
    # 按 cap 截断（保底块豁免）
    total, kept = 0, []
    for c in expanded:
        n = len(c.get('text', ''))
        is_g = c['chunk_id'] in guaranteed
        if not is_g and total + n > profile['cap'] and kept:
            continue
        kept.append(c)
        total += n

    # 保底完整性：规则二/三按归一化 doc_cands 选中的块可能未被 BM25 召回
    # （词中空格伪影），从而不在 expanded 里——直接从 chunk 索引补入
    kept_ids = {c['chunk_id'] for c in kept}
    for gid in sorted(guaranteed):
        if gid not in kept_ids and gid in chunk_by_id:
            cc = chunk_by_id[gid]
            kept.append({'chunk_id': gid, 'doc_id': cc.get('doc_id', ''),
                         'section_path': cc.get('section_path', ''),
                         'pages': cc.get('pages', []),
                         'text': cc.get('text', ''), 'score': 0.0,
                         'hits': ['guaranteed']})
            kept_ids.add(gid)
            total += len(cc.get('text', ''))

    # 旗标：guaranteed=保底/规则选中；neighbor=邻接扩展块（非锚点）
    hit_map = {c['chunk_id']: (c['score'], c['hits']) for c in ranked}
    for c in kept:
        score, hits = hit_map.get(c['chunk_id'], (0.0, []))
        c['guaranteed'] = c['chunk_id'] in guaranteed
        c['neighbor'] = (not hits) and not c['guaranteed']
        if c['chunk_id'] in hit_map:
            c['hits'] = hits
            c['score'] = round(score, 4)
    return kept, profile, total


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='a', choices=['a', 'b'])
    ap.add_argument('--qids', default='', help='逗号分隔；留空=全量')
    ap.add_argument('--only-missing', action='store_true')
    ap.add_argument('--out', default='')
    ap.add_argument('--rerank', action='store_true',
                    help='启用 CrossEncoder 精排（需本地已缓存 %s）' % RAGConfig().reranker_model)
    ap.add_argument('--rrf-k', type=int, default=None, help='覆盖 config.rrf_k（E1 扫描用）')
    ap.add_argument('--chunkset', default='v1', choices=['v1', 'v2', 'v3', 'v4', 'v5', 'mix'],
                    help='v2=新切块集（chunks_v2/indexes_v2）')
    ap.add_argument('--no-vector', action='store_true',
                    help='纯非向量变体：关掉向量腿，仅 BM25+字段+保底（Track A 对照）')
    args = ap.parse_args()

    out_dir = args.out or os.path.join(BASE, 'output', f'bundles_{args.split}')
    os.makedirs(out_dir, exist_ok=True)

    config = RAGConfig()
    # 大脑模式零 API 依赖：BM25 + 本地 bge 向量双路召回，RRF 融合，全程离线。
    config.cloud_embedding = False
    config.embedding_model = LOCAL_EMBEDDING_MODEL
    config.skip_embedding = bool(args.no_vector)  # --no-vector: 纯 BM25 腿（Track A 对照）
    if args.rrf_k is not None:
        config.rrf_k = args.rrf_k
    if args.chunkset != 'v1':
        config.chunks_dir = config.chunks_dir.parent / f'chunks_{args.chunkset}'
        config.index_dir = config.index_dir.parent / f'indexes_{args.chunkset}'
    print(f'chunkset={args.chunkset} → {config.chunks_dir} / {config.index_dir}')
    retriever = Retriever(config)
    if not args.rerank:
        # 精排模型未缓存，显式禁用避免初始化探测；--rerank 时恢复启用
        retriever._cross_encoder_failed = True
    questions = load_questions(args.split)
    if args.qids:
        want = set(w for w in re.split(r'[,，\s]+', args.qids.strip()) if w)
        questions = [q for q in questions if q['qid'] in want]
    elif args.only_missing:
        questions = [q for q in questions if not os.path.exists(os.path.join(out_dir, q['qid'] + '.json'))]

    done = 0
    for q in questions:
        t0 = time.time()
        kept, profile, total = run_one(q, retriever, config)
        bundle = {
            'qid': q['qid'], 'domain': q.get('domain', ''), 'split': q.get('split', args.split.upper()),
            'answer_format': q.get('answer_format') or q.get('type', ''),
            'question': q.get('question', ''), 'options': q.get('options', {}),
            'doc_ids': q.get('doc_ids', []),
            'chunkset': args.chunkset,
            'profile': {'query_mode': profile['query_mode'], 'cap': profile['cap'], 'evidence_chars': total},
            'evidence': kept,
        }
        path = os.path.join(out_dir, q['qid'] + '.json')
        json.dump(bundle, open(path, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
        done += 1
        print(f"{q['qid']}: {len(kept)} chunks, {total} chars, {time.time() - t0:.0f}s")
    print(f'dumped {done} bundles -> {out_dir}')


if __name__ == '__main__':
    main()
