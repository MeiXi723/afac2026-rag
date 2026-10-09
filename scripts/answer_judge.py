"""答案判分单一真源（2026-10-09）：字母题全等 + B 榜数值容差。

治理背景：eval_v2 / eval_faith / eval_report 三处各自拷贝判分逻辑，已发生两次
漂移事故（faith 参数序反、report 纯全等无容差）——本模块为唯一实现，三处
统一导入，禁止再行复刻（GOVERNANCE 注意事项 4「两侧必须同构」的根因治理）。
逻辑逐行提取自 eval_v2.b_value_ok（权威判分器，行为零变化）。
"""
import re

_NUM_PAT = re.compile(r'-?\d[\d,]*(?:\.\d+)?\s*%?')


def value_ok(g: str, o: str) -> bool:
    """B 榜数值容差判分（S4）：数值序列数量相等且逐位相对误差 <=0.5%
    （或去 % 后两位小数全等）即正确；全等字符串直接通过。g=GT，o=我方答案
    ——token 校验以 GT 为基准，参数序不可颠倒（faith 参数序事故根因）。"""
    # 日期/负号歧义：'-' 会被数值 regex 当负号，先归一为分隔符
    g2 = g.replace('约', '').replace('％', '%').replace('-', '，')
    o2 = o.replace('约', '').replace('％', '%').replace('-', '，')
    if g2.replace(' ', '') == o2.replace(' ', ''):
        return True
    gn = [(m.group().replace(',', ''), '%' in m.group()) for m in _NUM_PAT.finditer(g2)]
    on = [(m.group().replace(',', ''), '%' in m.group()) for m in _NUM_PAT.finditer(o2)]
    if not gn or len(gn) != len(on):
        return False
    for (gv, gp), (ov, op) in zip(gn, on):
        if gp != op:
            return False
        try:
            a, b_ = float(gv), float(ov)
        except ValueError:
            if gv != ov:
                return False
            continue
        if round(a, 2) != round(b_, 2) and abs(a - b_) / max(abs(a), 1e-9) > 0.005:
            return False
    # 中文/字母 token 序列校验——排序语义与单位语义（万元 vs 亿元）在纯数字
    # 比对下不可见；GT 含中文 token 时要求答案同类 token 序列完全一致
    gt_toks = re.findall(r'[一-龥A-Za-z]{2,}', g2)
    o_toks = re.findall(r'[一-龥A-Za-z]{2,}', o2)
    if gt_toks and gt_toks != o_toks:
        return False
    return True


def answer_ok(gt_answer: str, got_answer: str, split: str) -> bool:
    """统一入口：字母题（集合 ⊆ ABCD）大小写全等；B 榜非字母答案走数值容差。"""
    g = str(gt_answer).upper()
    o = str(got_answer).upper()
    if split == 'b' and not set(g) <= set('ABCD'):
        return value_ok(str(gt_answer), str(got_answer))
    return g == o
