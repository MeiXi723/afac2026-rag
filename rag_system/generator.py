import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from rag_system.chunker import Chunk
from rag_system.retriever import Retriever

logger = logging.getLogger(__name__)

DOMAIN_RULES: dict[str, str] = {
    "financial_contracts": "注意：合同中金额单位「万元」「亿元」需留意，产品代码（如JCK、JLX等字母编号）为唯一产品标识。",
    "financial_reports": "注意：财务数据百分比需区分「同比增长」「环比增长」「占比」，「增长X%」与「增至X%」含义不同。",
    "insurance": "注意：不同险种的保险金责任相互独立（如「一般医疗保险金」与「特定疾病保险金」不可混用），赔付以条款约定为准。",
    "regulatory": "注意：法规条款编号（「第X条」）为精确引用，「应」「必须」「可以」法律效力不同，施行日期与修订日期需区分。",
    "research": "注意：研究报告中的观点（预测/推测）与历史数据不可混淆，「建议」「认为」不等同于「数据显示」。",
}


def _extract_json_answer(raw: str) -> dict:
    if not raw:
        return {"answer": "", "reasoning": "", "evidence_ids": [], "parse_error": True}
    for extractor in [_try_direct_json, _try_code_block, _try_brace, _try_regex_letters]:
        result = extractor(raw)
        if result:
            return result
    return {"answer": "", "reasoning": "", "evidence_ids": [], "parse_error": True}


def _try_direct_json(raw: str) -> Optional[dict]:
    try:
        d = json.loads(raw)
        if "answer" in d or "judgment" in d:
            return d
    except json.JSONDecodeError:
        pass
    return None


def _try_code_block(raw: str) -> Optional[dict]:
    m = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw)
    if m:
        return _try_direct_json(m.group(1).strip())
    return None


def _try_brace(raw: str) -> Optional[dict]:
    m = re.search(r'\{[^{}]*\}', raw)
    if m:
        return _try_direct_json(m.group(0))
    return None


def _try_regex_letters(raw: str) -> Optional[dict]:
    letters = re.findall(r'\b([A-D])\b', raw)
    unique = list(dict.fromkeys(letters))
    if unique:
        return {"answer": "".join(unique), "reasoning": "", "evidence_ids": [], "parse_fallback": True}
    return None


def _neutralize_tf_options(options: dict[str, str]) -> tuple[dict[str, str], dict[str, str]]:
    new_options = {}
    label_map = {}
    for i, (orig_label, text) in enumerate(sorted(options.items()), start=1):
        neutral_label = f"选项{i}"
        new_options[neutral_label] = text
        label_map[neutral_label] = orig_label
    return new_options, label_map


def _reasoning_contradicts(reasoning: str, letter: str) -> bool:
    if not reasoning or not reasoning.strip():
        return False
    text = reasoning.strip()
    full_text_patterns = [
        rf'{letter}\s*[是以为]?\s*错误',
        rf'{letter}\s*[是以为]?\s*不正确',
        rf'{letter}\s*[是以为]?\s*不成立',
        rf'{letter}\s*[是以为]?\s*不对',
        rf'排除\s*(?:选项)?\s*{letter}',
        rf'不应选\s*(?:选项)?\s*{letter}',
        rf'不能选\s*(?:选项)?\s*{letter}',
        rf'故\s*(?:选项)?\s*{letter}\s*[错不]',
    ]
    for pat in full_text_patterns:
        if re.search(pat, text):
            if not re.search(rf'{letter}\s*不[是为]错误|{letter}\s*并非\s*(错误|不正确)', text):
                return True
    sentences = re.split(r'[。！？；\n]', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_patterns = [
        rf'{letter}\s*[是以为]?\s*不准确',
        rf'{letter}\s*[是以为]?\s*不符',
        rf'选项\s*{letter}\s*[是以为]?\s*错误',
        rf'选项\s*{letter}\s*[是以为]?\s*不正确',
        rf'选项\s*{letter}\s*[是以为]?\s*不成立',
        rf'不选\s*(?:选项)?\s*{letter}',
        rf'因此\s*(?:选项)?\s*{letter}\s*不',
        rf'所以\s*(?:选项)?\s*{letter}\s*不',
    ]
    for sentence in sentences:
        if letter not in sentence:
            continue
        for pat in sentence_patterns:
            if re.search(pat, sentence):
                if not re.search(rf'{letter}\s*不[是为]错误|{letter}\s*并非\s*(错误|不正确)', sentence):
                    return True
    return False


def _assemble_multi_answer(
    judgments: list[dict], all_option_evidence: dict[str, list]
) -> dict:
    yes_letters = sorted([j["letter"] for j in judgments if j.get("judgment") == "YES"])
    if not yes_letters:
        evidence_ids = []
        for ev_list in all_option_evidence.values():
            for ev in ev_list:
                evidence_ids.append(ev["chunk_id"])
        return {
            "answer": "", "reasoning": "", "evidence_ids": list(dict.fromkeys(evidence_ids))[:15],
            "needs_retry": True, "confidence": "low",
        }
    evidence_ids = []
    for j in judgments:
        if j["letter"] in yes_letters:
            evidence_ids.extend(j.get("evidence_ids", []))
    return {
        "answer": "".join(yes_letters),
        "reasoning": "; ".join(j.get("reasoning", "") for j in judgments if j["letter"] in yes_letters),
        "evidence_ids": list(dict.fromkeys(evidence_ids))[:15],
    }


def _expand_context(
    evidence: list[dict], chunk_map: list[dict], n: int = 1
) -> list[dict]:
    chunk_by_id: dict[str, dict] = {c["chunk_id"]: c for c in chunk_map}
    expanded_ids: set[str] = set()
    for ev in evidence:
        cid = ev.get("chunk_id", "")
        expanded_ids.add(cid)
        chunk = chunk_by_id.get(cid)
        if not chunk:
            continue
        current = chunk
        for _ in range(n):
            prev_id = current.get("prev_chunk_id")
            if prev_id and prev_id not in expanded_ids:
                expanded_ids.add(prev_id)
                current = chunk_by_id.get(prev_id)
                if not current:
                    break
            else:
                break
        current = chunk
        for _ in range(n):
            next_id = current.get("next_chunk_id")
            if next_id and next_id not in expanded_ids:
                expanded_ids.add(next_id)
                current = chunk_by_id.get(next_id)
                if not current:
                    break
            else:
                break
    expanded = []
    for cid in expanded_ids:
        c = chunk_by_id.get(cid)
        if c:
            expanded.append({
                "chunk_id": cid,
                "text": c["text"],
                "doc_id": c.get("doc_id", ""),
                "section_path": c.get("section_path", ""),
            })
    return expanded


class Generator:
    def __init__(self, config) -> None:
        self.config = config
        self.retriever = Retriever(config)
        self._deepseek_client: Optional[object] = None
        self._qwen_client: Optional[object] = None

    def answer_question(self, question: dict) -> dict:
        answer_format = question.get("answer_format", "mcq")
        evidence = self.retriever.search(question)
        if not evidence:
            return {"answer": "", "confidence": "failed", "reasoning": "no evidence"}
        evidence = _expand_context(
            evidence,
            self.retriever._load_domain_chunks(question.get("domain", "insurance")),
        )

        if answer_format == "multi":
            return self._answer_multi(question, evidence)
        return self._answer_single(question, evidence, is_tf=(answer_format == "tf"))

    def _answer_single(
        self, question: dict, evidence: list[dict], is_tf: bool = False
    ) -> dict:
        options = dict(question["options"])
        label_map: dict[str, str] = {}
        if is_tf:
            options, label_map = _neutralize_tf_options(options)
        domain = question.get("domain", "")
        prompt = self._build_prompt(question["question"], options, evidence, label_map, domain)
        ds_result = _extract_json_answer(self._call_deepseek(prompt))
        qw_result = _extract_json_answer(self._call_qwen(prompt))
        return {"deepseek": ds_result, "qwen": qw_result}

    def _answer_multi(self, question: dict, evidence: list[dict]) -> dict:
        ds_judgments: list[dict] = []
        qw_judgments: list[dict] = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for letter, option_text in question["options"].items():
                prompt = self._build_yesno_prompt(
                    question["question"], letter, option_text, evidence
                )
                futures[executor.submit(self._call_deepseek, prompt)] = ("deepseek", letter)
                futures[executor.submit(self._call_qwen, prompt)] = ("qwen", letter)
            for future in as_completed(futures):
                model, letter = futures[future]
                try:
                    result = _extract_json_answer(future.result())
                    judgment = result.get("judgment", "NO")
                    reasoning = result.get("reasoning", "")
                    if _reasoning_contradicts(reasoning, letter):
                        judgment = "NO"
                    entry = {
                        "letter": letter,
                        "judgment": judgment,
                        "reasoning": reasoning,
                        "evidence_ids": result.get("evidence_ids", []),
                    }
                except Exception:
                    logger.exception("Multi judgment failed for %s option %s", model, letter)
                    entry = {"letter": letter, "judgment": "NO"}
                if model == "deepseek":
                    ds_judgments.append(entry)
                else:
                    qw_judgments.append(entry)
        return {
            "deepseek": _assemble_multi_answer(ds_judgments, {}),
            "qwen": _assemble_multi_answer(qw_judgments, {}),
        }

    def _build_prompt(
        self, question: str, options: dict, evidence: list[dict],
        label_map: dict[str, str], domain: str = "",
    ) -> str:
        ctx_parts = []
        for e in evidence:
            ctx_parts.append(
                f"--- [doc_id: {e.get('doc_id', '')}, "
                f"section: {e.get('section_path', '')}] ---\n{e['text']}"
            )
        context = "\n\n".join(ctx_parts)
        opts_str = "\n".join(f"{k}: {v}" for k, v in options.items())
        hint = ""
        if label_map:
            hint = (
                "\n注意：" + " ".join(
                    f"选项{k}对应原始标签{v}" for k, v in label_map.items()
                ) + "。最终 answer 字段必须使用原始标签A或B。"
            )
        domain_note = DOMAIN_RULES.get(domain, "")
        return (
            "你是金融分析专家。只基于给定文档内容作答，禁止使用外部常识。\n"
            f"{domain_note + chr(10) if domain_note else ''}"
            "严格只输出一行 JSON，不含任何 markdown 代码块标记或其他文字。\n\n"
            f"context:\n{context}\n\n"
            f"question: {question}\n"
            f"options:\n{opts_str}\n\n"
            '输出格式: {"answer": "单个或多个大写字母", "reasoning": "基于证据的推理", '
            '"evidence_ids": ["chunk_id1", "chunk_id2"]}'
            f"{hint}"
        )

    def _build_yesno_prompt(
        self, question: str, letter: str, option_text: str, evidence: list[dict],
    ) -> str:
        ctx_parts = []
        for e in evidence:
            ctx_parts.append(
                f"--- [doc_id: {e.get('doc_id', '')}] ---\n{e['text']}"
            )
        context = "\n\n".join(ctx_parts)
        return (
            "你是金融分析专家。只基于给定文档内容判断以下选项是否应选入答案。\n"
            "严格只输出一行 JSON。\n\n"
            f"context:\n{context}\n\n"
            f"question: {question}\n"
            f"选项{letter}: {option_text}\n\n"
            '输出格式: {"judgment": "YES"|"NO", "reasoning": "简要说明", '
            '"evidence_ids": ["chunk_id"]}\n\n'
            "入选规则：\n"
            "- 先判断选项是在肯定还是否定题干的询问点（如\"可赔付\"\"正确\"\"成立\"\"给出了公式\"）。\n"
            "- 若选项的核心主张为否定题干询问点（如\"不赔\"\"不适用\"\"不属于\"\"未给公式\"）且该否定陈述本身正确 → 判 NO，不选入。\n"
            "- 若选项的核心主张为否定题干询问点但该否定陈述本身错误 → 判 YES，选入。\n"
            "- 若选项的核心主张为肯定题干询问点，按证据判断陈述是否正确：正确→YES，错误→NO。\n"
            "- 判断陈述是否正确必须基于文档证据，不能依赖常识。"
        )

    def _call_deepseek(self, prompt: str) -> str:
        return self._call_api("deepseek", prompt)

    def _call_qwen(self, prompt: str) -> str:
        return ""

    def _call_api(self, provider: str, prompt: str) -> str:
        if provider == "deepseek":
            client = self._get_deepseek_client()
            model = self.config.deepseek_model
        else:
            client = self._get_qwen_client()
            model = self.config.qwen_model

        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.generator_temperature,
                    max_tokens=self.config.generator_max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_base_delay * (2 ** attempt))
                else:
                    logger.exception(
                        "%s API failed after %d retries", provider, self.config.max_retries
                    )
                    return ""

    def _get_deepseek_client(self):
        if self._deepseek_client is None:
            import openai

            self._deepseek_client = openai.OpenAI(
                api_key=self.config.api_key_deepseek,
                base_url=self.config.api_base_deepseek,
                timeout=self.config.request_timeout,
            )
        return self._deepseek_client

    def _get_qwen_client(self):
        if self._qwen_client is None:
            import openai

            self._qwen_client = openai.OpenAI(
                api_key=self.config.api_key_qwen,
                base_url=self.config.api_base_qwen,
                timeout=self.config.request_timeout,
            )
        return self._qwen_client
