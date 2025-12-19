from __future__ import annotations

import json, re, time, random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
from vllm_client import chat_once

CONFIG: Dict[str, Any] = {
    "qa_path": "output/qa_pairs_english_llm_test.json",
    "chunks_path": "output/chunks_english.json",
    "abstracts_path": "output/abstracts_english_llm.json", 
    "output_path": "output/qa_pairs_english_llm_judged.json",
    "prompt_path": "gen_qa/judge_prompt.txt",

    "resume": True,
    "keep_raw": True,
    "limit": 0,

    "host": "http://219.216.98.15:8000/v1",
    "model": "qwen2.5-72b-awq",
    "temperature": 0.0,
    "max_tokens": 700,

    "retries": 5,
    "retry_base_sleep": 1.2,
    "retry_max_sleep": 15.0,

    # 给模型看的上下文：仅邻居窗口（防漂）
    "context_window": 1,
    "same_header_only": True,
    "max_chunk_chars_each": 12000,
    "max_abstract_chars": 4000,

    # 校验策略：证据必须能在“全文 chunks”里定位到原文子串
    "strict_evidence": True,     # pass/fix 但证据定位失败 => drop
    "max_evidence": 3,
}

ROOT = Path(__file__).resolve().parents[1]
FAIL_REASONS = {"judge_call_error", "judge_output_parse_error", "evidence_validation_failed"}

DASHES = "-‐-‒–—−"  # 各种连字符/减号（含 non-breaking hyphen: \u2011）
APOS = "'’"


def r_in(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return pp.resolve() if pp.exists() else (ROOT / pp).resolve()


def r_out(p: str) -> Path:
    pp = Path(p)
    return pp if pp.is_absolute() else (ROOT / pp).resolve()


def jload(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def trunc(s: str, n: int) -> str:
    return s if (n <= 0 or len(s) <= n) else s[:n] + "\n...[TRUNCATED]..."


def chunk_order(cid: str) -> Tuple[int, str]:
    m = re.search(r"(\d+)$", cid)
    return (int(m.group(1)), cid) if m else (10**9, cid)


def build_docs(chunks: List[Dict[str, Any]]):
    # docs[src] = {"order":[cid...], "chunks":{cid:chunkdict}}
    docs: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        src = c.get("source_file")
        cid = c.get("chunk_id")
        if isinstance(src, str) and isinstance(cid, (str, int)):
            cid = str(cid)
            docs.setdefault(src, {"order": [], "chunks": {}})
            docs[src]["order"].append(cid)
            docs[src]["chunks"][cid] = c
    for src in docs:
        docs[src]["order"].sort(key=chunk_order)
    return docs


def load_abstracts(path: str) -> Dict[str, str]:
    if not path:
        return {}
    p = r_in(path)
    if not p.exists():
        return {}
    recs = jload(p)
    out: Dict[str, str] = {}
    if isinstance(recs, list):
        for r in recs:
            if isinstance(r, dict) and r.get("source_file") and r.get("abstract"):
                out[r["source_file"]] = r["abstract"]
    return out


def get_context_chunks(docs, src: str, main_cid: str) -> List[Dict[str, Any]]:
    d = docs.get(src)
    if not d:
        return []
    main = d["chunks"].get(main_cid)
    if not main:
        return []
    ids: List[str] = d["order"]
    if main_cid not in ids:
        return [main]

    w = int(CONFIG.get("context_window", 0))
    i = ids.index(main_cid)
    lo, hi = max(0, i - w), min(len(ids) - 1, i + w)

    hdr0 = str(main.get("header_path") or "")
    out = []
    for j in range(lo, hi + 1):
        cid = ids[j]
        c = d["chunks"].get(cid)
        if not c:
            continue
        if CONFIG.get("same_header_only", True) and str(c.get("header_path") or "") != hdr0:
            continue
        out.append(c)

    if all(str(x.get("chunk_id")) != main_cid for x in out):
        out.append(main)
    out.sort(key=lambda x: chunk_order(str(x.get("chunk_id"))))
    return out


def make_prompt(tpl: str, qa: Dict[str, Any], ctx: List[Dict[str, Any]], abstract: str) -> str:
    main_cid = str(qa.get("chunk_id") or "")
    blocks = []
    for c in ctx:
        cid = str(c.get("chunk_id"))
        tag = "主chunk" if cid == main_cid else "上下文chunk"
        blocks.append(f"【{tag} | chunk_id={cid}】\n{trunc(str(c.get('content') or ''), int(CONFIG['max_chunk_chars_each']))}")
    bundle = "\n\n".join(blocks)

    abs_part = ""
    if abstract:
        abs_part = f"【论文摘要】\n{trunc(abstract, int(CONFIG['max_abstract_chars']))}\n\n"

    header_path = str(qa.get("header_path") or (ctx[0].get("header_path") if ctx else "") or "")
    return (tpl
            .replace("{abstract_part}", abs_part)
            .replace("{header_path}", header_path)
            .replace("{chunk_bundle}", bundle)
            .replace("{question}", str(qa.get("question", "")))
            .replace("{answer}", str(qa.get("answer", ""))))


def call_llm(prompt: str) -> str:
    retries = int(CONFIG.get("retries", 0))
    base = float(CONFIG.get("retry_base_sleep", 1.0))
    mx = float(CONFIG.get("retry_max_sleep", 15.0))

    for att in range(retries + 1):
        try:
            return chat_once(
                prompt,
                host=CONFIG["host"],
                model=CONFIG["model"],
                temperature=float(CONFIG["temperature"]),
                max_tokens=int(CONFIG["max_tokens"]),
            )
        except Exception as e:  # noqa
            msg = str(e).lower()
            retryable = any(x in msg for x in ["error code: 502", "error code: 503", "error code: 504", "timeout", "timed out"])
            if att < retries and retryable:
                time.sleep(min(mx, base * (2 ** att)) * (0.85 + 0.3 * random.random()))
                continue
            raise


def parse_result(raw: str) -> Optional[Dict[str, Any]]:
    raw = raw.strip()
    raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    m = re.search(r"<result>(.*?)</result>", raw, re.DOTALL | re.IGNORECASE)
    payload = (m.group(1) if m else raw).strip()
    if not payload.startswith("{"):
        m2 = re.search(r"\{.*\}", payload, re.DOTALL)
        if m2:
            payload = m2.group(0).strip()

    try:
        data = json.loads(payload)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    decision = str(data.get("decision") or "drop").strip().lower()
    if decision not in {"pass", "fix", "drop"}:
        decision = "drop"

    fb = data.get("feedback")
    if isinstance(fb, str):
        fb = [fb]
    if not isinstance(fb, list):
        fb = []

    ev = data.get("evidence_spans") or []
    if isinstance(ev, dict):
        ev = [ev]
    if isinstance(ev, str):
        ev = [{"chunk_id": "", "quote": ev}]
    ev2 = []
    if isinstance(ev, list):
        for x in ev:
            if isinstance(x, dict):
                cid = str(x.get("chunk_id") or "").strip()
                qt = str(x.get("quote") or "").strip()
                if qt:
                    ev2.append({"chunk_id": cid, "quote": qt})

    revised = data.get("revised_qa")
    if not (isinstance(revised, dict) and isinstance(revised.get("question"), str) and isinstance(revised.get("answer"), str)):
        revised = None

    return {
        "decision": decision,
        "reason": str(data.get("reason") or "").strip(),
        "feedback": [str(x).strip() for x in fb if str(x).strip()],
        "evidence_spans": ev2[: int(CONFIG["max_evidence"])],
        "revised_qa": revised,
    }


def _regex_from_quote(q: str) -> str:
    parts = []
    for ch in q:
        if ch.isspace():
            parts.append(r"\s+")
        elif ch in DASHES:
            parts.append("[" + re.escape(DASHES) + "]")
        elif ch in APOS:
            parts.append("[" + re.escape(APOS) + "]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def match_quote(text: str, quote: str) -> Optional[str]:
    if not quote or not text:
        return None

    # 允许用 ... 省略：用首尾片段在同一 chunk 内夹取原文
    if "..." in quote:
        parts = [p.strip() for p in quote.split("...") if p.strip()]
        if len(parts) >= 2:
            a, b = parts[0], parts[-1]
            ia = text.find(a)
            if ia != -1:
                ib = text.find(b, ia + len(a))
                if ib != -1:
                    return text[ia: ib + len(b)]

    if quote in text:
        return quote

    pat = _regex_from_quote(quote)
    m = re.search(pat, text, flags=re.IGNORECASE)
    return m.group(0) if m else None


def resolve_evidence(docs, src: str, ctx_ids: List[str], main_cid: str, evidence: List[Dict[str, str]]):
    """
    在全文 chunks 中定位 quote，回填真实 chunk_id 与原文子串。
    返回 (cleaned, outside_context, errors)
    """
    d = docs.get(src)
    if not d:
        return [], True, ["source_file not found in docs"]

    order: List[str] = d["order"]
    chunks: Dict[str, Any] = d["chunks"]
    ctx_set = set(ctx_ids)

    cleaned = []
    errors = []
    outside_context = False

    # 搜索优先级：提示 chunk_id -> context chunks -> 全文 chunks
    for e in evidence:
        hint = (e.get("chunk_id") or "").strip()
        quote = (e.get("quote") or "").strip()
        if not quote:
            continue

        def try_ids(ids: List[str]) -> Optional[Tuple[str, str]]:
            for cid in ids:
                c = chunks.get(cid)
                if not c:
                    continue
                hit = match_quote(str(c.get("content") or ""), quote)
                if hit:
                    return cid, hit
            return None

        found = None
        if hint and hint in chunks:
            found = try_ids([hint])
        if not found:
            found = try_ids(ctx_ids or [main_cid])
        if not found:
            found = try_ids(order)

        if not found:
            errors.append(f"quote not found in any chunk: {quote[:80]}")
            continue

        cid, hit = found
        if cid not in ctx_set:
            outside_context = True
        cleaned.append({"chunk_id": cid, "quote": hit})

    return cleaned[: int(CONFIG["max_evidence"])], outside_context, errors


def key_of(x: Dict[str, Any]) -> Tuple[str, str, str]:
    return (str(x.get("source_file") or ""), str(x.get("chunk_id") or ""), str(x.get("question") or ""))


def main():
    qa_path = r_in(CONFIG["qa_path"])
    chunks_path = r_in(CONFIG["chunks_path"])
    out_path = r_out(CONFIG["output_path"])
    tpl_path = r_in(CONFIG["prompt_path"])

    qa_list: List[Dict[str, Any]] = jload(qa_path)
    if int(CONFIG.get("limit", 0)) > 0:
        qa_list = qa_list[: int(CONFIG["limit"])]

    docs = build_docs(jload(chunks_path))
    abs_map = load_abstracts(CONFIG.get("abstracts_path", ""))
    tpl = tpl_path.read_text(encoding="utf-8")

    results: List[Dict[str, Any]] = []
    idx: Dict[Tuple[str, str, str], int] = {}
    done = set()

    if CONFIG.get("resume") and out_path.exists():
        try:
            old = jload(out_path)
            if isinstance(old, list):
                results = old
                for i, it in enumerate(results):
                    if isinstance(it, dict):
                        k = key_of(it)
                        idx[k] = i
                        reason = (it.get("judge") or {}).get("reason")
                        if reason not in FAIL_REASONS:
                            done.add(k)
        except Exception:
            results, idx, done = [], {}, set()

    pbar = tqdm(qa_list, desc="Judging", unit="qa")
    for qa in pbar:
        k = key_of(qa)
        if CONFIG.get("resume") and k in done:
            continue

        src, cid, _ = k
        ctx = get_context_chunks(docs, src, cid)
        ctx_ids = [str(c.get("chunk_id")) for c in ctx]

        raw = ""
        try:
            prompt = make_prompt(tpl, qa, ctx, abs_map.get(src, ""))
            raw = call_llm(prompt)
            parsed = parse_result(raw) or {
                "decision": "drop", "reason": "judge_output_parse_error",
                "feedback": [], "evidence_spans": [], "revised_qa": None
            }
        except Exception as e:
            raw = f"[judge_call_error] {e}"
            parsed = {"decision": "drop", "reason": "judge_call_error", "feedback": [str(e)], "evidence_spans": [], "revised_qa": None}

        cleaned, outside_ctx, ev_errors = resolve_evidence(docs, src, ctx_ids, cid, parsed.get("evidence_spans", []))
        parsed["evidence_spans"] = cleaned
        parsed["evidence_outside_context"] = outside_ctx
        if ev_errors:
            parsed.setdefault("feedback", [])
            parsed["feedback"] = parsed["feedback"] + ev_errors

        if CONFIG.get("strict_evidence", True) and parsed["decision"] in {"pass", "fix"} and not cleaned:
            parsed["decision"] = "drop"
            parsed["reason"] = "evidence_validation_failed"
            parsed["revised_qa"] = None

        out_item = dict(qa)
        out_item["judge"] = parsed
        if CONFIG.get("keep_raw"):
            out_item["judge_raw"] = raw

        if k in idx:
            results[idx[k]] = out_item
        else:
            idx[k] = len(results)
            results.append(out_item)

        done.add(k)
        jsave(out_path, results)
        pbar.set_postfix({"decision": parsed["decision"], "reason": parsed["reason"]})

    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
