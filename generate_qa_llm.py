"""Use vLLM to generate Chinese QA pairs from chunked papers."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from qa_extract.clients.vllm_client import chat_once


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def build_abstract_map(records: List[Dict[str, Any]]) -> Dict[str, str]:
    return {rec["source_file"]: rec["abstract"] for rec in records if rec.get("abstract")}


def build_prompt(template: str, chunk: Dict[str, Any], abstract: Optional[str]) -> str:
    """Fill the external prompt template with chunk-specific content."""
    abstract_part = f"【论文摘要】\n{abstract}\n\n" if abstract else ""
    replacements = {
        "{abstract_part}": abstract_part,
        "{chunk.get('header_path','')}": chunk.get("header_path", ""),
        "{chunk['content']}": chunk.get("content", ""),
    }
    prompt = template
    for key, value in replacements.items():
        prompt = prompt.replace(key, value)
    return prompt


def parse_qa(json_text: str) -> List[Dict[str, str]]:
    match = re.search(r"<result>(.*?)</result>", json_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    cleaned = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = item.get("question")
        a = item.get("answer")
        if isinstance(q, str) and isinstance(a, str) and q.strip() and a.strip():
            cleaned.append({"question": q.strip(), "answer": a.strip()})
    return cleaned


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate QA pairs with vLLM.")
    parser.add_argument(
        "--chunks",
        default="output/chunks_english.json",
        help="Path to chunk JSON.",
    )
    parser.add_argument(
        "--abstracts",
        default="output/abstracts_english_llm.json",
        help="Path to abstracts JSON.",
    )
    parser.add_argument(
        "--output",
        default="output/qa_pairs_english_llm_test.json",
        help="Where to write QA pairs JSON.",
    )
    parser.add_argument(
        "--prompt-file",
        default="prompt.txt",
        help="Path to the prompt template text file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of chunks to process (<=0 means use all chunks).",
    )
    parser.add_argument("--host", default="http://219.216.98.15:8000/v1", help="vLLM server base URL.")
    parser.add_argument("--model", default="qwen2.5-72b-awq", help="Model name registered on the server.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=12288, help="Max tokens to generate.")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)
    abstracts_path = Path(args.abstracts)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    chunks = load_json(chunks_path)
    if args.limit > 0:
        chunks = chunks[: args.limit]
    abstracts = build_abstract_map(load_json(abstracts_path))
    prompt_template = Path(args.prompt_file).read_text(encoding="utf-8")

    results: List[Dict[str, Any]] = []
    pbar = tqdm(chunks, desc="Generating QA", unit="chunk")
    for chunk in pbar:
        source = chunk.get("source_file")
        chunk_id = chunk.get("chunk_id")
        prompt = build_prompt(prompt_template, chunk, abstracts.get(source))
        try:
            llm_text = chat_once(
                prompt,
                host=args.host,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            qa_items = parse_qa(llm_text)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Failed on {source} {chunk_id}: {exc}")
            qa_items = []
        count_before = len(results)
        for qa in qa_items:
            results.append(
                {
                    "source_file": source,
                    "chunk_id": chunk_id,
                    "header_path": chunk.get("header_path", ""),
                    "question": qa["question"],
                    "answer": qa["answer"],
                }
            )
        produced = len(results) - count_before
        pbar.set_postfix({"qa_total": len(results), "qa_added": produced})

    output_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated {len(results)} QA items from {len(chunks)} chunks -> {output_path}")


if __name__ == "__main__":
    main()
