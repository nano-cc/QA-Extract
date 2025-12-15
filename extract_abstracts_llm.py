"""Use vLLM server to extract abstracts from markdown papers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from qa_extract.clients.vllm_client import chat_once


def collect_markdown_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.md") if p.is_file()])


def build_prompt(snippet: str) -> str:
    return (
        "You are an assistant that extracts the Abstract section from a research paper.\n"
        "Instructions:\n"
        "- The following is up to the first N characters of an OCR'd paper (may include noise).\n"
        "- If you see an Abstract/摘要/summary section, return its text exactly.\n"
        "- If there is no explicit heading, return the opening summary/intro paragraph as the abstract.\n"
        "- If you cannot find any abstract-like content, return an empty string.\n"
        "- Do not add any labels, preface, or explanations. Output abstract text only.\n"
        "----- PAPER SNIPPET START -----\n"
        f"{snippet}\n"
        "----- PAPER SNIPPET END -----"
    )


def extract_with_llm(
    md_path: Path,
    *,
    host: str,
    model: str,
    temperature: float,
    max_tokens: int,
    max_chars: int,
) -> str:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    snippet = text[:max_chars]
    prompt = build_prompt(snippet)
    return chat_once(
        prompt,
        host=host,
        model=model,
        temperature=temperature,
        # max_tokens=max_tokens,
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract abstracts using vLLM from markdown papers."
    )
    parser.add_argument(
        "--input-dir",
        default="/mnt/Data/projs/fucking_drug/英文论文",
        help="Root directory containing markdown papers.",
    )
    parser.add_argument(
        "--output",
        default="/mnt/Data/projs/fucking_drug/qa_extract/output/abstracts_english_llm.json",
        help="Path to write JSON output.",
    )
    parser.add_argument("--host", default="http://219.216.98.15:8000/v1", help="vLLM server base URL.")
    parser.add_argument("--model", default="qwen2.5-72b-awq", help="Model name registered on the server.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--max-tokens", type=int, default=12288, help="Max tokens to generate.")
    parser.add_argument("--max-chars", type=int, default=6000, help="Number of leading characters to send.")
    args = parser.parse_args()

    root = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    for md_file in collect_markdown_files(root):
        try:
            abstract = extract_with_llm(
                md_file,
                host=args.host,
                model=args.model,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                max_chars=args.max_chars,
            )
        except Exception as exc:  # noqa: BLE001
            abstract = ""
            print(f"[WARN] Failed on {md_file}: {exc}")
        if abstract:
            records.append({"source_file": str(md_file), "abstract": abstract})

    output_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Extracted {len(records)} abstracts from {len(collect_markdown_files(root))} files -> {output_path}")


if __name__ == "__main__":
    main()
