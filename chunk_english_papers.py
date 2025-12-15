"""CLI to chunk English markdown papers under 英文论文 and write JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from qa_extract.pipeline.chunking.hierarchical_chunker import Chunk, chunk_markdown


def collect_markdown_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.md") if p.is_file()])


def chunk_file(md_path: Path, min_tokens: int, max_tokens: int, overlap_ratio: float) -> List[dict]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_markdown(
        text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
    )
    return [
        {
            "chunk_id": chunk.chunk_id,
            "header_path": chunk.header_path,
            "content": chunk.content,
            "token_count": chunk.token_count,
            "source_file": str(md_path),
        }
        for chunk in chunks
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files under 英文论文 and save as JSON."
    )
    parser.add_argument(
        "--input-dir",
        default="./英文论文",
        help="Root directory containing markdown papers.",
    )
    parser.add_argument(
        "--output",
        default="./output/chunks_english.json",
        help="Path to write JSON output.",
    )
    parser.add_argument("--min-tokens", type=int, default=50, help="Small-section merge threshold.")
    parser.add_argument("--max-tokens", type=int, default=1000, help="Maximum tokens per chunk.")
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.18,
        help="Overlap ratio for long-section sliding window.",
    )
    args = parser.parse_args()

    root = Path(args.input_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[dict] = []
    for md_file in collect_markdown_files(root):
        file_chunks = chunk_file(md_file, args.min_tokens, args.max_tokens, args.overlap)
        all_chunks.extend(file_chunks)

    output_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunked {len(all_chunks)} segments from {len(collect_markdown_files(root))} files -> {output_path}")


if __name__ == "__main__":
    main()
