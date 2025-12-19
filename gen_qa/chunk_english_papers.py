"""CLI to chunk English markdown papers under 英文论文 and write JSON."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import List

from hierarchical_chunker import Chunk, chunk_markdown, split_large_content, token_count


def collect_markdown_files(root: Path) -> List[Path]:
    return sorted([p for p in root.rglob("*.md") if p.is_file()])


def _merge_small_chunks(chunks: List[dict], min_merged_tokens: int) -> List[dict]:
    merged_chunks: List[dict] = []
    index = 0
    while index < len(chunks):
        current = chunks[index]
        merged_content = [current["content"]]
        merged_tokens = current["token_count"]
        index += 1

        while merged_tokens < min_merged_tokens and index < len(chunks):
            candidate = chunks[index]
            merged_content.append(candidate["content"])
            merged_tokens += candidate["token_count"]
            index += 1

        merged_chunk = dict(current)
        merged_chunk["chunk_id"] = f"chunk_{uuid.uuid4().hex}"
        merged_chunk["header_path"] = [] if isinstance(current["header_path"], (list, tuple)) else ""
        merged_chunk["content"] = "\n\n".join(merged_content)
        merged_chunk["token_count"] = merged_tokens
        merged_chunks.append(merged_chunk)

    return merged_chunks


def _split_oversized_chunks(
    chunks: List[dict],
    max_tokens: int,
    overlap_ratio: float,
) -> List[dict]:
    split_chunks: List[dict] = []
    for chunk in chunks:
        if chunk["token_count"] <= max_tokens:
            split_chunks.append(chunk)
            continue
        for piece in split_large_content("", chunk["content"], max_tokens, overlap_ratio):
            split_chunk = dict(chunk)
            split_chunk["chunk_id"] = f"chunk_{uuid.uuid4().hex}"
            split_chunk["content"] = piece
            split_chunk["token_count"] = token_count(piece)
            split_chunks.append(split_chunk)
    return split_chunks


def chunk_file(
    md_path: Path,
    min_tokens: int,
    max_tokens: int,
    overlap_ratio: float,
    min_merged_tokens: int,
) -> List[dict]:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    chunks = chunk_markdown(
        text,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_ratio=overlap_ratio,
    )
    chunk_dicts = [
        {
            "chunk_id": chunk.chunk_id,
            "header_path": [] if isinstance(chunk.header_path, (list, tuple)) else "",
            "content": chunk.content,
            "token_count": chunk.token_count,
            "source_file": str(md_path),
        }
        for chunk in chunks
    ]
    merged = _merge_small_chunks(chunk_dicts, min_merged_tokens)
    return _split_oversized_chunks(merged, max_tokens, overlap_ratio)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk markdown files under 英文论文 and save as JSON."
    )
    parser.add_argument(
        "--input-dir",
        default="../英文论文",
        help="Root directory containing markdown papers.",
    )
    parser.add_argument(
        "--output",
        default="./output/chunks_english.json",
        help="Path to write JSON output.",
    )
    parser.add_argument("--min-tokens", type=int, default=50, help="Small-section merge threshold.")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Maximum tokens per chunk.")
    parser.add_argument(
        "--min-merge-tokens",
        type=int,
        default=1000,
        help="Minimum tokens per merged chunk after hierarchical chunking.",
    )
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
        file_chunks = chunk_file(
            md_file,
            args.min_tokens,
            args.max_tokens,
            args.overlap,
            args.min_merge_tokens,
        )
        all_chunks.extend(file_chunks)

    output_path.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Chunked {len(all_chunks)} segments from {len(collect_markdown_files(root))} files -> {output_path}")


if __name__ == "__main__":
    main()
