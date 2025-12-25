import argparse
import json
import uuid
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
from langchain_text_splitters import MarkdownHeaderTextSplitter

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
    ("####", "Header 4"),
]

markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on)
plt.switch_backend("Agg")


def combine_headers(h1: str, h2: str) -> str:
    def parts(header: str) -> List[str]:
        if not header:
            return []
        return [p for p in header.split(" | ") if p]

    seen = []
    for part in parts(h1) + parts(h2):
        if part not in seen:
            seen.append(part)
    return " | ".join(seen)


def build_chunks_for_file(md_header_splits, min_len: int, max_len: int, overlap_ratio: float) -> List[dict]:
    overlap = int(max_len * overlap_ratio)
    overlap = max(0, min(overlap, max_len - 1))

    chunks: List[dict] = []
    buf_content = ""
    buf_header = ""

    docs = list(md_header_splits)
    for idx, doc in enumerate(docs):
        remaining_docs = len(docs) - idx - 1
        header_parts = []
        for _, header_key in headers_to_split_on:
            value = doc.metadata.get(header_key)
            if value:
                header_parts.append(value)
        header = " ".join(header_parts)

        # Clean special tokens from content.
        cleaned_content = (
            doc.page_content.replace("<｜end▁of▁sentence｜>", "").replace("<--- Page Split --->", "")
        )

        # Append current block into buffer.
        if buf_content:
            buf_content = buf_content.rstrip() + "\n\n" + cleaned_content
            buf_header = combine_headers(buf_header, header)
        else:
            buf_content = cleaned_content
            buf_header = header

        # Emit chunks while buffer is too large.
        while len(buf_content) >= max_len:
            piece = buf_content[:max_len]
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_content": piece,
                    "chunk_len": len(piece),
                    "header": buf_header,
                }
            )
            # Slide window forward, keeping overlap.
            buf_content = buf_content[max_len - overlap :]

        # If buffer is within target band and there is more content ahead, finalize this chunk.
        if remaining_docs > 0 and min_len <= len(buf_content) <= max_len:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_content": buf_content,
                    "chunk_len": len(buf_content),
                    "header": buf_header,
                }
            )
            buf_content = ""
            buf_header = ""

    # Handle tail: avoid overlong merge, allow only final fragment < min_len.
    if buf_content:
        # Split tail further if still too long.
        while len(buf_content) > max_len:
            piece = buf_content[:max_len]
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_content": piece,
                    "chunk_len": len(piece),
                    "header": buf_header,
                }
            )
            buf_content = buf_content[max_len - overlap :]

        if chunks and len(buf_content) < min_len:
            # Try to merge into last chunk only if it stays within max_len.
            last = chunks[-1]
            merged_len = last["chunk_len"] + len(buf_content)
            if merged_len <= max_len:
                merged_content = last["chunk_content"].rstrip() + "\n\n" + buf_content
                last["chunk_content"] = merged_content
                last["chunk_len"] = merged_len
                last["header"] = combine_headers(last["header"], buf_header)
                buf_content = ""

        if buf_content:
            chunks.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "chunk_content": buf_content,
                    "chunk_len": len(buf_content),
                    "header": buf_header,
                }
            )

    return chunks


def main():
    parser = argparse.ArgumentParser(description="Split markdown files into chunks with post-processing.")
    parser.add_argument(
        "--min-len",
        type=int,
        default=800,
        help="Preferred minimum chunk length. Non-final chunks will be at least this long when possible.",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=2000,
        help="Preferred maximum chunk length before splitting.",
    )
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=0.1,
        help="Overlap ratio (0-1) applied when splitting oversized chunks.",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate and save a histogram of chunk length distribution.",
    )
    parser.add_argument(
        "--no-histogram",
        action="store_false",
        dest="histogram",
        help="Disable histogram generation.",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default="/home/cong/data/projs/QA-Extract/英文论文",
        help="Directory containing markdown papers to chunk.",
    )
    parser.set_defaults(histogram=True)
    args = parser.parse_args()

    min_len = max(1, args.min_len)
    max_len = max(min_len, args.max_len)
    overlap_ratio = max(0.0, min(args.overlap_ratio, 0.9))
    papers_dir = Path(args.papers_dir)
    folder_name = papers_dir.name

    # Walk all markdown files and flatten metadata + page content.
    json_rows = []
    for md_path in sorted(papers_dir.glob("*.md")):
        content = md_path.read_text(encoding="utf-8")
        md_header_splits = markdown_splitter.split_text(content)
        json_rows.extend(
            build_chunks_for_file(
                md_header_splits,
                min_len=min_len,
                max_len=max_len,
                overlap_ratio=overlap_ratio,
            )
        )

    if json_rows:
        lengths = [row["chunk_len"] for row in json_rows]
        max_chunk = max(json_rows, key=lambda row: row["chunk_len"])
        min_chunk = min(json_rows, key=lambda row: row["chunk_len"])
        max_len_val = max_chunk["chunk_len"]
        min_len_val = min_chunk["chunk_len"]
        avg_len = sum(lengths) / len(lengths)
        print(
            "Chunk length stats - "
            f"max: {max_len_val} (chunk_id={max_chunk['chunk_id']}), "
            f"min: {min_len_val} (chunk_id={min_chunk['chunk_id']}), "
            f"avg: {avg_len:.2f}"
        )
        if args.histogram:
            hist_path = Path("/home/cong/data/projs/QA-Extract/output") / f"{folder_name}.png"
            plt.figure(figsize=(10, 6))
            plt.hist(lengths, bins=30, color="#4B8BBE", edgecolor="white")
            plt.title("Chunk Length Distribution")
            plt.xlabel("Chunk length (chars)")
            plt.ylabel("Frequency")
            plt.tight_layout()
            hist_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(hist_path)
            print(f"Wrote histogram to {hist_path}")
    else:
        print("No chunks produced; length stats unavailable.")

    output_path = Path("/home/cong/data/projs/QA-Extract/output") / f"{folder_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(json_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(json_rows)} items to {output_path}")


if __name__ == "__main__":
    main()
