"""Hierarchical chunking for OCR Markdown papers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass
class Section:
    level: int
    title: str
    header_path: str
    content_lines: List[str]


@dataclass
class Chunk:
    chunk_id: str
    header_path: str
    content: str
    token_count: int


HEADING_HASH = re.compile(r"^\s*(#{1,6})\s*(.+)$")
HEADING_NUM = re.compile(r"^\s*(\d+(?:\.\d+)*)[)\.\s]+\s*(.+)$")
HEADING_ROMAN = re.compile(r"^\s*[IVXLCM]+\.\s+(.+)$", re.IGNORECASE)


def is_pseudo_heading(line: str) -> bool:
    """Heuristics for page numbers/footers."""
    stripped = line.strip()
    if not stripped:
        return True
    if re.match(r"^\d{1,3}$", stripped):
        return True
    if len(stripped) < 3 and not re.match(r"^\d", stripped):
        return True
    return False


def _numeric_depth(title: str) -> int:
    """Infer depth from a leading number pattern like 3.1.2."""
    m = re.match(r"^\s*(\d+(?:\.\d+)*)\b", title)
    if not m:
        return 0
    return m.group(1).count(".") + 1


def parse_heading(line: str) -> Optional[tuple[int, str]]:
    """Return (level, title) if the line looks like a heading.

    OCR 噪声下所有标题可能都只有一个 `#`，因此使用数字序号推断层级深度。
    """
    m = HEADING_HASH.match(line)
    if m:
        title = m.group(2).strip()
        depth = _numeric_depth(title)
        level = depth if depth > 0 else len(m.group(1))
        return level, title
    m = HEADING_NUM.match(line)
    if m:
        level = m.group(1).count(".") + 1
        return level, f"{m.group(1)} {m.group(2).strip()}"
    m = HEADING_ROMAN.match(line)
    if m:
        return 1, m.group(0).strip()
    return None


def token_count(text: str) -> int:
    """Approximate token count for mixed Chinese/English."""
    tokens = re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9]+", text)
    return max(1, len(tokens))


def split_sentences(text: str) -> List[str]:
    parts = re.split(r"([。！？!?])", text)
    sentences: List[str] = []
    for i in range(0, len(parts), 2):
        sent = parts[i]
        if i + 1 < len(parts):
            sent += parts[i + 1]
        if sent.strip():
            sentences.append(sent)
    return sentences or [text]


def build_sections(lines: Iterable[str]) -> List[Section]:
    sections: List[Section] = []
    header_stack: List[tuple[int, str]] = []
    current_lines: List[str] = []

    def flush():
        if not current_lines:
            return
        path = " > ".join(title for _, title in header_stack) or "ROOT"
        sections.append(
            Section(
                level=header_stack[-1][0] if header_stack else 0,
                title=header_stack[-1][1] if header_stack else "ROOT",
                header_path=path,
                content_lines=current_lines.copy(),
            )
        )

    for line in lines:
        heading = parse_heading(line)
        if heading and not is_pseudo_heading(line):
            flush()
            level, title = heading
            while header_stack and header_stack[-1][0] >= level:
                header_stack.pop()
            header_stack.append((level, title))
            current_lines = [line.rstrip("\n")]
        else:
            if not header_stack:
                header_stack.append((1, "Preface"))
            current_lines.append(line.rstrip("\n"))
    flush()
    return sections


def merge_small_sections(
    sections: List[Section], min_tokens: int
) -> List[Section]:
    merged: List[Section] = []
    for sec in sections:
        text = "\n".join(sec.content_lines).strip()
        if token_count(text) < min_tokens and merged:
            merged[-1].content_lines.append("")
            merged[-1].content_lines.extend(sec.content_lines)
        else:
            merged.append(sec)
    return merged


def split_large_content(
    header_path: str,
    content: str,
    max_tokens: int,
    overlap_ratio: float,
) -> List[str]:
    sentences = split_sentences(content)
    chunks: List[str] = []
    start = 0
    overlap_tokens = max(1, int(max_tokens * overlap_ratio))
    while start < len(sentences):
        tokens = 0
        end = start
        while end < len(sentences):
            seg_tokens = token_count(sentences[end])
            if tokens + seg_tokens > max_tokens and end > start:
                break
            tokens += seg_tokens
            end += 1
        chunk_text = "".join(sentences[start:end]).strip()
        if chunk_text:
            chunks.append(chunk_text)
        # compute new start with overlap
        if end >= len(sentences):
            break
        back_tokens = 0
        back = end - 1
        while back > start and back_tokens < overlap_tokens:
            back_tokens += token_count(sentences[back])
            back -= 1
        start = max(back, start + 1)
    return chunks


def chunk_markdown(
    markdown_text: str,
    *,
    min_tokens: int = 50,
    max_tokens: int = 1000,
    overlap_ratio: float = 0.18,
) -> List[Chunk]:
    lines = markdown_text.splitlines()
    sections = build_sections(lines)
    sections = merge_small_sections(sections, min_tokens)

    chunks: List[Chunk] = []
    counter = 1
    for sec in sections:
        content = "\n".join(sec.content_lines).strip()
        if not content:
            continue
        base_tokens = token_count(content)
        if base_tokens > max_tokens:
            for piece in split_large_content(
                sec.header_path, content, max_tokens, overlap_ratio
            ):
                chunks.append(
                    Chunk(
                        chunk_id=f"chunk_{counter:04d}",
                        header_path=sec.header_path,
                        content=piece,
                        token_count=token_count(piece),
                    )
                )
                counter += 1
        else:
            chunks.append(
                Chunk(
                    chunk_id=f"chunk_{counter:04d}",
                    header_path=sec.header_path,
                    content=content,
                    token_count=base_tokens,
                )
            )
            counter += 1
    return chunks


__all__ = ["Chunk", "chunk_markdown"]
