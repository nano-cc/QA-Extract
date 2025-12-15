#!/usr/bin/env python3
"""Send a single chat request using the contents of a text file as the prompt."""
from __future__ import annotations

from pathlib import Path

from qa_extract.clients.vllm_client import (
    DEFAULT_HOST,
    DEFAULT_MODEL,
    DEFAULT_TEMPERATURE,
    chat_once,
)


def main() -> None:
    text_file = "/mnt/Data/projs/QA-Extract/test.txt"
    prompt = Path(text_file).read_text(encoding="utf-8")
    content = chat_once(
        prompt,
        host=DEFAULT_HOST,
        model=DEFAULT_MODEL,
        temperature=DEFAULT_TEMPERATURE,
        max_tokens=2048
    )
    print(content)


if __name__ == "__main__":
    main()
