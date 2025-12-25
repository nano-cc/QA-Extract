#!/usr/bin/env python3
"""Minimal vLLM OpenAI-compatible chat client."""
import argparse

from openai import OpenAI

DEFAULT_HOST = "http://219.216.98.15:8000/v1"
DEFAULT_MODEL = "qwen2.5-72b-awq"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 128


def create_client(host: str = DEFAULT_HOST) -> OpenAI:
    """Create an OpenAI SDK client that points to the local vLLM server."""
    return OpenAI(base_url=host, api_key="EMPTY")


def chat_once(
    prompt: str,
    *,
    host: str = DEFAULT_HOST,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> str:
    """Send a single chat request and return the text content."""
    client = create_client(host)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return resp.choices[0].message.content


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a sample chat request to a vLLM OpenAI-compatible server."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Base URL of the server.")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="Model name registered on the server."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        dest="max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max tokens to generate.",
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="你好，请介绍下自己",
        help="User prompt to send to the model.",
    )
    args = parser.parse_args()

    content = chat_once(
        args.prompt,
        host=args.host,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    print(content)


if __name__ == "__main__":
    main()
