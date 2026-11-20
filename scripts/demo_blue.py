#!/usr/bin/env python3
"""Interactive Demo CLI for Blue Agent (Chặng 6 — Demo cả lớp tấn công Blue).

Usage (from repo root):
  python scripts/demo_blue.py
  python scripts/demo_blue.py --prompt "What is the savings interest rate?"

- Uses the exact CP2/CP3 Blue production pipeline (RateLimit -> Input -> Output).
- Never prints .env or API keys.
- Type 'reset' to clear the rate-limit window, or 'exit' / 'quit' to stop.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.agent import create_blue_agent  # noqa: E402
from agents.guards_agent import check_secret_leak  # noqa: E402
from assignment.pipeline import build_production_plugins  # noqa: E402
from assignment.rate_limiter import RateLimitPlugin  # noqa: E402
from core.utils import chat_with_agent  # noqa: E402
from guardrails.input_guardrails import InputGuardrailPlugin  # noqa: E402
from guardrails.output_guardrails import OutputGuardrailPlugin  # noqa: E402


def _find(plugins: list, cls):
    for p in plugins:
        if isinstance(p, cls):
            return p
    return None


async def send_one(agent, runner, plugins: list, prompt: str) -> None:
    rl = _find(plugins, RateLimitPlugin)
    ig = _find(plugins, InputGuardrailPlugin)
    og = _find(plugins, OutputGuardrailPlugin)

    rl_before = rl.blocked_count if rl else 0
    in_before = ig.blocked_count if ig else 0
    out_blk_before = og.blocked_count if og else 0
    out_red_before = og.redacted_count if og else 0

    response = ""
    transient = ("429", "500", "502", "503", "504", "rate", "timeout", "provider")
    for attempt in range(3):
        try:
            response, _ = await chat_with_agent(agent, runner, prompt)
            break
        except Exception as exc:
            if attempt < 2 and any(t in str(exc).lower() for t in transient):
                await asyncio.sleep(2.0 * (attempt + 1))
                continue
            print(f"[API ERROR] {type(exc).__name__} (không tính là lộ secret)")
            return

    if rl and rl.blocked_count > rl_before:
        layer = "rate_limiter (BLOCKED)"
    elif ig and ig.blocked_count > in_before:
        reason = getattr(ig, "last_block_reason", None) or "input"
        layer = f"input_guardrail ({reason})"
    elif og and og.blocked_count > out_blk_before:
        layer = "output_guardrail (BLOCKED)"
    elif og and og.redacted_count > out_red_before:
        layer = "output_guardrail (REDACTED)"
    else:
        layer = "LLM (ALLOWED)"

    leaked = check_secret_leak(response)
    print(f"  Layer  : {layer}")
    print(f"  Leaked : {leaked}")
    print(f"  Reply  : {response}\n")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Blue Agent Demo CLI")
    parser.add_argument("--prompt", type=str, default=None, help="Run one prompt and exit")
    parser.add_argument("--max-requests", type=int, default=15, help="Rate limit per 60s")
    args = parser.parse_args()

    plugins = build_production_plugins(
        max_requests=args.max_requests,
        window_seconds=60,
        use_llm_judge=False,
    )
    agent, runner = create_blue_agent(plugins)

    if args.prompt is not None:
        await send_one(agent, runner, plugins, args.prompt)
        return

    print("=" * 60)
    print("BLUE AGENT LIVE DEMO (gõ 'reset' để reset rate limit, 'exit' để thoát)")
    print("=" * 60)
    rl = _find(plugins, RateLimitPlugin)
    turn = 0
    while True:
        try:
            prompt = input(f"[Turn {turn + 1}] Prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if prompt.lower() in {"exit", "quit", "q"}:
            break
        if prompt.lower() == "reset":
            if rl:
                rl.reset()
            print("Rate limit window reset.\n")
            continue
        turn += 1
        await send_one(agent, runner, plugins, prompt)


if __name__ == "__main__":
    asyncio.run(main())
