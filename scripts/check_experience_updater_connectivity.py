#!/usr/bin/env python3
"""Check whether ExperienceUpdater can reach an OpenAI-compatible model service."""

import argparse
import json
import os
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ExperienceUpdater connectivity to an OpenAI-compatible "
            "chat-completions endpoint."
        )
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("EXPERIENCE_UPDATE_API_KEY"),
        help="API key. Defaults to EXPERIENCE_UPDATE_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EXPERIENCE_UPDATE_BASE_URL"),
        help="OpenAI-compatible base URL, for example http://host:8000/v1. "
        "Defaults to EXPERIENCE_UPDATE_BASE_URL.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("EXPERIENCE_UPDATE_MODEL"),
        help="Model name. Defaults to EXPERIENCE_UPDATE_MODEL.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("EXPERIENCE_UPDATE_MAX_TOKENS", "256")),
        help="Token budget for the test request.",
    )
    parser.add_argument(
        "--token-limit-param",
        default=os.environ.get("EXPERIENCE_UPDATE_TOKEN_LIMIT_PARAM", "max_tokens"),
        choices=("max_tokens", "max_completion_tokens"),
        help="Token-limit parameter expected by the endpoint.",
    )
    parser.add_argument(
        "--skip-experience-test",
        action="store_true",
        help="Only run a minimal raw chat-completions connectivity check.",
    )
    parser.add_argument(
        "--show-response",
        action="store_true",
        help="Print the raw response text from the minimal connectivity check.",
    )
    return parser.parse_args()


def _validate_required(args: argparse.Namespace) -> None:
    missing = [
        name
        for name, value in (
            ("api key", args.api_key),
            ("base URL", args.base_url),
            ("model", args.model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            "Missing required config: "
            + ", ".join(missing)
            + ". Provide CLI args or EXPERIENCE_UPDATE_API_KEY, "
            "EXPERIENCE_UPDATE_BASE_URL, and EXPERIENCE_UPDATE_MODEL."
        )


def _make_experience_updater(args: argparse.Namespace):
    sys.path.insert(0, str(_repo_root()))
    try:
        from agent_system.memory.experience_updater import ExperienceUpdater
    except ModuleNotFoundError as exc:
        if exc.name == "openai":
            raise RuntimeError(
                "Python package 'openai' is not installed in this environment."
            ) from exc
        raise

    return ExperienceUpdater(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        max_new_experiences_per_update=1,
        max_completion_tokens=args.max_tokens,
        token_limit_param=args.token_limit_param,
    )


def _raw_connectivity_check(updater, args: argparse.Namespace) -> str:
    request_kwargs = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": "Reply with exactly: ok",
            }
        ],
        args.token_limit_param: min(args.max_tokens, 32),
    }
    response = updater.client.chat.completions.create(**request_kwargs)
    return response.choices[0].message.content or ""


def _experience_updater_check(updater) -> list:
    failed_trajectories = [
        {
            "task": "put the apple in the basket",
            "task_type": "pick_and_place",
            "trajectory": [
                {
                    "action": "go to basket",
                    "observation": "You are near the basket but do not have the apple.",
                },
                {
                    "action": "put apple in basket",
                    "observation": "You are not carrying the apple.",
                },
            ],
        }
    ]
    current_experiences = {
        "general_experiences": [
            {
                "experience_id": "base_001",
                "title": "Check Inventory Before Placing",
                "principle": "Before placing an object, verify that it is currently held.",
                "when_to_apply": "When the next action places or uses an object.",
            }
        ],
        "task_specific_experiences": {},
    }
    return updater.analyze_failures(
        failed_trajectories=failed_trajectories,
        current_experiences=current_experiences,
    )


def main() -> int:
    args = _build_args()

    try:
        _validate_required(args)
        print("[ExperienceUpdaterCheck] Config")
        print(f"  base_url: {args.base_url}")
        print(f"  model: {args.model}")
        print(f"  api_key: {_mask_secret(args.api_key)}")
        print(f"  token_limit_param: {args.token_limit_param}")

        updater = _make_experience_updater(args)

        print("[ExperienceUpdaterCheck] Running raw chat-completions check...")
        raw_text = _raw_connectivity_check(updater, args).strip()
        print("[ExperienceUpdaterCheck] Raw connectivity: OK")
        if args.show_response:
            print(f"[ExperienceUpdaterCheck] Raw response: {raw_text}")

        if args.skip_experience_test:
            return 0

        print("[ExperienceUpdaterCheck] Running ExperienceUpdater analyze_failures check...")
        experiences = _experience_updater_check(updater)
        if not experiences:
            print(
                "[ExperienceUpdaterCheck] ExperienceUpdater request completed but produced "
                "no parseable experiences. Connectivity is OK; check model JSON "
                "following ability, max tokens, or prompt behavior."
            )
            return 2

        print("[ExperienceUpdaterCheck] ExperienceUpdater parse: OK")
        print(json.dumps(experiences, ensure_ascii=False, indent=2))
        return 0

    except Exception as exc:
        print(f"[ExperienceUpdaterCheck] FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
