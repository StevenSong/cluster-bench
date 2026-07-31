#!/usr/bin/env python3
"""
Run one OpenAI-compatible endpoint over MedXpertQA and dump raw answers.

Usage: python evaluate.py --base-url [URL] --model [MODEL-NAME] --out [OUT.jsonl]
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any, Literal, Sequence, Type

from datasets import load_dataset
from openai import APIError, APITimeoutError, LengthFinishReasonError, OpenAI
from pydantic import BaseModel, create_model


def multiple_choice_model(letters: Sequence[str]) -> Type[BaseModel]:
    return create_model(
        "MultipleChoice",
        answer=(Literal[tuple(letters)], ...),
    )


def make_prompt_and_choices(x: dict) -> tuple[str, list[str]]:
    return (
        dedent(f"""\
            Answer the following multiple choice question:
            {x["question"]}

            Your final answer must be just the letter choice."""),
        list(x["options"].keys()),
    )


@dataclass
class Record:
    index: int
    item_id: Any
    pred_answer: str | None
    true_answer: Any
    finish_reason: str | None
    error: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    reasoning: str | None = None


def run_item(
    client: OpenAI, index: int, item: dict, args: argparse.Namespace
) -> Record:
    prompt, choices = make_prompt_and_choices(item)
    true_answer = item.get(args.label_field)
    item_id = item.get("id")

    kwargs: dict[str, Any] = dict(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=args.max_tokens,
        temperature=args.temperature,
        response_format=multiple_choice_model(choices),
    )
    if args.seed is not None:
        kwargs["seed"] = args.seed

    for attempt in range(args.retries + 1):
        try:
            resp = client.chat.completions.parse(**kwargs)
        except LengthFinishReasonError as e:
            usage = getattr(getattr(e, "completion", None), "usage", None)
            return Record(
                index=index,
                item_id=item_id,
                pred_answer=None,
                true_answer=true_answer,
                finish_reason="length",
                error="LengthFinishReasonError",
                prompt_tokens=getattr(usage, "prompt_tokens", None),
                completion_tokens=getattr(usage, "completion_tokens", None),
            )
        except (APIError, APITimeoutError) as e:
            if attempt < args.retries:
                time.sleep(min(2**attempt, 10))
                continue
            return Record(
                index=index,
                item_id=item_id,
                pred_answer=None,
                true_answer=true_answer,
                finish_reason=None,
                error=f"{type(e).__name__}: {e}",
                prompt_tokens=None,
                completion_tokens=None,
            )

        choice = resp.choices[0]
        parsed = choice.message.parsed
        return Record(
            index=index,
            item_id=item_id,
            pred_answer=parsed.answer if parsed is not None else None,
            true_answer=true_answer,
            finish_reason=choice.finish_reason,
            error=None,
            prompt_tokens=resp.usage.prompt_tokens if resp.usage else None,
            completion_tokens=resp.usage.completion_tokens if resp.usage else None,
            reasoning=(
                getattr(choice.message, "reasoning", None)
                if args.save_reasoning
                else None
            ),
        )

    raise AssertionError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="NONE")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dataset", default="/opt/gpudata/MedXpertQA")
    ap.add_argument("--config", default="Text")
    ap.add_argument("--split", default="test")
    ap.add_argument("--label-field", default="label")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--save-reasoning", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(args.dataset, name=args.config, split=args.split)
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Loaded {len(ds)} items | {args.model} @ {args.base_url}")

    client = OpenAI(base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    records: list[Record | None] = [None] * len(ds)

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(run_item, client, i, ds[i], args): i for i in range(len(ds))
        }
        done = 0
        for fut in as_completed(futures):
            records[futures[fut]] = fut.result()
            done += 1
            if done % args.log_every == 0 or done == len(ds):
                print(f"  {done}/{len(ds)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in records:
            assert r is not None
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
