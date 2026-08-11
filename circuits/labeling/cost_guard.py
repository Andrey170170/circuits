"""Conservative pre-submit cost guard for generic labeling runs."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from circuits.analysis.bonafide.canonical import canonical_sha256, file_sha256
from circuits.labeling.config import LabelingRecipe
from circuits.labeling.io import atomic_write_json
from circuits.labeling.pricing import load_price_snapshot
from circuits.labeling.runtime import load_run_manifest, load_stage_requests

COST_PLAN_SCHEMA = "adag.labeling.pre-submit-cost-plan.v1"
PLAN_FILE = "pre-submit-cost-plan.json"


def _message_input_upper_bound(request: Any) -> int:
    """Bound tokenizer input by UTF-8 bytes plus explicit chat framing overhead."""

    payload = json.dumps(
        [message.model_dump(mode="json") for message in request.messages],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return len(payload) + 64 * len(request.messages) + 256


def _cost(
    snapshot: dict[str, Any],
    *,
    provider: str,
    model: str,
    transport: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    try:
        rates = snapshot["rates"][provider][model][transport]
        input_rate = float(rates["input_per_million"])
        output_rate = float(rates["output_per_million"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"missing price for {provider}/{model}/{transport}") from error
    return (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000


def _compute_pre_submit_cost_plan(
    *, run_root: Path, max_cumulative_cost_usd: float
) -> dict[str, Any]:
    if not math.isfinite(max_cumulative_cost_usd) or max_cumulative_cost_usd <= 0:
        raise ValueError("max cumulative cost must be finite and positive")
    manifest = load_run_manifest(run_root)
    recipe = LabelingRecipe.model_validate(manifest["recipe"])
    if recipe.prompt_policy != "hybrid_candidate_v1":
        raise ValueError("hybrid cost guard requires hybrid_candidate_v1")
    price_path = Path(manifest["price_snapshot_path"])
    if file_sha256(price_path) != manifest["price_snapshot_sha256"]:
        raise ValueError("price snapshot hash mismatch")
    snapshot = load_price_snapshot(price_path)
    candidate_requests = load_stage_requests(run_root, "candidate_generation")
    if not candidate_requests:
        raise ValueError("candidate request set is empty")

    candidate_input = sum(
        _message_input_upper_bound(item) for item in candidate_requests
    )
    candidate_output = sum(int(item.max_output_tokens) for item in candidate_requests)
    candidate_role = recipe.candidate_generator
    candidate_transport = candidate_requests[0].transport
    if any(item.transport != candidate_transport for item in candidate_requests):
        raise ValueError("candidate request transports are mixed")
    candidate_cost = _cost(
        snapshot,
        provider=candidate_role.provider,
        model=candidate_role.model,
        transport=candidate_transport,
        input_tokens=candidate_input,
        output_tokens=candidate_output,
    )

    by_cluster: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for request in candidate_requests:
        by_cluster[(request.state, request.cluster_id)].append(request)
    summary_input = 0
    for requests in by_cluster.values():
        # Candidate output tokens are charged again as future summary input. Four copies of
        # the exact generation prompt plus fixed framing conservatively reserve room for the
        # additional selection witnesses and summary instructions. The exact prepared summary
        # request is checked against this ceiling before its provider submission.
        summary_input += (
            4 * max(_message_input_upper_bound(item) for item in requests)
            + sum(int(item.max_output_tokens) for item in requests)
            + 8192
        )
    summary_output = len(by_cluster) * recipe.cluster_summarizer.max_output_tokens
    summary_role = recipe.cluster_summarizer
    summary_transport = summary_role.transport
    summary_cost = _cost(
        snapshot,
        provider=summary_role.provider,
        model=summary_role.model,
        transport=summary_transport,
        input_tokens=summary_input,
        output_tokens=summary_output,
    )
    projected = candidate_cost + summary_cost
    plan: dict[str, Any] = {
        "schema_version": COST_PLAN_SCHEMA,
        "run_manifest_sha256": manifest["manifest_sha256"],
        "price_snapshot_id": snapshot["snapshot_id"],
        "price_snapshot_sha256": manifest["price_snapshot_sha256"],
        "estimator": (
            "native_batch_one_attempt_utf8_byte_proxy_plus_exact_prepared_stage_gate.v2"
        ),
        "retry_policy": "external_live_and_explicit_retries_forbidden",
        "max_cumulative_cost_usd": max_cumulative_cost_usd,
        "projected_upper_bound_usd": projected,
        "within_cap": projected <= max_cumulative_cost_usd,
        "stages": {
            "candidate_generation": {
                "request_count": len(candidate_requests),
                "input_token_upper_bound": candidate_input,
                "output_token_upper_bound": candidate_output,
                "cost_upper_bound_usd": candidate_cost,
            },
            "cluster_summary": {
                "request_count": len(by_cluster),
                "input_token_upper_bound": summary_input,
                "output_token_upper_bound": summary_output,
                "cost_upper_bound_usd": summary_cost,
            },
        },
    }
    plan["plan_sha256"] = canonical_sha256(plan)
    return plan


def build_pre_submit_cost_plan(
    *, run_root: Path, max_cumulative_cost_usd: float
) -> dict[str, Any]:
    """Persist a conservative planning ceiling; this does not submit provider work."""

    plan = _compute_pre_submit_cost_plan(
        run_root=run_root, max_cumulative_cost_usd=max_cumulative_cost_usd
    )
    atomic_write_json(run_root / PLAN_FILE, plan)
    if not plan["within_cap"]:
        raise ValueError(
            f"projected API cost ${plan['projected_upper_bound_usd']:.6f} exceeds "
            f"${max_cumulative_cost_usd:.6f} cap"
        )
    return plan


def load_pre_submit_cost_plan(
    *, run_root: Path, stage: str | None = None
) -> dict[str, Any]:
    """Deep-validate the persisted hybrid plan against the current run and stage."""

    path = run_root / PLAN_FILE
    if not path.is_file():
        raise ValueError("hybrid provider submission requires a persisted cost plan")
    try:
        observed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("hybrid cost plan is unreadable") from error
    if not isinstance(observed, dict):
        raise TypeError("hybrid cost plan must be an object")
    core = dict(observed)
    recorded_hash = core.pop("plan_sha256", None)
    if recorded_hash != canonical_sha256(core):
        raise ValueError("hybrid cost plan hash mismatch")
    if observed.get("schema_version") != COST_PLAN_SCHEMA:
        raise ValueError("unsupported hybrid cost plan schema")
    maximum = observed.get("max_cumulative_cost_usd")
    if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
        raise TypeError("hybrid cost plan maximum is invalid")
    expected = _compute_pre_submit_cost_plan(
        run_root=run_root, max_cumulative_cost_usd=float(maximum)
    )
    if observed != expected:
        raise ValueError("hybrid cost plan differs from the current run")
    projected = observed.get("projected_upper_bound_usd")
    if (
        observed.get("within_cap") is not True
        or isinstance(projected, bool)
        or not isinstance(projected, (int, float))
        or not math.isfinite(float(projected))
        or float(projected) > float(maximum)
    ):
        raise ValueError("hybrid cost plan does not authorize spend within its cap")
    if stage is not None:
        if stage not in {"candidate_generation", "cluster_summary"}:
            raise ValueError(f"unsupported cost-guard stage: {stage}")
        requests = load_stage_requests(run_root, stage)
        planned = observed["stages"][stage]
        actual_input = sum(_message_input_upper_bound(item) for item in requests)
        actual_output = sum(int(item.max_output_tokens) for item in requests)
        if (
            len(requests) != planned["request_count"]
            or actual_input > planned["input_token_upper_bound"]
            or actual_output > planned["output_token_upper_bound"]
        ):
            raise ValueError(
                f"prepared {stage} requests exceed the hybrid cost-plan ceiling"
            )
    return observed
