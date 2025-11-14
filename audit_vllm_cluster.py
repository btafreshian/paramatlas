#!/usr/bin/env python3
"""Utility for auditing models served via vLLM or Hugging Face ids."""

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import requests
import torch
from torch import nn
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer

try:  # AutoModelForImageTextToText was introduced in newer transformers releases
    from transformers import AutoModelForImageTextToText
except Exception:  # pragma: no cover - older transformers do not expose this auto class
    AutoModelForImageTextToText = None  # type: ignore
from transformers.utils import is_torch_bf16_gpu_available

# accelerate is only needed for --fast
try:
    from accelerate import init_empty_weights

    ACCEL_AVAILABLE = True
except Exception:
    ACCEL_AVAILABLE = False


logger = logging.getLogger("audit_vllm_cluster")


# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000"),
        help="vLLM/OpenAI-compatible base URL (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Optional HF model id or local path to skip endpoint discovery "
            "(e.g., Qwen/Qwen3-VL-30B-A3B-Instruct)"
        ),
    )
    parser.add_argument(
        "--no-hooks",
        action="store_true",
        help="Disable tiny one-shot hooks sanity check (full mode only)",
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "full"],
        default=None,
        help="Select run mode explicitly: fast (empty weights) or full (load real weights)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="FAST: build skeleton & counts from config with empty weights (no real weight load)",
    )
    parser.add_argument(
        "--outdir",
        default="reports",
        help="Directory to create report folder in",
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Also emit summary.json and modules.json alongside existing reports",
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        choices=["quiet", "normal", "verbose"],
        default="normal",
        help="Logging verbosity: quiet, normal, or verbose (default: normal)",
    )
    return parser.parse_args()


def configure_logging(verbosity: str) -> None:
    """Configure logging according to verbosity level."""

    verbosity_levels = {
        "quiet": logging.WARNING,
        "normal": logging.INFO,
        "verbose": logging.DEBUG,
    }
    log_level = verbosity_levels.get(verbosity, logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    logger.setLevel(log_level)


def resolve_fast_mode(args: argparse.Namespace) -> bool:
    """Resolve whether FAST mode should be used."""

    if args.mode == "fast":
        return True
    if args.mode == "full":
        return False
    return bool(args.fast)


# -----------------------------
# Endpoint helpers
# -----------------------------

def get_models_from_endpoint(base_url: str) -> Dict[str, Any]:
    """Query `/v1/models` from the provided endpoint."""

    url = base_url.rstrip("/") + "/v1/models"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def pick_model_id(models_payload: Dict[str, Any]) -> str:
    """Select the first non-embedding model id from an endpoint payload."""

    data = models_payload.get("data", [])
    if not data:
        raise RuntimeError("Endpoint returned no models in /v1/models")
    for model in data:
        model_id = model.get("id", "")
        if model_id and "embed" not in model_id.lower():
            return model_id
    return data[0].get("id", "")


# -----------------------------
# MoE heuristics
# -----------------------------


@dataclass
class Row:
    """Flat representation of module metadata used for reporting."""

    name: str
    cls: str
    parent: str
    depth: int
    n_params: int
    n_trainable: int
    n_buffers: int
    is_router: bool
    is_expert: bool
    moe_group: str


def write_modules_json(report_dir: str, rows: Iterable[Row]) -> None:
    """Write modules.json mirroring modules.csv order and columns."""

    modules_path = os.path.join(report_dir, "modules.json")
    payload: List[Dict[str, Any]] = []
    for row in rows:
        payload.append(
            {
                "name": row.name,
                "class": row.cls,
                "parent": row.parent,
                "depth": row.depth,
                "n_params": row.n_params,
                "n_trainable": row.n_trainable,
                "n_buffers": row.n_buffers,
                "is_router": bool(row.is_router),
                "is_expert": bool(row.is_expert),
                "moe_group": row.moe_group,
            }
        )
    with open(modules_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def write_summary_json(
    report_dir: str,
    *,
    model_id: str,
    source: Dict[str, str],
    mode_label: str,
    integrity_info: Dict[str, Any],
) -> None:
    """Write summary.json with compact summary data."""

    summary_path = os.path.join(report_dir, "summary.json")
    totals: Dict[str, Optional[int]] = {
        "parameters": integrity_info.get("csv_total"),
    }
    if integrity_info.get("csv_train") is not None:
        totals["trainable_parameters"] = integrity_info.get("csv_train")

    moe_groups_payload: List[Dict[str, Any]] = []
    for entry in integrity_info.get("moe_groups", []):
        moe_groups_payload.append(
            {
                "name": entry.get("name"),
                "experts": entry.get("experts"),
                "routers": entry.get("routers"),
                "parameters": entry.get("parameters"),
            }
        )

    payload: Dict[str, Any] = {
        "model_id": model_id,
        "mode": mode_label,
        "source": source,
        "totals": totals,
        "integrity": {
            "status": integrity_info.get("integrity_status", "UNKNOWN"),
            "reasons": integrity_info.get("integrity_reasons", []),
        },
        "moe_groups": moe_groups_payload,
    }
    if integrity_info.get("model_total") is not None:
        payload["model_parameter_total"] = integrity_info.get("model_total")

    payload["integrity"]["bad_groups"] = sorted(
        list(integrity_info.get("bad_groups", {}).keys())
    )

    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


MOE_ROUTER_KEYS = ("router", "gate", "gating", "topk", "switch", "score", "routing")
MOE_EXPERT_KEYS = ("expert", "experts", "ffn_expert", "moe", "sparse", "mixture")
MOE_ROUTER_CLASS_HINTS = (
    "Router",
    "TopKGate",
    "SwitchRouter",
    "TopKRouter",
    "MoERouter",
)
MOE_EXPERT_CLASS_HINTS = (
    "Moe",
    "MoE",
    "Sparse",
    "Experts",
    "Expert",
    "Mixture",
)


def _normalized_architectures(cfg: Optional[AutoConfig]) -> List[str]:
    """Return lowercase architecture names from a config, if any."""

    if cfg is None:
        return []
    arch_list = getattr(cfg, "architectures", None) or []
    return [arch.lower() for arch in arch_list]


def _is_vision_language_config(cfg: Optional[AutoConfig]) -> bool:
    """Detect configs that describe VLM / image-text models."""

    if cfg is None:
        return False
    if getattr(cfg, "vision_config", None):
        return True
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    if any(key in model_type for key in ("vision", "vl", "imagetext", "multimodal")):
        return True
    archs = _normalized_architectures(cfg)
    return any(
        any(key in arch for key in ("vision", "vl", "image", "imagetext"))
        for arch in archs
    )


def _has_causal_generation_head(cfg: Optional[AutoConfig]) -> bool:
    """Detect configs whose recommended classes expose a generation head."""

    if cfg is None:
        return False
    archs = _normalized_architectures(cfg)
    return any(
        any(token in arch for token in ("conditionalgeneration", "causallm"))
        for arch in archs
    )


def preferred_model_loaders(cfg: Optional[AutoConfig]) -> List[Any]:
    """Return the ordered list of AutoModel classes to try for a given config."""

    loaders: List[Any] = []
    if _is_vision_language_config(cfg) and AutoModelForImageTextToText is not None:
        loaders.append(AutoModelForImageTextToText)
    if _has_causal_generation_head(cfg):
        loaders.append(AutoModelForCausalLM)
    # AutoModel remains the universal fallback and should always be tried.
    loaders.append(AutoModel)

    # Remove duplicates while preserving order.
    deduped: List[Any] = []
    for loader in loaders:
        if loader not in deduped:
            deduped.append(loader)
    return deduped


def format_int(value: int) -> str:
    """Return a human-friendly formatted integer."""

    return f"{value:,}"


def get_parent(name: str) -> str:
    """Return the dotted parent name for a module."""

    return "" if "." not in name else name.rsplit(".", 1)[0]


def depth_of(name: str) -> int:
    """Return the depth of a dotted module name."""

    return 0 if not name else name.count(".")


def guess_is_router(name: str, cls: str) -> bool:
    """Heuristically determine whether the module appears to be a router."""

    lower_name = name.lower()
    if any(key in lower_name for key in MOE_ROUTER_KEYS):
        return True
    if any(key.lower() in cls.lower() for key in MOE_ROUTER_CLASS_HINTS):
        return True
    return False


def guess_is_expert(name: str, cls: str) -> bool:
    """Heuristically determine whether the module appears to be an MoE expert."""

    lower_name = name.lower()
    if any(key in lower_name for key in MOE_EXPERT_KEYS):
        return True
    if any(key.lower() in cls.lower() for key in MOE_EXPERT_CLASS_HINTS):
        return True
    return False


def detect_structural_moe_groups(
    named_modules: Iterable[Tuple[str, nn.Module]],
    class_chain: Dict[str, str],
) -> Set[str]:
    """Infer parent blocks that own both routers and experts."""

    children_by_parent: Dict[str, List[str]] = {}
    for name, _ in named_modules:
        parent = get_parent(name)
        if not parent:
            continue
        children_by_parent.setdefault(parent, []).append(name)

    structural_groups: Set[str] = set()
    for parent, children in children_by_parent.items():
        has_router_child = any(
            guess_is_router(child, class_chain.get(child, "")) for child in children
        )
        has_expert_child = any(
            guess_is_expert(child, class_chain.get(child, "")) for child in children
        )
        if has_router_child and has_expert_child:
            # Qwen-style VL MoE blocks place routers and experts under siblings
            # (e.g., layers.N.mlp.gate + layers.N.mlp.experts). Treating their
            # shared parent as the moe_group keeps the structure intact.
            structural_groups.add(parent)
    return structural_groups


def nearest_moe_group(
    name: str, class_chain: Dict[str, str], structural_moe_groups: Set[str]
) -> str:
    """Return the nearest ancestor considered an MoE group."""

    parent = get_parent(name)
    while parent:
        cls = class_chain.get(parent, "")
        if (
            parent in structural_moe_groups
            or guess_is_expert(parent, cls)
            or "Moe" in cls
            or "MoE" in cls
            or "Sparse" in cls
        ):
            return parent
        parent = get_parent(parent)
    return ""


def model_skeleton(named_modules: Iterable[Tuple[str, nn.Module]]) -> str:
    """Return a textual representation of the model skeleton."""

    out = io.StringIO()
    for name, module in named_modules:
        indent = "  " * depth_of(name)
        cls = module.__class__.__name__
        out.write(f"{indent}{name or '<root>'} :: {cls}\n")
    return out.getvalue()


# -----------------------------
# Validation helpers
# -----------------------------


def build_class_chain(named_modules: Iterable[Tuple[str, nn.Module]]) -> Dict[str, str]:
    """Map module names to their class names."""

    return {name: module.__class__.__name__ for name, module in named_modules}


def build_rows(
    named_modules: Iterable[Tuple[str, nn.Module]],
    class_chain: Dict[str, str],
) -> Tuple[List[Row], int, int]:
    """Construct Row entries and aggregate parameter totals."""

    rows: List[Row] = []
    total_params = 0
    total_train = 0
    structural_moe_groups = detect_structural_moe_groups(named_modules, class_chain)
    for name, module in named_modules:
        cls = module.__class__.__name__
        parent = get_parent(name)
        depth = depth_of(name)
        n_params = sum(param.numel() for param in module.parameters(recurse=False))
        n_trainable = sum(
            param.numel() for param in module.parameters(recurse=False) if param.requires_grad
        )
        n_buffers = sum(buffer.numel() for buffer in module.buffers(recurse=False))
        total_params += n_params
        total_train += n_trainable
        is_router = guess_is_router(name, cls)
        is_expert = guess_is_expert(name, cls)
        moe_group = nearest_moe_group(name, class_chain, structural_moe_groups)
        rows.append(
            Row(
                name=name,
                cls=cls,
                parent=parent,
                depth=depth,
                n_params=n_params,
                n_trainable=n_trainable,
                n_buffers=n_buffers,
                is_router=is_router,
                is_expert=is_expert,
                moe_group=moe_group,
            )
        )
    return rows, total_params, total_train


def write_modules_csv(report_dir: str, rows: Iterable[Row]) -> str:
    """Write modules.csv and return its path."""

    path = os.path.join(report_dir, "modules.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                "name",
                "class",
                "parent",
                "depth",
                "n_params",
                "n_trainable",
                "n_buffers",
                "is_router",
                "is_expert",
                "moe_group",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.name,
                    row.cls,
                    row.parent,
                    row.depth,
                    row.n_params,
                    row.n_trainable,
                    row.n_buffers,
                    int(row.is_router),
                    int(row.is_expert),
                    row.moe_group,
                ]
            )
    return path


def write_router_and_expert_csvs(report_dir: str, rows: Iterable[Row]) -> Tuple[str, str]:
    """Write routers.csv and experts.csv, returning their paths."""

    routers_path = os.path.join(report_dir, "routers.csv")
    experts_path = os.path.join(report_dir, "experts.csv")

    with open(routers_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "class", "moe_group"])
        for row in rows:
            if row.is_router:
                writer.writerow([row.name, row.cls, row.moe_group])

    with open(experts_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["name", "class", "parent", "moe_group"])
        for row in rows:
            if row.is_expert:
                writer.writerow([row.name, row.cls, row.parent, row.moe_group])

    return routers_path, experts_path


def write_skeleton(report_dir: str, named_modules: Iterable[Tuple[str, nn.Module]]) -> str:
    """Write skeleton.txt and return its path."""

    path = os.path.join(report_dir, "skeleton.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(model_skeleton(named_modules))
    return path


def write_summary_text(
    report_dir: str,
    *,
    endpoint: str,
    model_id: str,
    rows: Iterable[Row],
    total_params: int,
    total_train: int,
    mode_label: str,
) -> str:
    """Write summary.txt and return its path."""

    header = "==== vLLM Cluster Audit Summary ===="
    if mode_label.upper() == "FAST":
        header = "==== vLLM Cluster Audit Summary (FAST) ===="

    rows_list = list(rows)
    n_modules = len(rows_list)
    n_routers = sum(1 for row in rows_list if row.is_router)
    n_experts = sum(1 for row in rows_list if row.is_expert)

    path = os.path.join(report_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"{header}\n")
        fh.write(f"Endpoint           : {endpoint}\n")
        fh.write(f"Model ID           : {model_id}\n")
        fh.write(f"Modules            : {n_modules:,}\n")
        fh.write(f"Total params       : {format_int(total_params)}\n")
        fh.write(f"Trainable params   : {format_int(total_train)}\n")
        fh.write(f"Routers detected   : {n_routers:,}\n")
        fh.write(f"Experts detected   : {n_experts:,}\n")
        fh.write(f"Reports directory  : {report_dir}\n")
    return path


def write_standard_artifacts(
    report_dir: str,
    *,
    endpoint: str,
    model_id: str,
    rows: List[Row],
    named_modules: Iterable[Tuple[str, nn.Module]],
    total_params: int,
    total_train: int,
    mode_label: str,
) -> Dict[str, str]:
    """Write core report files shared between FAST and FULL modes."""

    modules_path = write_modules_csv(report_dir, rows)
    routers_path, experts_path = write_router_and_expert_csvs(report_dir, rows)
    skeleton_path = write_skeleton(report_dir, named_modules)
    summary_path = write_summary_text(
        report_dir,
        endpoint=endpoint,
        model_id=model_id,
        rows=rows,
        total_params=total_params,
        total_train=total_train,
        mode_label=mode_label,
    )
    return {
        "modules": modules_path,
        "routers": routers_path,
        "experts": experts_path,
        "skeleton": skeleton_path,
        "summary": summary_path,
    }


def write_integrity_and_moe_reports(
    report_dir: str,
    rows: Iterable[Row],
    mode_label: str,
    model_params_full: Optional[int] = None,
) -> Dict[str, Any]:
    """Write validation details and append integrity information to summary.txt."""

    rows_list = list(rows)
    sum_csv_params = sum(row.n_params for row in rows_list)
    sum_csv_train = sum(row.n_trainable for row in rows_list)

    groups: Dict[str, Dict[str, int]] = {}
    for row in rows_list:
        moe_group = row.moe_group or "(none)"
        if moe_group not in groups:
            groups[moe_group] = {"experts": 0, "routers": 0, "params": 0}
        groups[moe_group]["params"] += row.n_params
        if row.is_expert:
            groups[moe_group]["experts"] += 1
        if row.is_router:
            groups[moe_group]["routers"] += 1

    bad_groups: Dict[str, Dict[str, int]] = {}
    for group_name, stats in groups.items():
        if group_name == "(none)":
            continue
        if stats["routers"] == 0 or stats["experts"] == 0:
            bad_groups[group_name] = stats

    validation_path = os.path.join(report_dir, "validation_report.txt")
    sorted_groups = sorted(
        groups.items(), key=lambda kv: kv[1]["params"], reverse=True
    )
    with open(validation_path, "w", encoding="utf-8") as fh:
        fh.write(f"==== Validation Report ({mode_label}) ====\n")
        fh.write(f"CSV param total     : {format_int(sum_csv_params)}\n")
        fh.write(f"CSV trainable total : {format_int(sum_csv_train)}\n\n")
        fh.write("MoE group summary (top 50 by params):\n")
        for name, stats in sorted_groups[:50]:
            fh.write(
                f"[moe] {name:70s} params={format_int(stats['params'])} "
                f"experts={stats['experts']:>4} routers={stats['routers']:>3}\n"
            )
        if bad_groups:
            fh.write("\n[warn] Incomplete MoE groups detected (missing router or experts):\n")
            for name, stats in bad_groups.items():
                fh.write(
                    f"  - {name}: experts={stats['experts']} routers={stats['routers']}\n"
                )
        if model_params_full is not None:
            delta = abs(model_params_full - sum_csv_params)
            fh.write("\nModel total vs CSV total:\n")
            fh.write(f"  Model param total : {format_int(model_params_full)}\n")
            fh.write(f"  CSV param total   : {format_int(sum_csv_params)}\n")
            fh.write(f"  Delta             : {format_int(delta)}\n")

    summary_path = os.path.join(report_dir, "summary.txt")
    status = "PASS"
    reasons: List[str] = []
    with open(summary_path, "a", encoding="utf-8") as fh:
        fh.write(f"CSV param total     : {format_int(sum_csv_params)}\n")
        fh.write(f"CSV trainable total : {format_int(sum_csv_train)}\n")
        if bad_groups:
            status = "FAIL"
            reasons.append(f"{len(bad_groups)} incomplete MoE group(s)")
        if model_params_full is not None:
            delta = abs(model_params_full - sum_csv_params)
            fh.write(f"Model param total   : {format_int(model_params_full)}\n")
            fh.write(f"Param delta         : {format_int(delta)}\n")
            if delta != 0:
                status = "FAIL"
                reasons.append("model vs CSV param mismatch")
        if reasons:
            fh.write(f"INTEGRITY STATUS    : {status}  ({'; '.join(reasons)})\n")
        else:
            fh.write("INTEGRITY STATUS    : PASS\n")

    moe_group_entries = [
        {
            "name": name,
            "experts": stats["experts"],
            "routers": stats["routers"],
            "parameters": stats["params"],
        }
        for name, stats in sorted_groups
    ]

    return {
        "csv_total": sum_csv_params,
        "csv_train": sum_csv_train,
        "bad_groups": bad_groups,
        "model_total": model_params_full,
        "integrity_status": status,
        "integrity_reasons": reasons[:],
        "moe_groups": moe_group_entries,
    }


# -----------------------------
# Hooks (one-shot, full mode)
# -----------------------------


def attach_example_hooks(
    model: nn.Module,
    example_routers: List[str],
    example_experts: List[str],
    log_writer,
) -> List[Any]:
    """Attach one-shot hooks to representative router/expert modules."""

    handles: List[Any] = []

    def once(fn):
        fired = {"value": False}

        def wrapper(*args, **kwargs):
            if not fired["value"]:
                fired["value"] = True
                return fn(*args, **kwargs)

        return wrapper

    if example_routers:
        module = dict(model.named_modules()).get(example_routers[0])
        if module is not None:

            def router_hook(mod, inputs, outputs):
                with torch.no_grad():
                    try:
                        tensor = outputs[0] if isinstance(outputs, tuple) else outputs
                        log_writer(
                            f"[router:{example_routers[0]}] out shape: {tuple(tensor.shape)}"
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        log_writer(f"[router:{example_routers[0]}] hook error: {exc}")

            handles.append(module.register_forward_hook(once(router_hook)))

    if example_experts:
        module = dict(model.named_modules()).get(example_experts[0])
        if module is not None:

            def expert_hook(mod, inputs, outputs):
                with torch.no_grad():
                    try:
                        tensor = outputs if isinstance(outputs, torch.Tensor) else outputs[0]
                        mean = tensor.float().mean().item()
                        std = tensor.float().std().item()
                        log_writer(
                            f"[expert:{example_experts[0]}] out mean={mean:.4f} "
                            f"std={std:.4f} shape={tuple(tensor.shape)}"
                        )
                    except Exception as exc:  # pragma: no cover - defensive
                        log_writer(f"[expert:{example_experts[0]}] hook error: {exc}")

            handles.append(module.register_forward_hook(once(expert_hook)))

    return handles


# -----------------------------
# Mode runners
# -----------------------------


def write_endpoint_payload(report_dir: str, endpoint: str, payload: Dict[str, Any]) -> None:
    """Persist the endpoint payload for later inspection."""

    path = os.path.join(report_dir, "endpoint.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"endpoint": endpoint, "models_payload": payload}, fh, indent=2)


def run_fast_mode(
    cfg: AutoConfig,
    *,
    report_dir: str,
    endpoint: str,
    model_id: str,
    source_info: Dict[str, str],
    json_output: bool,
) -> Dict[str, Any]:
    """Execute FAST mode using empty weights."""

    if not ACCEL_AVAILABLE:
        logger.error("[error] FAST mode requires 'accelerate' (pip install accelerate)")
        sys.exit(1)

    logger.info("[info] FAST mode: building skeleton with empty weights (no real weight loading)")
    with init_empty_weights():
        tmp_model = None
        last_exc: Optional[Exception] = None
        loader_chain = preferred_model_loaders(cfg)
        for loader in loader_chain:
            loader_name = getattr(loader, "__name__", str(loader))
            try:
                tmp_model = loader.from_config(cfg, trust_remote_code=True)
                logger.debug(
                    "FAST mode instantiated skeleton with %s", loader_name
                )
                break
            except Exception as exc:  # pragma: no cover - defensive fallback
                last_exc = exc
                logger.warning(
                    "[warn] %s.from_config failed (%s); trying next candidate ...",
                    loader_name,
                    exc,
                )
        if tmp_model is None:
            logger.error(
                "[error] Failed to instantiate model from config after trying %s",
                ", ".join(getattr(ld, "__name__", str(ld)) for ld in loader_chain),
            )
            if last_exc:
                raise last_exc
            sys.exit(1)

    named_modules = list(tmp_model.named_modules())
    class_chain = build_class_chain(named_modules)
    rows, total_params, total_train = build_rows(named_modules, class_chain)

    artifacts = write_standard_artifacts(
        report_dir,
        endpoint=endpoint,
        model_id=model_id,
        rows=rows,
        named_modules=named_modules,
        total_params=total_params,
        total_train=total_train,
        mode_label="FAST",
    )

    integrity_info = write_integrity_and_moe_reports(
        report_dir=report_dir,
        rows=rows,
        mode_label="FAST",
        model_params_full=None,
    )

    if json_output:
        try:
            write_modules_json(report_dir, rows)
            write_summary_json(
                report_dir,
                model_id=model_id,
                source=source_info,
                mode_label="FAST",
                integrity_info=integrity_info,
            )
        except Exception as exc:
            logger.error("[error] Failed to write JSON outputs: %s", exc)
            sys.exit(1)

    logger.info("[done][FAST] Wrote reports to: %s", report_dir)
    logger.info(
        "[done][FAST] Key files: %s, %s, %s, %s, %s",
        artifacts["modules"],
        artifacts["skeleton"],
        artifacts["routers"],
        artifacts["experts"],
        artifacts["summary"],
    )
    return integrity_info


def run_full_mode(
    *,
    report_dir: str,
    endpoint: str,
    model_id: str,
    source_info: Dict[str, str],
    json_output: bool,
    no_hooks: bool,
    cfg: Optional[AutoConfig],
) -> Dict[str, Any]:
    """Execute FULL mode by loading real model weights."""

    if torch.cuda.is_available() and is_torch_bf16_gpu_available():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    device_map: Any
    if torch.cuda.is_available():
        device_map = "auto"
    else:
        device_map = {"": "cpu"}

    logger.info(
        "[info] Loading model '%s' (dtype=%s, device_map=%s) ...",
        model_id,
        dtype,
        device_map,
    )
    model = None
    last_exc: Optional[Exception] = None
    loader_chain = preferred_model_loaders(cfg)
    for loader in loader_chain:
        loader_name = getattr(loader, "__name__", str(loader))
        try:
            model = loader.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=device_map,
                trust_remote_code=True,
            )
            logger.info("[info] Loaded weights with %s", loader_name)
            break
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_exc = exc
            logger.warning(
                "[warn] %s.from_pretrained failed (%s); trying next candidate ...",
                loader_name,
                exc,
            )
    if model is None:
        logger.error(
            "[error] Failed to load model '%s' after trying %s",
            model_id,
            ", ".join(getattr(ld, "__name__", str(ld)) for ld in loader_chain),
        )
        if last_exc:
            raise last_exc
        sys.exit(1)
    model.eval()

    named_modules = list(model.named_modules())
    class_chain = build_class_chain(named_modules)
    rows, total_params, total_train = build_rows(named_modules, class_chain)

    artifacts = write_standard_artifacts(
        report_dir,
        endpoint=endpoint,
        model_id=model_id,
        rows=rows,
        named_modules=named_modules,
        total_params=total_params,
        total_train=total_train,
        mode_label="FULL",
    )

    model_params = sum(param.numel() for param in model.parameters())
    integrity_info = write_integrity_and_moe_reports(
        report_dir=report_dir,
        rows=rows,
        mode_label="FULL",
        model_params_full=model_params,
    )

    if not no_hooks:
        hook_log = os.path.join(report_dir, "hook_log.txt")

        def log_writer(message: str) -> None:
            with open(hook_log, "a", encoding="utf-8") as fh:
                fh.write(message + "\n")
            logger.info(message)

        example_routers = [row.name for row in rows if row.is_router][:1]
        example_experts = [row.name for row in rows if row.is_expert][:1]
        handles = attach_example_hooks(model, example_routers, example_experts, log_writer)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )
        except Exception:
            tokenizer = None
        if tokenizer is not None:
            device0 = next(model.parameters()).device
            encoded = tokenizer(
                "A short test prompt for model introspection.",
                return_tensors="pt",
            ).to(device0)
            with torch.no_grad():
                model.generate(**encoded, max_new_tokens=8)
        for handle in handles:
            with contextlib.suppress(Exception):
                handle.remove()

    if json_output:
        try:
            write_modules_json(report_dir, rows)
            write_summary_json(
                report_dir,
                model_id=model_id,
                source=source_info,
                mode_label="FULL",
                integrity_info=integrity_info,
            )
        except Exception as exc:
            logger.error("[error] Failed to write JSON outputs: %s", exc)
            sys.exit(1)

    logger.info("[done] Wrote reports to: %s", report_dir)
    logger.info(
        "[done] Key files: %s, %s, %s, %s, %s, validation_report.txt",
        artifacts["modules"],
        artifacts["skeleton"],
        artifacts["routers"],
        artifacts["experts"],
        artifacts["summary"],
    )
    return integrity_info


def handle_integrity_failure(mode_label: str, integrity_info: Dict[str, Any]) -> None:
    """Exit if integrity checks failed for the provided mode."""

    status = (integrity_info or {}).get("integrity_status", "UNKNOWN")
    if status != "FAIL":
        return
    reasons = (integrity_info or {}).get("integrity_reasons", [])
    reason_text = f" ({'; '.join(reasons)})" if reasons else ""
    logger.error("[fail] Integrity checks failed in %s mode%s", mode_label, reason_text)
    sys.exit(1)


# -----------------------------
# Main
# -----------------------------


def create_report_dir(outdir: str, model_id: str) -> str:
    """Create the timestamped report directory for the run."""

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_id = re.sub(r"[^a-zA-Z0-9_.\\-]+", "_", model_id)
    report_dir = os.path.join(outdir, f"{timestamp}_{safe_model_id}")
    try:
        os.makedirs(report_dir, exist_ok=True)
    except OSError as exc:
        logger.error("[error] Failed to create report directory '%s': %s", report_dir, exc)
        sys.exit(1)
    return report_dir


def main() -> None:
    """Entry point for running the vLLM cluster audit."""

    args = parse_args()
    configure_logging(args.verbosity)
    fast_mode = resolve_fast_mode(args)
    logger.debug(
        "Resolved mode: fast_mode=%s (mode arg=%s, legacy --fast=%s)",
        fast_mode,
        args.mode,
        args.fast,
    )

    models_payload: Optional[Dict[str, Any]] = None
    if args.model:
        model_id = args.model
    else:
        try:
            models_payload = get_models_from_endpoint(args.endpoint)
        except Exception as exc:
            logger.error("[error] Failed to query endpoint '%s': %s", args.endpoint, exc)
            sys.exit(1)
        try:
            model_id = pick_model_id(models_payload)
        except Exception as exc:
            logger.error("[error] Failed to pick a model id from endpoint payload: %s", exc)
            sys.exit(1)
        if not model_id:
            logger.error("[error] Could not infer a model id from endpoint payload.")
            sys.exit(1)

    if args.model:
        source_info = {"type": "model_argument", "value": args.model}
    else:
        source_info = {"type": "endpoint_discovery", "endpoint": args.endpoint}

    report_dir = create_report_dir(args.outdir, model_id)

    if models_payload is not None:
        write_endpoint_payload(report_dir, args.endpoint, models_payload)

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        cfg = None

    if fast_mode:
        if cfg is None:
            logger.error(
                "[error] FAST mode requested but AutoConfig failed; cannot proceed."
            )
            sys.exit(1)
        integrity_info = run_fast_mode(
            cfg,
            report_dir=report_dir,
            endpoint=args.endpoint,
            model_id=model_id,
            source_info=source_info,
            json_output=args.json_output,
        )
        handle_integrity_failure("FAST", integrity_info)
        return

    integrity_info = run_full_mode(
        report_dir=report_dir,
        endpoint=args.endpoint,
        model_id=model_id,
        source_info=source_info,
        json_output=args.json_output,
        no_hooks=args.no_hooks,
        cfg=cfg,
    )
    handle_integrity_failure("FULL", integrity_info)


if __name__ == "__main__":
    main()
