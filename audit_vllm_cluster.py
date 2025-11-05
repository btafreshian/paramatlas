#!/usr/bin/env python3
"""
audit_vllm_cluster.py
---------------------
All-in-one, zero-hyperparameter inspector for a model served via vLLM's
OpenAI-compatible API OR a direct HF id. It can:

1) Discover the model id from your endpoint (or accept --model to skip discovery).
2) FAST mode (recommended for huge models): build the full module skeleton and
   per-module param/buffer counts from config WITHOUT loading real weights
   using accelerate.init_empty_weights (no VRAM moves, very quick).
3) Full mode (optional): load the actual model with transformers (device_map='auto',
   dtype auto), enumerate modules, and (optionally) attach one-shot hooks.

Outputs (in ./reports/<timestamp>_<model_id>/):
- endpoint.json        : endpoint + /v1/models payload (if endpoint used)
- summary.txt          : human summary (+ integrity stamps & PASS/FAIL)
- skeleton.txt         : indented module tree
- modules.csv          : flat table with names/classes/counts + router/expert flags
- routers.csv          : suspected routers
- experts.csv          : suspected experts
- validation_report.txt: MoE consistency & integrity details
- hook_log.txt         : (full mode only) one-liners from tiny forward

Usage:
  # Auto-detect model via endpoint (zero knobs)
  python audit_vllm_cluster.py --endpoint http://localhost:8000 --fast

  # Direct HF id, no endpoint
  python audit_vllm_cluster.py --model Qwen/Qwen3-VL-30B-A3B-Instruct --fast

Deps: transformers, torch, requests, accelerate (FAST mode)
"""

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
from typing import Dict, List, Tuple

import torch
from torch import nn

import requests
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
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

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default=os.environ.get("VLLM_ENDPOINT", "http://localhost:8000"),
                    help="vLLM/OpenAI-compatible base URL (default: %(default)s)")
    ap.add_argument("--model", default=None,
                    help="Optional HF model id or local path to skip endpoint discovery (e.g., Qwen/Qwen3-VL-30B-A3B-Instruct)")
    ap.add_argument("--no-hooks", action="store_true", help="Disable tiny one-shot hooks sanity check (full mode only)")
    ap.add_argument("--mode", choices=["fast", "full"], default=None,
                    help="Select run mode explicitly: fast (empty weights) or full (load real weights)")
    ap.add_argument("--fast", action="store_true", help="FAST: build skeleton & counts from config with empty weights (no real weight load)")
    ap.add_argument("--outdir", default="reports", help="Directory to create report folder in")
    ap.add_argument("--json-output", action="store_true",
                    help="Also emit summary.json and modules.json alongside existing reports")
    ap.add_argument("-v", "--verbosity", choices=["quiet", "normal", "verbose"], default="normal",
                    help="Logging verbosity: quiet, normal, or verbose (default: normal)")
    return ap.parse_args()


# -----------------------------
# Endpoint helpers
# -----------------------------

def get_models_from_endpoint(base_url: str) -> dict:
    url = base_url.rstrip("/") + "/v1/models"
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()

def pick_model_id(models_payload: dict) -> str:
    data = models_payload.get("data", [])
    if not data:
        raise RuntimeError("Endpoint returned no models in /v1/models")
    # Prefer the first non-embedding entry
    for m in data:
        mid = m.get("id", "")
        if mid and "embed" not in mid.lower():
            return mid
    return data[0].get("id", "")


# -----------------------------
# MoE heuristics
# -----------------------------

@dataclass
class Row:
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


def write_modules_json(report_dir: str, rows: List[Row]) -> None:
    """Write modules.json mirroring modules.csv order and columns."""
    modules_path = os.path.join(report_dir, "modules.json")
    payload = []
    for r in rows:
        payload.append({
            "name": r.name,
            "class": r.cls,
            "parent": r.parent,
            "depth": r.depth,
            "n_params": r.n_params,
            "n_trainable": r.n_trainable,
            "n_buffers": r.n_buffers,
            "is_router": bool(r.is_router),
            "is_expert": bool(r.is_expert),
            "moe_group": r.moe_group,
        })
    with open(modules_path, "w") as f:
        json.dump(payload, f, indent=2)


def write_summary_json(
    report_dir: str,
    *,
    model_id: str,
    source: Dict[str, str],
    mode_label: str,
    integrity_info: dict,
) -> None:
    """Write summary.json with compact summary data."""
    summary_path = os.path.join(report_dir, "summary.json")
    totals = {
        "parameters": integrity_info.get("csv_total"),
    }
    if integrity_info.get("csv_train") is not None:
        totals["trainable_parameters"] = integrity_info.get("csv_train")

    moe_groups_payload = []
    for entry in integrity_info.get("moe_groups", []):
        moe_groups_payload.append({
            "name": entry.get("name"),
            "experts": entry.get("experts"),
            "routers": entry.get("routers"),
            "parameters": entry.get("parameters"),
        })

    payload = {
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

    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)

MOE_ROUTER_KEYS = ("router", "gate", "gating", "topk", "switch", "score", "routing")
MOE_EXPERT_KEYS = ("expert", "experts", "ffn_expert", "moe", "sparse", "mixture")
MOE_ROUTER_CLASS_HINTS = ("Router", "TopKGate", "SwitchRouter", "TopKRouter", "MoERouter")
MOE_EXPERT_CLASS_HINTS = ("Moe", "MoE", "Sparse", "Experts", "Expert", "Mixture")

def format_int(n: int) -> str:
    return f"{n:,}"

def get_parent(name: str) -> str:
    return "" if "." not in name else name.rsplit(".", 1)[0]

def depth_of(name: str) -> int:
    return 0 if not name else name.count(".")

def guess_is_router(name: str, cls: str) -> bool:
    nm = name.lower()
    if any(k in nm for k in MOE_ROUTER_KEYS):
        return True
    if any(k.lower() in cls.lower() for k in MOE_ROUTER_CLASS_HINTS):
        return True
    return False

def guess_is_expert(name: str, cls: str) -> bool:
    nm = name.lower()
    if any(k in nm for k in MOE_EXPERT_KEYS):
        return True
    if any(k.lower() in cls.lower() for k in MOE_EXPERT_CLASS_HINTS):
        return True
    return False

def nearest_moe_group(name: str, class_chain: Dict[str, str]) -> str:
    parent = get_parent(name)
    while parent:
        cls = class_chain.get(parent, "")
        if guess_is_expert(parent, cls) or "Moe" in cls or "MoE" in cls or "Sparse" in cls:
            return parent
        parent = get_parent(parent)
    return ""

def model_skeleton(named_modules: List[Tuple[str, nn.Module]]) -> str:
    out = io.StringIO()
    for name, mod in named_modules:
        indent = "  " * depth_of(name)
        cls = mod.__class__.__name__
        out.write(f"{indent}{name or '<root>'} :: {cls}\n")
    return out.getvalue()


# -----------------------------
# Validation helpers (new)
# -----------------------------

def write_integrity_and_moe_reports(
    report_dir: str,
    rows: List[Row],
    mode_label: str,
    model_params_full: int = None
) -> dict:
    """
    - Appends CSV totals to summary.txt
    - Writes validation_report.txt with MoE group stats
    - If FULL mode, compares CSV total vs model parameter count and stamps PASS/FAIL
    Returns a dict of computed integrity numbers.
    """
    # Aggregate CSV totals
    sum_csv_params = sum(r.n_params for r in rows)
    sum_csv_train  = sum(r.n_trainable for r in rows)

    # MoE group accounting
    from collections import defaultdict
    groups = defaultdict(lambda: {"experts": 0, "routers": 0, "params": 0})
    for r in rows:
        mg = r.moe_group or "(none)"
        groups[mg]["params"]  += r.n_params
        if r.is_expert:
            groups[mg]["experts"] += 1
        if r.is_router:
            groups[mg]["routers"] += 1

    # Basic MoE consistency checks
    bad_groups = {}
    for k, v in groups.items():
        # (none) group is allowed to be anything; focus on actual MoE containers
        if k == "(none)":
            continue
        # Failure conditions: no router or no experts under a declared MoE container
        if v["routers"] == 0 or v["experts"] == 0:
            bad_groups[k] = v

    # Write validation report
    vr_path = os.path.join(report_dir, "validation_report.txt")
    with open(vr_path, "w") as vf:
        vf.write(f"==== Validation Report ({mode_label}) ====\n")
        vf.write(f"CSV param total     : {format_int(sum_csv_params)}\n")
        vf.write(f"CSV trainable total : {format_int(sum_csv_train)}\n\n")
        vf.write("MoE group summary (top 50 by params):\n")
        # Sort MoE groups by params descending
        sorted_groups = sorted(groups.items(), key=lambda kv: kv[1]["params"], reverse=True)
        for k, v in sorted_groups[:50]:
            vf.write(f"[moe] {k:70s} params={format_int(v['params'])} experts={v['experts']:>4} routers={v['routers']:>3}\n")
        if bad_groups:
            vf.write("\n[warn] Incomplete MoE groups detected (missing router or experts):\n")
            for k, v in bad_groups.items():
                vf.write(f"  - {k}: experts={v['experts']} routers={v['routers']}\n")

        # FULL-only: compare model total vs CSV total
        if model_params_full is not None:
            delta = abs(model_params_full - sum_csv_params)
            vf.write("\nModel total vs CSV total:\n")
            vf.write(f"  Model param total : {format_int(model_params_full)}\n")
            vf.write(f"  CSV param total   : {format_int(sum_csv_params)}\n")
            vf.write(f"  Delta             : {format_int(delta)}\n")

    # Append to summary and stamp PASS/FAIL
    summary_path = os.path.join(report_dir, "summary.txt")
    with open(summary_path, "a") as sf:
        sf.write(f"CSV param total     : {format_int(sum_csv_params)}\n")
        sf.write(f"CSV trainable total : {format_int(sum_csv_train)}\n")

        status = "PASS"
        reasons = []

        if bad_groups:
            status = "FAIL"
            reasons.append(f"{len(bad_groups)} incomplete MoE group(s)")

        if model_params_full is not None:
            delta = abs(model_params_full - sum_csv_params)
            sf.write(f"Model param total   : {format_int(model_params_full)}\n")
            sf.write(f"Param delta         : {format_int(delta)}\n")
            # Accept small non-zero deltas in case of tied embeddings or special tying (set to 0 to be strict)
            if delta != 0:
                status = "FAIL"
                reasons.append("model vs CSV param mismatch")

        if reasons:
            sf.write(f"INTEGRITY STATUS    : {status}  ({'; '.join(reasons)})\n")
        else:
            sf.write("INTEGRITY STATUS    : PASS\n")

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

def attach_example_hooks(model: nn.Module, example_routers: List[str], example_experts: List[str], log_writer) -> List:
    handles = []
    def _once(fn):
        fired = {"v": False}
        def wrap(*a, **kw):
            if not fired["v"]:
                fired["v"] = True
                return fn(*a, **kw)
        return wrap

    if example_routers:
        m = dict(model.named_modules()).get(example_routers[0])
        if m is not None:
            def router_hook(mod, inp, out):
                with torch.no_grad():
                    try:
                        t = out[0] if isinstance(out, tuple) else out
                        log_writer(f"[router:{example_routers[0]}] out shape: {tuple(t.shape)}")
                    except Exception as e:
                        log_writer(f"[router:{example_routers[0]}] hook error: {e}")
            handles.append(m.register_forward_hook(_once(router_hook)))

    if example_experts:
        m = dict(model.named_modules()).get(example_experts[0])
        if m is not None:
            def expert_hook(mod, inp, out):
                with torch.no_grad():
                    try:
                        t = out if isinstance(out, torch.Tensor) else out[0]
                        mean = t.float().mean().item()
                        std  = t.float().std().item()
                        log_writer(f"[expert:{example_experts[0]}] out mean={mean:.4f} std={std:.4f} shape={tuple(t.shape)}")
                    except Exception as e:
                        log_writer(f"[expert:{example_experts[0]}] hook error: {e}")
            handles.append(m.register_forward_hook(_once(expert_hook)))

    return handles


# -----------------------------
# Main
# -----------------------------

def main():
    args = parse_args()

    verbosity_levels = {
        "quiet": logging.WARNING,
        "normal": logging.INFO,
        "verbose": logging.DEBUG,
    }
    log_level = verbosity_levels.get(args.verbosity, logging.INFO)
    logging.basicConfig(level=log_level, format="%(message)s")
    logger.setLevel(log_level)

    if args.mode == "fast":
        fast_mode = True
    elif args.mode == "full":
        fast_mode = False
    else:
        fast_mode = args.fast

    logger.debug(
        "Resolved mode: fast_mode=%s (mode arg=%s, legacy --fast=%s)",
        fast_mode,
        args.mode,
        args.fast,
    )

    # Determine model id
    models_payload = None
    if args.model:
        model_id = args.model
    else:
        try:
            models_payload = get_models_from_endpoint(args.endpoint)
        except Exception as e:
            logger.error("[error] Failed to query endpoint '%s': %s", args.endpoint, e)
            sys.exit(1)
        try:
            model_id = pick_model_id(models_payload)
        except Exception as e:
            logger.error("[error] Failed to pick a model id from endpoint payload: %s", e)
            sys.exit(1)
        if not model_id:
            logger.error("[error] Could not infer a model id from endpoint payload.")
            sys.exit(1)

    if args.model:
        source_info = {"type": "model_argument", "value": args.model}
    else:
        source_info = {"type": "endpoint_discovery", "endpoint": args.endpoint}

    # Report dir
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mid = re.sub(r'[^a-zA-Z0-9_.\\-]+', '_', model_id)
    report_dir = os.path.join(args.outdir, f"{ts}_{safe_mid}")
    try:
        os.makedirs(report_dir, exist_ok=True)
    except OSError as e:
        logger.error("[error] Failed to create report directory '%s': %s", report_dir, e)
        sys.exit(1)

    if models_payload is not None:
        with open(os.path.join(report_dir, "endpoint.json"), "w") as f:
            json.dump({"endpoint": args.endpoint, "models_payload": models_payload}, f, indent=2)

    # Config
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception:
        cfg = None

    # ------------------ FAST mode (no weights) ------------------
    if fast_mode:
        if cfg is None:
            logger.error("[error] FAST mode requested but AutoConfig failed; cannot proceed.")
            sys.exit(1)
        if not ACCEL_AVAILABLE:
            logger.error("[error] FAST mode requires 'accelerate' (pip install accelerate)")
            sys.exit(1)
        logger.info("[info] FAST mode: building skeleton with empty weights (no real weight loading)")
        with init_empty_weights():
            try:
                # Try CausalLM first; many VLMs need base AutoModel
                try:
                    tmp_model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
                except Exception:
                    tmp_model = AutoModel.from_config(cfg, trust_remote_code=True)
            except Exception as e:
                logger.error("[error] Failed to instantiate model from config: %s", e)
                sys.exit(1)

        class_chain: Dict[str, str] = {}
        named_mods = list(tmp_model.named_modules())
        for name, mod in named_mods:
            class_chain[name] = mod.__class__.__name__

        rows: List[Row] = []
        total_params = 0
        total_train = 0
        for name, mod in named_mods:
            cls = mod.__class__.__name__
            parent = get_parent(name)
            depth = depth_of(name)
            # Empty-weights still has shapes; numel() is valid
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            n_train  = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
            n_buffers = sum(b.numel() for b in mod.buffers(recurse=False))
            total_params += n_params
            total_train += n_train
            is_router = guess_is_router(name, cls)
            is_expert = guess_is_expert(name, cls)
            moe_group = nearest_moe_group(name, class_chain)
            rows.append(Row(name, cls, parent, depth, n_params, n_train, n_buffers, is_router, is_expert, moe_group))

        # Write files
        mods_csv = os.path.join(report_dir, "modules.csv")
        with open(mods_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["name", "class", "parent", "depth", "n_params", "n_trainable", "n_buffers", "is_router", "is_expert", "moe_group"])
            for r in rows:
                w.writerow([r.name, r.cls, r.parent, r.depth, r.n_params, r.n_trainable, r.n_buffers, int(r.is_router), int(r.is_expert), r.moe_group])

        routers_csv = os.path.join(report_dir, "routers.csv")
        experts_csv = os.path.join(report_dir, "experts.csv")
        with open(routers_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "class", "moe_group"])
            for r in rows:
                if r.is_router: w.writerow([r.name, r.cls, r.moe_group])
        with open(experts_csv, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["name", "class", "parent", "moe_group"])
            for r in rows:
                if r.is_expert: w.writerow([r.name, r.cls, r.parent, r.moe_group])

        skeleton_txt = os.path.join(report_dir, "skeleton.txt")
        with open(skeleton_txt, "w") as f:
            f.write(model_skeleton(named_mods))

        summary_txt = os.path.join(report_dir, "summary.txt")
        n_mods = len(rows)
        n_routers = sum(1 for r in rows if r.is_router)
        n_experts = sum(1 for r in rows if r.is_expert)
        with open(summary_txt, "w") as f:
            f.write("==== vLLM Cluster Audit Summary (FAST) ====\n")
            f.write(f"Endpoint           : {args.endpoint}\n")
            f.write(f"Model ID           : {model_id}\n")
            f.write(f"Modules            : {n_mods:,}\n")
            f.write(f"Total params       : {format_int(total_params)}\n")
            f.write(f"Trainable params   : {format_int(total_train)}\n")
            f.write(f"Routers detected   : {n_routers:,}\n")
            f.write(f"Experts detected   : {n_experts:,}\n")
            f.write(f"Reports directory  : {report_dir}\n")

        # ---- SAFETY CHECKS (FAST) ----
        integrity_info = write_integrity_and_moe_reports(
            report_dir=report_dir,
            rows=rows,
            mode_label="FAST",
            model_params_full=None,   # no real weights in FAST mode
        )

        if args.json_output:
            try:
                write_modules_json(report_dir, rows)
                write_summary_json(
                    report_dir,
                    model_id=model_id,
                    source=source_info,
                    mode_label="FAST",
                    integrity_info=integrity_info,
                )
            except Exception as e:
                logger.error("[error] Failed to write JSON outputs: %s", e)
                sys.exit(1)

        integrity_status = (integrity_info or {}).get("integrity_status", "UNKNOWN")
        if integrity_status == "FAIL":
            reasons = (integrity_info or {}).get("integrity_reasons", [])
            reason_text = f" ({'; '.join(reasons)})" if reasons else ""
            logger.error("[fail] Integrity checks failed in FAST mode%s", reason_text)
            sys.exit(1)

        logger.info("[done][FAST] Wrote reports to: %s", report_dir)
        logger.info("[done][FAST] Key files: %s, %s, %s, %s, %s", mods_csv, skeleton_txt, routers_csv, experts_csv, summary_txt)
        return

    # ------------------ Full mode (real weights) ------------------
    if torch.cuda.is_available() and is_torch_bf16_gpu_available():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    device_map = "auto" if torch.cuda.is_available() else {"": "cpu"}

    logger.info("[info] Loading model '%s' (dtype=%s, device_map=%s) ...", model_id, dtype, device_map)
    try:
        # VLM-friendly first try
        model = AutoModel.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    except Exception as e:
        logger.warning("[warn] AutoModel load failed (%s); trying AutoModelForCausalLM ...", e)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
    model.eval()

    # Enumerate
    class_chain: Dict[str, str] = {}
    named_mods = list(model.named_modules())
    for name, mod in named_mods:
        class_chain[name] = mod.__class__.__name__

    rows: List[Row] = []
    total_params = 0
    total_train = 0
    for name, mod in named_mods:
        cls = mod.__class__.__name__
        parent = get_parent(name)
        depth = depth_of(name)
        n_params = sum(p.numel() for p in mod.parameters(recurse=False))
        n_train  = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
        n_buffers = sum(b.numel() for b in mod.buffers(recurse=False))
        total_params += n_params
        total_train += n_train
        is_router = guess_is_router(name, cls)
        is_expert = guess_is_expert(name, cls)
        moe_group = nearest_moe_group(name, class_chain)
        rows.append(Row(name, cls, parent, depth, n_params, n_train, n_buffers, is_router, is_expert, moe_group))

    # Write files
    mods_csv = os.path.join(report_dir, "modules.csv")
    with open(mods_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["name", "class", "parent", "depth", "n_params", "n_trainable", "n_buffers", "is_router", "is_expert", "moe_group"])
        for r in rows:
            w.writerow([r.name, r.cls, r.parent, r.depth, r.n_params, r.n_trainable, r.n_buffers, int(r.is_router), int(r.is_expert), r.moe_group])

    routers_csv = os.path.join(report_dir, "routers.csv")
    experts_csv = os.path.join(report_dir, "experts.csv")
    with open(routers_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "class", "moe_group"])
        for r in rows:
            if r.is_router: w.writerow([r.name, r.cls, r.moe_group])
    with open(experts_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["name", "class", "parent", "moe_group"])
        for r in rows:
            if r.is_expert: w.writerow([r.name, r.cls, r.parent, r.moe_group])

    skeleton_txt = os.path.join(report_dir, "skeleton.txt")
    with open(skeleton_txt, "w") as f:
        f.write(model_skeleton(named_mods))

    summary_txt = os.path.join(report_dir, "summary.txt")
    n_mods = len(rows)
    n_routers = sum(1 for r in rows if r.is_router)
    n_experts = sum(1 for r in rows if r.is_expert)
    with open(summary_txt, "w") as f:
        f.write("==== vLLM Cluster Audit Summary ====\n")
        f.write(f"Endpoint           : {args.endpoint}\n")
        f.write(f"Model ID           : {model_id}\n")
        f.write(f"Modules            : {n_mods:,}\n")
        f.write(f"Total params       : {format_int(total_params)}\n")
        f.write(f"Trainable params   : {format_int(total_train)}\n")
        f.write(f"Routers detected   : {n_routers:,}\n")
        f.write(f"Experts detected   : {n_experts:,}\n")
        f.write(f"Reports directory  : {report_dir}\n")

    # ---- SAFETY CHECKS (FULL) ----
    model_params = sum(p.numel() for p in model.parameters())
    integrity_info = write_integrity_and_moe_reports(
        report_dir=report_dir,
        rows=rows,
        mode_label="FULL",
        model_params_full=model_params,
    )

    # Hooks (full mode only; useful mainly for CausalLMs)
    if not args.no_hooks:
        hook_log = os.path.join(report_dir, "hook_log.txt")
        def log_writer(s: str):
            with open(hook_log, "a") as fh:
                fh.write(s + "\n")
            logger.info(s)

        example_routers = [r.name for r in rows if r.is_router][:1]
        example_experts = [r.name for r in rows if r.is_expert][:1]
        handles = attach_example_hooks(model, example_routers, example_experts, log_writer)

        try:
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        except Exception:
            tok = None
        if tok is not None:
            device0 = next(model.parameters()).device
            enc = tok("A short test prompt for model introspection.", return_tensors="pt").to(device0)
            with torch.no_grad():
                _ = model.generate(**enc, max_new_tokens=8)
        for h in handles:
            with contextlib.suppress(Exception):
                h.remove()

    if args.json_output:
        try:
            write_modules_json(report_dir, rows)
            write_summary_json(
                report_dir,
                model_id=model_id,
                source=source_info,
                mode_label="FULL",
                integrity_info=integrity_info,
            )
        except Exception as e:
            logger.error("[error] Failed to write JSON outputs: %s", e)
            sys.exit(1)

    integrity_status = (integrity_info or {}).get("integrity_status", "UNKNOWN")
    if integrity_status == "FAIL":
        reasons = (integrity_info or {}).get("integrity_reasons", [])
        reason_text = f" ({'; '.join(reasons)})" if reasons else ""
        logger.error("[fail] Integrity checks failed in FULL mode%s", reason_text)
        sys.exit(1)

    logger.info("[done] Wrote reports to: %s", report_dir)
    logger.info(
        "[done] Key files: %s, %s, %s, %s, %s, validation_report.txt",
        mods_csv,
        skeleton_txt,
        routers_csv,
        experts_csv,
        summary_txt,
    )


if __name__ == "__main__":
    main()