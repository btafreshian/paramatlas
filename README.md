# ParamAtlas

ParamAtlas inspects large language and vision-language models to build a
structured view of their internal module hierarchy. It works directly from a
Hugging Face model id or can discover the model id from a running vLLM
(OpenAI-compatible) endpoint. The tool offers a FAST mode that constructs the
module skeleton and parameter counts from configuration files without loading
weights, and a FULL mode that loads weights for complete validation.

## Features

- FAST mode builds the module tree and parameter counts using empty weights,
  avoiding heavy downloads or GPU memory usage.
- VLM-friendly model loading first attempts `AutoModel` and falls back to
  `AutoModelForCausalLM` when needed.
- Mixture-of-Experts awareness flags routers and experts and assigns them to the
  nearest MoE container.
- Safety checks compute CSV totals, perform integrity comparisons, and write a
  detailed validation report.
- Outputs are organized as text, CSV, and optional JSON files that are easy to
  diff and script.

## Installation

```bash
pip install -U torch "transformers>=4.42" accelerate requests
```

## Quick Start

### Inspect a Hugging Face model (FAST mode)

```bash
python audit_vllm_cluster.py --model <org/model> --fast
```

### Inspect a Hugging Face model (FULL mode)

```bash
python audit_vllm_cluster.py --model <org/model> --no-hooks
```

### Discover the model from a vLLM endpoint

```bash
python audit_vllm_cluster.py --endpoint http://localhost:8000 --fast
```

> The `--model` flag works without vLLM; the endpoint is only required when you
> want automatic model discovery.

## Example Workflow

1. Install the dependencies listed above.
2. Run FAST mode to skip weight downloads while building the module skeleton and
   MoE annotations.
3. (Optional) Run FULL mode to load weights and perform integrity comparisons.
4. Inspect the newest timestamped directory under `./reports/` for generated
   artifacts such as `summary.txt`, `modules.csv`, and `validation_report.txt`.

## Output Artifacts

Each run writes a timestamped folder under `./reports/<timestamp>_<model_id>/`.
Enable `--json-output` to add structured mirrors alongside the existing text and
CSV artifacts.

| File | Description |
| --- | --- |
| `summary.txt` | Totals, counts, and **INTEGRITY STATUS: PASS/FAIL**. |
| `summary.json` | When `--json-output` is set, a compact summary of totals, integrity status, and MoE statistics. |
| `skeleton.txt` | Indented module tree showing dotted names and classes. |
| `modules.csv` | One row per module: hierarchy metadata, parameter/buffer counts, router/expert flags, and `moe_group`. |
| `modules.json` | JSON mirror of `modules.csv` when `--json-output` is set. |
| `routers.csv` / `experts.csv` | Subsets highlighted by the MoE heuristic. |
| `validation_report.txt` | MoE per-group statistics, warnings, and (FULL mode) comparisons between CSV totals and model parameters. |
| `hook_log.txt` | Only in FULL mode when hooks run; one-shot tensor statistics for representative modules. |
| `endpoint.json` | Saved when an endpoint is used; records the `/v1/models` payload. |

## Understanding the Reports

- `summary.txt` gives headline counts, the integrity verdict, and any warnings.
- `summary.json` (with `--json-output`) mirrors the summary in a machine-friendly
  format.
- `skeleton.txt` exposes the hierarchical module tree for spotting unexpected
  components.
- `modules.csv` contains per-module metrics suitable for scripting or diffing.
- `modules.json` mirrors the CSV for downstream tooling when JSON output is
  requested.
- `routers.csv` and `experts.csv` filter MoE components for focused inspection.
- `validation_report.txt` analyzes MoE groups and compares CSV totals against
  instantiated model parameters.
- `hook_log.txt` captures initial tensor statistics when example hooks are
  enabled in FULL mode.
- `endpoint.json` preserves endpoint metadata whenever discovery is used.

## Mode Selection

- Choose **FAST** when you need immediate structure without downloading weights.
  It requires the model configuration (and any remote code referenced by
  `trust_remote_code`).
- Choose **FULL** when you want enumeration based on real instantiated modules
  and optional one-shot hooks. For VLMs, consider `--no-hooks`.
- Select a mode explicitly with `--mode fast` or `--mode full`, or rely on the
  legacy `--fast` flag when `--mode` is omitted.

## Integrity and Validation

ParamAtlas writes guardrails so you can trust the outputs:

- Summed CSV totals for parameters and trainable parameters.
- In FULL mode, a comparison between the CSV total and the actual model
  parameter count.
- MoE consistency checks that flag containers missing routers or experts.
- An integrity stamp appended to `summary.txt`, including reasons when a run
  fails validation.

## Usage Examples

```bash
# Mixtral (MoE LLM), structure only
python audit_vllm_cluster.py --model mistralai/Mixtral-8x7B-v0.1 --fast

# Qwen-VL (VLM), structure only
python audit_vllm_cluster.py --model Qwen/Qwen3-VL-30B-A3B-Instruct --fast

# Same model, full load while skipping hooks (recommended for VLMs)
python audit_vllm_cluster.py --model Qwen/Qwen3-VL-30B-A3B-Instruct --no-hooks

# Discover the model from vLLM, then run FAST mode
python audit_vllm_cluster.py --endpoint http://localhost:8000 --fast
```

## Deterministic Runs

To keep outputs stable across runs:

```bash
# Pin toolchain versions
pip install "transformers==4.44.2" "accelerate==0.34.2" torch==<your_version>

# Pin model revision (no drifting)
python audit_vllm_cluster.py --model org/model@<commit_sha> --fast

# After the first successful online run, force offline re-runs
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
python audit_vllm_cluster.py --model org/model@<commit_sha> --fast
```

## Supported Models

- LLMs such as Llama, Mistral, Gemma, Falcon, MPT, OPT, GPT-Neo/NeoX, Jamba,
  OLMo, DBRX, Yi, Phi, GLM, DeepSeek, and related families.
- MoE LLMs including Mixtral, Ministral-MoE, Qwen/Qwen3-MoE, Granite-MoE,
  GLM-MoE, and similar architectures.
- Vision-language models (using the `AutoModel` path) such as Qwen-VL/Qwen3-VL,
  LLaVA/Next, InternVL, MiniCPM-V, mPLUG-Owl, Kosmos-2, and Emu.
- Any repository providing a valid `config.json` and compatible
  `transformers.AutoModel*` class (possibly via `trust_remote_code=True`) works in
  FAST mode; most also work in FULL mode.

Unsupported targets include GGUF-only artifacts, TensorRT-LLM engines, and
ONNX-only repositories without a Transformers configuration.

## Command Reference

```text
usage: audit_vllm_cluster.py [--endpoint URL] [--model HF_ID_OR_PATH]
                             [--mode {fast,full}] [--fast]
                             [--verbosity {quiet,normal,verbose}]
                             [--no-hooks] [--outdir DIR]
                             [--json-output]

optional arguments:
  --endpoint URL            vLLM/OpenAI-compatible base URL (uses /v1/models)
  --model ID|PATH           HF model id or local path (skips endpoint discovery)
  --mode {fast,full}        Select fast (empty weights) or full (load weights)
  --fast                    Build skeleton from config (no weights)
  --verbosity {quiet,normal,verbose}
                            Control logging output level (default: normal)
  --no-hooks                Disable tiny one-shot hooks (full mode only)
  --outdir DIR              Output directory (default: reports)
  --json-output             Emit `summary.json` and `modules.json` with reports
```

> Prefer `--mode fast` or `--mode full` for explicit selection. When `--mode`
> is omitted, the legacy `--fast` flag continues to toggle FAST mode for
> backward compatibility.

## Quick Analysis Snippets

```bash
# Largest modules by parameter count
python - <<'PY'
import csv

rows = []
with open('reports/<run>/modules.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        row['n_params'] = int(row['n_params'])
        rows.append(row)

rows.sort(key=lambda item: item['n_params'], reverse=True)
for row in rows[:25]:
    print(f"{row['n_params']:>12,}  {row['name']}  ({row['class']})")
PY

# MoE groups by total parameters, experts, and routers
python - <<'PY'
import collections
import csv

groups = collections.defaultdict(lambda: {'experts': 0, 'routers': 0, 'params': 0})
with open('reports/<run>/modules.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        group = row['moe_group'] or '(none)'
        groups[group]['params'] += int(row['n_params'])
        if row['is_expert'] == '1':
            groups[group]['experts'] += 1
        if row['is_router'] == '1':
            groups[group]['routers'] += 1

for name, stats in sorted(groups.items(), key=lambda kv: kv[1]['params'], reverse=True)[:20]:
    print(
        f"{name:60s}  params={stats['params']:>12,}  "
        f"experts={stats['experts']:>4}  routers={stats['routers']:>3}"
    )
PY
```

## Integration Notes

ParamAtlas can be scripted from notebooks or CI jobs: invoke
`audit_vllm_cluster.py` with the desired `--model` or `--endpoint`, then load
`modules.csv`, `routers.csv`, or `experts.csv` into pandas for dashboards or
regression checks. Use `--json-output` when downstream tooling expects
structured JSON—the emitted `summary.json` and `modules.json` mirror the
text/CSV artifacts without additional parsing.

## Troubleshooting

- **Missing dependencies (`transformers`, `accelerate`, `torch`)**: reinstall
  using the installation command from the quick start or recreate the virtual
  environment with compatible versions pinned.
- **Endpoint discovery failures (`/v1/models` errors, timeouts)**: verify the
  vLLM server is reachable, networking is configured, and fall back to `--model`
  if necessary.
- **Invalid model id or local path**: double-check the spelling, ensure the
  repository exposes a `config.json`, and confirm local directories contain the
  expected Transformers files.
- **Out-of-memory during FULL mode**: switch to FAST mode, audit a smaller
  checkpoint, or run on hardware with additional VRAM/system RAM.
- **Permission errors writing to `reports/`**: set `--outdir` to a writable
  location or adjust filesystem permissions so timestamped report folders can be
  created.

## Contributing

Contributions are welcome. High-impact ideas include improved MoE detection,
per-layer statistics, or additional structured export schemas that build on the
existing `--json-output` option.
