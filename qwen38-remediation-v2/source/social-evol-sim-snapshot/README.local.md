# Local RTX 3090 setup (Windows, PowerShell, Python 3.12+)

From this repository directory:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/setup-local.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/start-local-server.ps1
```

Setup installs official llama.cpp **b11669 CUDA 12.4** binaries and their CUDA DLLs
under `.local/`, verifies the pinned **ggml-org/Qwen3.8-27B-GGUF Q4_K_M** model's
SHA256, and installs the existing `pyproject.toml` dependencies in `.venv`.
No CUDA toolkit, paid key, database, projector, or new runtime dependency is needed.
The initial download is about 19.6 GB. Inference then runs locally.

The server binds to **127.0.0.1:8080**, aliases the model as
`Qwen3.8-27B-Q4_K_M`, and requests all layers on CUDA0, 32768 context tokens,
one inference slot, flash attention, and `q8_0` K/V caches. Automatic fitting is
disabled so an allocation failure is visible. Settings have not been expanded
beyond these defaults. Logs: `.local/llama-server.log`. Stop with Ctrl+C.

In a second PowerShell terminal:

```powershell
$env:PATH = "$PWD\.venv\Scripts;$env:PATH"
$env:LITELLM_LOCAL_MODEL_COST_MAP = 'True'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONUTF8 = '1'
python -m pytest scr/tests/ -q
python scripts/smoke_local.py
```

The smoke test calls the repository's original three-phase `step()` **exactly
twice**, with file-only checkpoints, all eight agents (two of each of the four
active moral types), and the unchanged `configZ_major_v2` prompts and world.
The original initial time step is 1; steps 1 and 2 are social rounds under this
configuration. No 80-step experiment is started. It saves full conversations,
checkpoints, GPU samples, and `smoke_report.json` in `data/<run_id>/`. It verifies
all 16 decisions, both draft/reflection calls, contextual/Pydantic validation,
separate memory objects and updates, and checkpoint reloads. Retries are retained.
Throughput comes from llama-server metric deltas; VRAM is sampled with nvidia-smi
and includes Windows/other GPU processes (WDDM does not expose reliable per-process
VRAM here). Inspect the server log for actual layer offloading and buffer residency.

For a run through the existing CLI, use these explicit local overrides:

```powershell
python main.py run --config_dir configZ_major_v2 --no_db `
  --config.llm.provider llama_cpp `
  --config.llm.chat_model Qwen3.8-27B-Q4_K_M `
  --config.llm.two_stage_model true `
  --config.llm.async_config.max_concurrent_calls 1 `
  --config.llm.async_config.call_timeout_seconds 3600 `
  --config.world.max_life_steps 2
```

**CLI life steps include social rounds:** that last command allows six simulation
steps; use the smoke script for exactly two. It is provided for reference and is
not run as part of setup. Both stages use the same resident GPU model, while each
agent retains its own messages and memories. The single-slot semaphore changes
inference scheduling only; all decisions still see the same frozen step state.

The existing `response_format={"type":"json_object"}` on both calls activates
llama.cpp's JSON grammar. The existing Pydantic/action/context validators and retry
loop still check semantic validity. Model-internal hidden thinking is disabled
with `--reasoning off`; the prompt's explicit `thinking` field and full second
reflection prompt are unchanged. LiteLLM has a 1800-second HTTP timeout per call;
the existing retry timeout is 3600 seconds for the two-call attempt.

Provider configuration: `scr/api/llm_api/config.py` maps `llama_cpp` to LiteLLM's
`openai/` transport at `http://127.0.0.1:8080/v1`, using `local` solely as an SDK
placeholder. It never reads a paid API key for this provider. The environment
settings above keep model-cost metadata and Hugging Face access offline at run time.

Sources: [GGUF](https://huggingface.co/ggml-org/Qwen3.8-27B-GGUF),
[llama.cpp b11669](https://github.com/ggml-org/llama.cpp/releases/tag/b11669),
[JSON response formats](https://github.com/ggml-org/llama.cpp/blob/b11669/tools/server/README.md),
[LiteLLM OpenAI-compatible transport](https://docs.litellm.ai/docs/providers/openai_compatible).

## Measured validation — 2026-10-16 (Dhaka)

Base repository commit: `4863257c941c17ecf7556a1b2947c91400df8a47`.
Python 3.12.3, LiteLLM 1.105.1, OpenAI SDK 2.54.0, Pydantic 2.14.1;
llama.cpp b11669 / `620d2225c`, NVIDIA driver 591.86.

| Check | Actual result |
| --- | --- |
| Existing tests + local provider tests | **101 passed**, two existing Pydantic deprecation warnings |
| Smoke run | `1015-201445`: **2 steps, 8 agents, 16 final decisions passed** |
| Model calls | 34 JSON responses: 17 drafts + 17 reflections |
| Retry path | One missing-action schema failure; original retry succeeded on attempt two |
| Applied actions | 13 communicate, 3 do_nothing |
| Memories/checkpoints | Independent objects and updates verified; 16 checkpoints reloaded; both original file layouts saved |
| GPU residency | **65/65 layers offloaded**, flash attention enabled, one 32768-token slot |
| CUDA buffers | Model 17402.38 MiB; Q8 K/V 1088.00 MiB; recurrent state 149.62 MiB; compute 260.29 MiB |
| Board VRAM before server / loaded idle | 949 MiB / 20156 MiB |
| Board VRAM during smoke | **20142–20255 MiB**, peak **19.78 GiB** of 24 GiB |
| Generation | **32.49 tokens/s**, 65981 tokens in 2030.67 generation seconds |
| Uncached prompt processing | **1255.98 tokens/s**, 143624 tokens in 114.352 seconds |
| Total smoke wall time | **35 min 56 sec** |
| Largest observed sequence | 25717 / 32768 tokens |

llama.cpp also reports a normal 682.03 MiB CPU-mapped model buffer and host
staging buffers; the full-layer offload count above is its actual residency report.
Board VRAM measurements include other Windows GPU usage. Only loopback TCP
connections were observed for the inference processes. No external LLM API was used.

Evidence: [smoke report](data/1015-201445/smoke_report.json),
[full run artifacts](data/1015-201445/), [test output](tests-local.log),
and [server log](.local/llama-server.log). The server was left healthy and idle
at `127.0.0.1:8080`; no third step or 80-step experiment was started.

There are no remaining setup/smoke blockers. Coverage is limited to the original
first two social rounds; long-run context growth, reproduction, and population
changes have not been measured. Linear scaling of this smoke run gives **about
24 hours for 80 simulation steps**, or **72 hours for `max_life_steps=80`**
(240 simulation steps). Longer memories, retries, and population changes can extend
these estimates.

Changed files: `scr/api/llm_api/config.py` (the sole application-code change),
`.gitignore`, `scripts/setup-local.ps1`, `scripts/start-local-server.ps1`,
`scripts/smoke_local.py`, `scr/tests/test_llama_cpp_provider.py`,
`scr/tests/test_cli_parser.py` (three Windows interpreter fixes), and this README.
