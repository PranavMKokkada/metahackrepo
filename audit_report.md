# Repository Audit Report

## Remediation Status

This report was originally written before the security remediation pass. The following findings have now been addressed in code:

- Constrained patch execution: `patch_file` now only accepts `OLD|NEW` edits, validates Python AST, denies dangerous calls/imports, and no longer executes tests through host subprocesses.
- API boundary: mutable and sensitive endpoints now require an API key, CORS defaults to local origins, and per-key/client rate limiting is enforced.
- MCP: `/tools/call` now imports and uses `CodeOrganismActionType` correctly and is covered by API tests.
- Runtime portability: synthetic tests no longer create temporary source trees, avoiding the Windows temp permission failure path.
- API/schema drift: quarantine now documents `module` and remains backward-compatible with older `path` payloads.
- Terminal-state crashes: UI and Gym wrapper now handle `observation=None`.
- Determinism: curriculum seed derivation now uses SHA-256 instead of Python's randomized `hash()`.
- Sessions: session storage now has TTL and max-session eviction.
- Testing/CI/Docker: `test_api.py` is now a TestClient suite; `pytest -q` passes; Docker no longer swallows test failures, no longer uses recursive `COPY . .`, and runs as a non-root user.
- Dependency hygiene: runtime and training dependencies are split, runtime dependencies are pinned, and CI installs from `requirements.txt`.

Remaining operational note: the old `tmp_pytest/` directory created by the previous failing implementation is ignored and excluded from pytest, but Windows denied deletion of its locked subfolders in this environment.

## 1. Executive Summary

Overall health: **medium-risk prototype quality, not production-ready**. The core idea is coherent, but several runtime, security, testing, and deployment problems would block reliable public use.

Severity overview:
- **High:** remote code execution path through agent patches/tests, broken MCP endpoint, unauthenticated public mutation surface, reset/test failures in this environment.
- **Medium:** API/schema drift, nondeterministic seeds, fragile tests/CI, Docker masking failures, dependency drift.
- **Low:** docs/product naming inconsistencies, tracked generated/binary artifacts, unused imports/dead code.

Verification:
- `python -m py_compile ...` passed.
- `pytest -q` failed during `test_api.py` collection because it performs live HTTP calls at import time.
- `pytest -q test_env.py` produced **31 passed, 22 failed**, mainly `PermissionError` from `data.py` temporary test execution.

## 2. Critical Issues (High Severity)

### Issue: Agent-controlled patch content can execute arbitrary Python during tests
File/Location: `data.py:525`, `data.py:664`, `app.py:114`

Explanation: `patch_file` accepts full file overwrite when `diff` has no `|`. Later `run_all_tests()` writes those files and executes generated test wrappers via `subprocess.run(["python", test_path])`. The wrapper imports the patched module, so any top-level code in an agent patch executes on the host process environment.

Suggested Fix: Do not execute user-supplied Python directly. Use a locked-down container/process sandbox with network disabled, strict CPU/memory/time limits, a scrubbed environment, and a read-only root. Prefer AST-level validation or a constrained patch DSL.

### Issue: Public API has no authentication and permissive CORS
File/Location: `app.py:31`, `app.py:114`, `app.py:180`

Explanation: `allow_origins=["*"]` plus unauthenticated `/reset`, `/step`, `/grader`, `/sessions/create`, and `/tools/call` lets any website or client mutate environment state, create sessions, and trigger expensive test execution.

Suggested Fix: Require API keys or signed session tokens for mutation endpoints, restrict CORS to known origins, and rate-limit session creation/step execution.

### Issue: MCP `/tools/call` endpoint is broken
File/Location: `app.py:173`

Explanation: `CodeOrganismActionType` is referenced but never imported. Verified with FastAPI TestClient: `/tools/call` returns `400` with `name 'CodeOrganismActionType' is not defined`.

Suggested Fix:

```python
from models import Action, Observation, StepResult, EnvState, CodeOrganismActionType
```

Add a TestClient regression test for every advertised MCP tool.

### Issue: Core reset/test execution fails in current Windows/sandbox environment
File/Location: `data.py:557`, `data.py:593`

Explanation: `CodebaseSimulator.run_all_tests()` uses `TemporaryDirectory()` then creates `src/` or `schema/`; this repeatedly failed with `PermissionError`. Because `reset()` runs tests immediately, this breaks `/reset`, grader, sessions, and many unit tests.

Suggested Fix: Make the temp root configurable, create it under an app-owned writable directory, handle cleanup errors, and add Windows CI. Avoid chmod/read-only simulation on platforms where it breaks cleanup.

## 3. Major Issues (Medium Severity)

- API/action schema drift: `inference.py:55` tells models to send `{"action_type":"quarantine","path":...}`, but `models.py:133` and `environment.py:399` require `module`. Pydantic ignores the extra `path`, so the action fails.
- UI can crash at episode completion: `ui.py:189` reads `result.observation`, then `ui.py:192` dereferences `obs.timestep`; `environment.step()` returns `observation=None` when done.
- Gym wrapper can crash after terminal step: `gym_wrapper.py:99` calls `_get_obs_vec(result.observation)` even when `observation` is `None`.
- Seeds are not reproducible across processes: `data.py:855` uses Python `hash()`, which is randomized per interpreter unless `PYTHONHASHSEED` is fixed.
- Grader overcounts watchdog violations: `tasks.py:103` adds cumulative `env._watchdog_violations` each step instead of per-step deltas.
- Session storage is unbounded: `environment.py:593` creates sessions indefinitely with no TTL, quota, or eviction.
- `test_api.py` is not a real pytest-safe test: `test_api.py:18` performs live HTTP at import time; `test_api.py:10` also contains a bad `"/ health"` request.
- Docker hides build failures: `Dockerfile:13` uses `pytest ... || echo "Tests completed"`, so broken tests still produce an image.

## 4. Minor Issues (Low Severity)

- Unused imports in `app.py:5` and `data.py:13` increase noise.
- Product naming is inconsistent: `pyproject.toml:2` says `shadow-council`, while `openenv.yaml:1` says `autonomous-sre`, and code says `code-organism-vm`.
- README action names do not match API names: `README.md:19` mentions `patch_node`, `rollback_canary`, `circuit_break`, `spawn_team`, while API uses `patch_file`, `rollback`, `quarantine`, `spawn_subagent`.
- Tracked/generated artifacts bloat the repo: `training/sft_data.jsonl` is ~1.2 MB, PDFs include an ~8.4 MB deck, and local `__pycache__` files are present in the working tree even though ignored.

## 5. Security Analysis

High-risk surface is the combination of unauthenticated mutation, broad CORS, Gradio UI mounting, and subprocess execution of patched code. `data.py:661` also starts test subprocesses with `os.environ.copy()`, so secrets such as `OPENAI_API_KEY`/HF tokens could be available to executed code. `environment.py:562` exposes simulated `env_vars` in observations; keep real secrets out of this path.

Gradio is loosely constrained as `gradio>=4.0.0` in `requirements.txt:8`. This range permits historically vulnerable versions; current advisories include Gradio path traversal issues fixed in later versions, including CVE-2026-28414 fixed in 6.7.0.

## 6. Performance Analysis

- Every `state()` call reruns all tests: `environment.py:319`. This is expensive and can be triggered repeatedly over HTTP.
- Each test is a separate Python subprocess: `data.py:641` to `data.py:664`. With 20-40 tests per step and many episodes, this will dominate latency.
- Docker installs heavyweight training dependencies (`torch`, `transformers`, `unsloth`) for runtime API serving, increasing build time and image size.

## 7. Code Quality & Maintainability

Architecture is readable but tightly coupled: `environment.py`, `tasks.py`, UI, and tests access private fields like `_simulator`, `_vitality`, and `_thriving_streak`. Fault simulation mixes real execution with shortcut logic: `data.py:647` marks corrupted files failed before executing tests, while other fault types only append comments or mutate env vars. This weakens fidelity and makes results hard to reason about.

## 8. Dependency Audit

- Manifest drift: `pyproject.toml:6` omits several runtime/training packages present in `requirements.txt:9`.
- Several dependencies are floating: `transformers`, `datasets`, `accelerate`, `torch`, `unsloth`, `networkx`, `numpy`, `gymnasium`, `gradio`.
- CI does not install from either manifest: `ci.yml:26` manually installs a subset, so local, Docker, and CI dependency graphs differ.
- Current web check: PyPI shows Gradio 6.x releases in April 2026, while advisories warn that older Gradio ranges have path traversal/file-read issues. Pin a known fixed version and run `pip-audit` or OSV scanning in CI.

## 9. Testing & Coverage Review

`pytest -q` currently fails before running the full suite because `test_api.py` calls a live server during import. `pytest -q test_env.py` fails 22 tests in this environment due temp-dir permissions. CI only runs `test_env.py`, so it would miss `test_api.py`, training scripts, inference, UI, MCP, Docker, and client behavior. Add TestClient tests for the API and avoid live localhost dependency in unit tests.

## 10. DevOps / Deployment Issues

- Docker build masks test failures, copies large binaries/docs/training data, and runs as root by default.
- `.dockerignore` excludes most Markdown but keeps large PDFs/DOCX and `training/sft_data.jsonl`.
- Healthcheck only hits `/`, not `/reset`, `/health`, or `/tools/call`, so it misses broken core functionality.
- CI does not run Docker build, dependency audit, API tests, or lint/type checks.

## 11. Recommendations & Action Plan

1. Fix the security boundary first: authenticate mutation endpoints, restrict CORS, remove host env from subprocesses, and replace direct Python execution with a real sandbox.
2. Repair runtime blockers: import `CodeOrganismActionType`, make temp execution portable, handle `observation=None`, and align quarantine schema/docs/prompts.
3. Stabilize tests: convert `test_api.py` to TestClient tests, run all tests in CI, fail Docker builds on test failure, and add Windows/Linux coverage.
4. Normalize packaging: reconcile `pyproject.toml` and `requirements.txt`, split runtime vs training dependencies, pin all packages, and add dependency scanning.
5. Clean repository hygiene: remove generated caches/artifacts from tracking, move large PDFs/decks to releases or external storage, and update README/OpenEnv naming.

Sources used for current dependency/security context: PyPI Gradio release listing, GitLab advisory for CVE-2026-28414, GitLab advisory for CVE-2024-47166, Snyk uvicorn package page.
