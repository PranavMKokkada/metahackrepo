# Video demo — pre-flight checklist and script

Use this before recording a walkthrough of **Autonomous SRE** (local, Docker, or [Hugging Face Space](https://huggingface.co/spaces/teletubbies/autonomous-sre)).

---

## 1. Environment checklist (avoid surprises on camera)

| Item | Why it matters |
|------|----------------|
| **App healthy** | Open `/health` — expect JSON `status: healthy`. |
| **HTTPS / Gradio** | On Spaces, the app uses Uvicorn `proxy_headers=True` so Gradio loads assets over **https://** (see `.github/hf-space/README.md`). If the UI is blank or the console shows mixed-content errors, redeploy from current `main`. |
| **API key for `/console` and REST** | Set Space secret **`CODEORGANISM_API_KEYS`** to a comma-separated list of keys you control. If you omit it, the server logs an **ephemeral** `x-api-key` once at startup — copy it from the Space **Logs** tab into the console’s API key field, then **Save**. |
| **Auth off only for local scratch** | `CODEORGANISM_AUTH_DISABLED=true` skips API keys (not for public Spaces). |
| **LLM expert (optional)** | **`OPENAI_API_KEY`** in the Space (or local env) enables Snorkel-style **patch quality** scoring via OpenAI in `data.py`. Without it, **`request_expert`** and patch evaluation still run using **heuristic fallback** — safe to demo; say “heuristic oracle” if no key. |
| **External LLM agent (`inference.py`)** | Separate flow: needs **`HF_TOKEN`** / **`API_KEY`**, **`MODEL_NAME`**, **`API_BASE_URL`**, and **`CODEORGANISM_API_KEY`** pointing at this app. Not required for Gradio UI recording. |
| **Rate limits** | Default 120 requests / 60s per key+IP. Rapid clicking in `/console` is usually fine; automated bursts can hit **429**. |

---

## 2. Surfaces to show (backend + frontends)

1. **Root** — `GET /` — short JSON with links to console and UI.
2. **Gradio Control Center** — `/ui/` — primary “judge-friendly” dashboard (SLA, impact, diagnostics, guided demo).
3. **Static operations console** — `/console/` — human-in-the-loop platform: production mode, guardrails, CI/CD snapshot, memory, predictive block, log ingest (all backed by `/platform/...` routes).
4. **OpenEnv API** — `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/tools/list`, `/tools/call` — narrate as the **agent contract**; use `validate.py` or `client.py` if you want a terminal B-roll.

---

## 3. Suggested recording order (~3–6 minutes)

1. **Hook (15s)** — Open the Space URL; show `/health` or the JSON from `/` in a second tab if you want a “serious API” beat.
2. **Gradio — Initialize (30s)** — `/ui/` → choose **Incident Profile** (phase_1 → phase_3) → **Initialize Session**. Call out **vitality**, **diagnostics table**, **dependency graph**, **signals**.
3. **Gradio — Chaos (20s)** — **Trigger Chaos Incident** — show vitality and stack / alerts updating (live fault injection, not a canned video).
4. **Gradio — Guided demo (60–90s)** — **Run Guided Demo Mode** — narrate the on-screen **stages** (awareness → chaos → scripted remediation: signal → tests → patch → subagent → expert → tests). This is the strongest single take for judges.
5. **Gradio — Manual protocol (30s)** — **Remediation Protocol** tab → run **Execute Remediation** with a simple **`patch_file`** (`OLD|NEW`) or **run_tests** — shows operator control.
6. **Console — API key (20s)** — `/console/` → paste **`x-api-key`** → **Save** → **Refresh platform** — episode JSON and toggles populate.
7. **Console — Production + guardrails (45s)** — Enable **production mode**, mention **approve/reject** workflow; adjust **guardrails** and **Save**; optional **Reset phase_1** + **Run tests** to show **last step** + explainability panels.
8. **Optional B-roll** — Terminal: `python validate.py --api-url ... --api-key ...` or mention **`inference.py`** only if you have HF/OpenAI keys configured.

---

## 4. Functionalities worth highlighting (talking points)

- **Self-corrupting codebase simulator** — faults, tests, vitality, checkpoints; long-horizon **incident response** framing.
- **Multi-phase curriculum** — `phase_1` / `phase_2` / `phase_3` with different difficulty and gates (`tasks.py`).
- **Reward rubric (R1–R5)** — vitality, test recovery, efficiency, coordination, novelty; **watchdog** penalties (`rubrics.py`, `environment.py`).
- **World model** — **dependency graph** in observations; **alerts** and **signals** for multi-agent / coordination narrative.
- **Enterprise SRE layer** — `/platform/session/state`: **production mode** (suggestions → approve), **guardrails**, **CI/CD**, **business**, **predictive alerts**, **memory**, **evolution**, **last-step explainability** (`sre_platform/`).
- **MCP-style tools** — `/tools/list` + `/tools/call` mapping to the same **`Action`** model as `/step`.
- **Expert path** — **`request_expert`** + optional **OpenAI** patch review; **deterministic heuristic** when the API key is absent.
- **Hugging Face / Docker hardening** — `Dockerfile` runs **pytest** in the image; **port 7860**, **healthcheck**, **`training/rollout.py`** bundled so **`ui.py`** always mounts.

---

## 5. One-line “pitch” variants

- *“A Gym/OpenEnv-style hostile codebase where an agent or operator fights injected faults under explicit SRE rewards and a production-style control plane.”*
- *“Same environment over REST, Gradio, and a static console — guided mode is built for demos.”*

---

## 6. After recording

- Re-run **`python scripts/submission_preflight.py`** before submit.
- Confirm CI: **`ruff check .`**, **`pytest`**, **`pip-audit`**, Docker build (see `.github/workflows/ci.yml`).
