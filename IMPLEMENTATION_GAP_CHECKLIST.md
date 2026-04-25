# Implementation Gap Checklist

This checklist captures what is still missing to move this repository from a strong prototype to a submission-ready and winner-competitive OpenEnv hackathon project.

Source alignment used for this checklist:
- `OpenEnv Hackathon Opening Ceremony _ 25th Apr.pdf`
- `CODEBASE_CONTEXT_FOR_AI.md`
- `HACKATHON_WINNING_ROADMAP.md`
- Current repository implementation state (environment/API/UI/training/tests/docs)

## 1) Readiness Snapshot

- `Environment/API/UI/Test foundation`: mostly implemented and runnable.
- `Largest gap`: real, reproducible training/evaluation evidence with committed artifacts.
- `Current risk`: claims can exceed committed proof if README is not strictly evidence-backed.
- `Competitiveness gap`: realism, anti-gaming reward behavior, and judge-first demo path still need strengthening.

---

## 2) Judging Alignment Matrix (Gap-Oriented)

## Environment Innovation (40%)
- [x] Core concept is novel enough (autonomous SRE under chaos pressure).
- [x] Multi-phase fault environment exists.
- [x] Make incidents feel clearly production-like at observation level (alerts, service health, SLO signals).
- [x] Upgrade generic module/task flavor into realistic service components and incident workflows.
- [x] Add postmortem output so judges can see meaningful causal learning beyond patching toy code.

Why this matters:
- Innovation has the highest weight. The environment must immediately feel frontier-level, not just technically functional.

## Storytelling & Presentation (30%)
- [x] High-level narrative exists.
- [x] Replace any unsupported claims with measured values linked to artifacts.
- [x] Add concise judge-first README flow: problem -> environment -> learning evidence -> impact.
- [x] Ensure all supporting assets are linked from README (HF Space + demo video/blog/slides + results files).
- [x] Prepare 90-120 second demo script and evidence-driven demo path.

Why this matters:
- Great internals can still score low if the story is not verifiable in 3-5 minutes.

## Showing Improvement in Rewards (20%)
- [x] Build and commit baseline-vs-improved evaluation outputs.
- [x] Commit real plots generated from logs, not mocked notebook data.
- [x] Include held-out seed evaluation summary.
- [x] Show trained/improved behavior against random/no-op/heuristic baselines.

Why this matters:
- Judges explicitly require evidence that learning improved behavior.

## Reward & Training Pipeline (10%)
- [x] Reward structure exists (R1-R5 + watchdog).
- [x] Make training loop truly environment-connected and reproducible.
- [x] Harden reward against signal/no-op farming and shallow exploit patterns.
- [x] Add policy-level tests proving reward encourages useful repair behavior.

Why this matters:
- Even at lower weight, this is a direct credibility filter for the whole project.

---

## 3) Non-Negotiable Submission Requirements Gap Check

## OpenEnv usage and compatibility
- [x] Uses OpenEnv-style API and manifest structure.
- [x] Re-verify manifest-to-runtime consistency before final submission (`actions`, `reward names`, endpoints, metadata).
- [x] Add/verify judge-friendly validation command sequence in README.

## Working training script path (TRL/Unsloth)
- [x] Replace scaffold-like training script behavior with real end-to-end run path.
- [x] Ensure at least one practical reproducible training or policy-improvement path is documented and runnable.

## Real training/evaluation evidence
- [x] Commit reward/loss and baseline comparison plots from real runs.
- [x] Commit machine-readable evaluation summary and rollout logs.

## Runnable HF Space deployment
- [ ] Validate live Space health endpoint, tasks endpoint, and interaction loop. (Blocked in local environment: remote Space access/auth verification required.)
- [x] Ensure README links to exact Space URL and usage path for judges.

## README with proof links
- [x] Include links to Space, results artifacts, and demo material.
- [x] Ensure every quantitative claim traces to committed files or external run links.

---

## 4) Priority Backlog (P0, P1, P2)

## P0 - Must Complete Before Submission

### P0.1 Real evaluation pipeline
- [x] Implement deterministic multi-policy evaluator with fixed seed support.
- [x] Include policies: `no_op`, `random`, `heuristic`, and one improved policy.
- [x] Log per-episode JSONL with reward history, key actions, survival outcome, and final vitality.

Acceptance criteria:
- [x] Same seed set can be evaluated across all policies.
- [x] Re-running with same seed set reproduces summary metrics.

### P0.2 Real artifacts package
- [x] Generate and commit `eval_summary.json`.
- [x] Generate and commit reward and baseline comparison plots from logged rollouts.
- [x] Add artifact README explaining exactly how each file was generated.

Acceptance criteria:
- [x] Plots are script-generated from committed JSON/JSONL data.
- [x] No mocked or hand-entered metric data.

### P0.3 README claim integrity
- [x] Remove/soften unsupported percentages or training claims.
- [x] Add results section with clearly labeled baseline vs improved outputs.
- [x] Add reproducibility commands for eval and plotting.

Acceptance criteria:
- [x] Every metric in README is reproducible via committed commands.

### P0.4 Judge-ready deployment path
- [ ] Verify HF Space can run one full episode path. (Blocked in local environment: requires remote Space runtime validation.)
- [x] Verify `/health`, `/tasks`, `/reset`, `/step` behavior in demo mode.
- [x] Ensure judge path is explicit in README.

Acceptance criteria:
- [x] Judge can run and verify environment without hidden setup assumptions.

---

## P1 - High-Impact for Finalist/Winning Odds

### P1.1 Credible model-improvement path
- [x] Implement either:
  - [x] SFT-before/after evaluation path (minimum), or
  - [x] GRPO/TRL/Unsloth full path (preferred).
- [x] Evaluate improved policy on held-out seeds and compare against baselines.

Acceptance criteria:
- [x] Improved policy beats random/no-op clearly on meaningful metrics.
- [x] Improvement includes held-out/generalization reporting.

### P1.2 Reward anti-gaming hardening
- [x] Penalize repeated `emit_signal` and low-value no-op loops.
- [x] Reward outcome-linked useful actions (recovery, containment, regression prevention).
- [x] Add policy tests that fail if exploit policies score too well.

Acceptance criteria:
- [x] Signal-spam and no-op policies reliably underperform targeted repair policy.

### P1.3 SRE realism upgrade
- [x] Increase realism of service modules/incidents/logs/alerts/SLO context.
- [x] Expose richer incident state in observations (latency/error/availability/blast radius).
- [x] Add post-episode incident/postmortem summary.

Acceptance criteria:
- [x] Non-technical judge can understand what failed, what agent did, and why score changed.

---

## P2 - Polish and Differentiation

### P2.1 Demo UX accelerators
- [x] Add one-click baseline episode run.
- [x] Add one-click heuristic/improved run.
- [x] Add reward breakdown and episode summary panels.
- [x] Add timeline-style incident/action/reward view.

### P2.2 Story assets
- [x] Publish 90-120 second demo video or concise HF blog/slides.
- [x] Link assets in README.

### P2.3 Repo/hf-space hygiene
- [x] Reduce runtime image/repo bloat by excluding nonessential heavy assets from build context.
- [x] Keep naming and metadata consistent across README, API metadata, and manifest.

---

## 5) Per-Domain Missing Work (Implementation Checklist)

## Domain A: OpenEnv Compliance
- [x] Re-validate `openenv.yaml` against runtime behavior.
- [x] Confirm compatibility expectations around OpenEnv base class usage and fallback behavior.
- [x] Add explicit local validator/smoke-test command sequence for judges.

## Domain B: Environment Innovation
- [x] Strengthen realistic incident scenarios and service-role semantics.
- [x] Improve phase narratives and failure cascades for clear long-horizon behavior.

## Domain C: Reward Design
- [x] Tighten anti-exploit guardrails.
- [x] Improve action usefulness attribution in reward components.
- [x] Add reward-hacking regression tests.

## Domain D: Training + Evaluation Pipeline
- [x] Add robust rollout/evaluation scripts.
- [x] Add reproducible baseline and improved policy comparison path.
- [x] Add held-out seed reporting.

## Domain E: Results + Artifacts
- [x] Commit machine-readable summaries.
- [x] Commit plots with clear axes and reproducibility scripts.
- [x] Add artifacts index/readme.

## Domain F: README + Storytelling
- [x] Make README proof-first and reproducibility-first.
- [x] Ensure all claims are evidence-linked.
- [x] Keep narrative short, ambitious, and verifiable.

## Domain G: HF Space + Judge Path
- [ ] Verify remote Space runs with clear public judge interaction path. (Blocked in local environment: requires live Space check.)
- [ ] Validate endpoint behavior used in judging. (Blocked in local environment: requires live Space check.)

## Domain H: Grader + Evaluation Robustness
- [x] Ensure grader cannot be gamed by signal/no-op loops.
- [x] Add deterministic hidden-seed style evaluation coverage.

## Domain I: Security + Reliability
- [x] Confirm safe defaults for auth/demo mode.
- [x] Add payload/time-limit protections where needed.
- [x] Verify session lifecycle reliability under limits.

## Domain J: Repo Hygiene
- [x] Keep runtime/deploy context lean.
- [x] Remove or isolate nonessential heavy assets from production image path.
- [x] Ensure naming consistency throughout user-facing docs/configs.

---

## 6) File-Level Implementation Targets

## Existing files likely to modify
- `README.md`
- `openenv.yaml`
- `training/grpo_train.py`
- `training/generate_sft_data.py`
- `training/curriculum.py`
- `rubrics.py`
- `environment.py`
- `data.py`
- `tasks.py`
- `ui.py`
- `validate.py`
- `test_env.py`
- `test_api.py`
- `Dockerfile`
- `.github/workflows/ci.yml`

## New files recommended to add
- `training/rollout.py`
- `training/evaluate_policy.py`
- `training/plot_results.py`
- `training/train_sft.py`
- `training/train_grpo.py` (or replace/upgrade existing scaffold)
- `evaluation/run_eval.py`
- `results/README.md`
- `results/eval_summary.json`
- `results/random_rollouts.jsonl`
- `results/noop_rollouts.jsonl`
- `results/heuristic_rollouts.jsonl`
- `results/improved_rollouts.jsonl`
- `results/reward_curve.png`
- `results/baseline_vs_agent.png`
- `results/survival_by_phase.png`
- `results/action_distribution.png`

---

## 7) Definition of Done (Submission Gate)

All checks below should be green before final submission:

## Evidence and reproducibility
- [x] Baseline vs improved evaluation is committed and reproducible.
- [x] `results/eval_summary.json` exists and matches plotted values.
- [x] Result plots are generated from committed logs via documented command(s).

## Honesty and claims
- [x] README has no unsupported performance/training claims.
- [x] Every metric claim points to a file, run log, or linked public run artifact.

## Judge experience
- [ ] HF Space link works. (Blocked in local environment: public remote check required.)
- [x] Judge can run at least one episode and inspect evidence quickly.
- [x] README is understandable within 3-5 minutes.

## Technical confidence
- [x] `pytest -q` passes.
- [ ] Docker build passes. (Blocked locally: Docker daemon unavailable on this machine.)
- [x] Validator/smoke run passes against live/local server.
- [x] Grader behavior is robust against low-value exploit policies.

---

## 8) Execution Timeline (Dependency-Aware)

## Phase 0 (30-60 min): Claim cleanup
- [x] Remove unsupported claims and set evidence-first README baseline.

## Phase 1 (3-5 hrs): Real evaluation foundation
- [x] Implement rollout/eval scripts and generate baseline artifacts.

## Phase 2 (1-2 hrs): Results packaging
- [x] Generate plots and summary files; embed in README.

## Phase 3 (6-12 hrs): Credible improvement
- [x] Add SFT/GRPO or equivalent improved policy path and compare on held-out seeds.

## Phase 4 (3-6 hrs): Realism + reward hardening
- [x] Upgrade incident realism and anti-gaming behavior with tests.

## Phase 5 (2-4 hrs): Final polish + submission checks
- [ ] Verify HF Space + demo asset links + full submission gate. (Pending final remote submission-day validation.)

---

## 9) Critical Pitfalls to Avoid

- [ ] Do not submit with mocked reward curves or unverifiable metrics.
- [ ] Do not call heuristic-only improvement "trained" unless model training truly occurred.
- [ ] Do not leave README claims disconnected from committed evidence.
- [ ] Do not optimize UI polish before evidence quality is complete.
- [ ] Do not rely on post-deadline commits for judged fixes.

---

## 10) Immediate Next Build Order

1. Implement `training/evaluate_policy.py` + deterministic seeds + JSONL logging.
2. Generate baseline rollouts and commit `results/eval_summary.json`.
3. Implement `training/plot_results.py` and commit plot artifacts.
4. Update `README.md` to evidence-backed numbers and reproducibility commands.
5. Validate HF Space judge path and link demo material.
6. Then execute reward hardening + realism upgrades.
