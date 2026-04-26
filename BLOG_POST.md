# The bruised codebase

**Autonomous SRE / CodeOrganismVM**  
OpenEnv-style incident sandbox

> Green dashboards can still feel wrong. Incidents are long stories, not single diffs.

---

## Executive summary

| Expectation | Delivery |
|-------------|----------|
| A surface where large language models **train**, not only chat | Hostile codebase simulator with a **reset**, **step**, and **state** application programming interface |
| Emphasis on **self-improvement**, **multi-agent** behaviour, and **world modelling** | Vitality and fault pressure plus **signals**, **subagents**, **dependency graph**, and **platform memory** |
| Evidence, not narration alone | **Plots and structured logs** in the repository, **Unsloth** and **Transformer Reinforcement Learning (TRL)** notebook, **baselines compared with trained policies** |
| A readable account, not a raw specification | This article, the Hugging Face Space interface, and **README** entry points |

---

## Parallel perspective

The following two columns carry the same story from complementary viewpoints (layout reference: paired columns, independent reading order).

<table>
<tr valign="top">
<td width="50%">

<p><strong>Operations and reliability.</strong> On-call work is sequential and cumulative. A single green build does not erase the memory of last week’s regression. Teams need practice environments where mistakes have duration, where vitality falls when control is lost, and where coordination signals precede risky edits.</p>

<p><strong>What we simulate.</strong> A procedural service codebase, failing tests, checkpoints, guardrails, and an operations console so human reviewers and automated policies share the same incident thread.</p>

</td>
<td width="50%">

<p><strong>Research and learning.</strong> Reinforcement learning with environments only teaches when rewards are informative across many steps. Sparse end-of-episode scores are not enough here; we use a decomposed rubric so improvement is visible and harder to obtain by shortcut.</p>

<p><strong>What we optimise.</strong> Policies interact through the same OpenEnv-shaped loop as production clients: observations return structure, actions remain typed, and post-step analytics update business and platform state for reproducible evaluation.</p>

</td>
</tr>
</table>

---

## Hackathon themes and this product

| Theme (opening session) | Expression in this environment |
|--------------------------|----------------------------------|
| **Self-improvement** | Recovery under **new** fault draws; no static winning script. |
| **Long-horizon planning** | Episodes extend over many steps; hasty patching consumes vitality. |
| **Multi-agent** | **Signals**, **subagents**, and an expert channel reward coordination before blind changes. |
| **World modelling** | **Dependency graph**, alerts, and memory reduce confusion between symptoms and causes. |

---

## Observations, actions, and outcomes

| **Observations** | **Actions** | **Outcome signal (rubric family)** |
|------------------|-------------|-----------------------------------|
| Tests, files, vitality | `patch_file`, `run_tests`, `rollback`, `quarantine`, and related verbs | Vitality change, test recovery |
| Stack traces, dependency graph | `emit_signal`, `spawn_subagent`, `request_expert` | Efficiency, coordination, novelty |
| Watchdog boundaries | Refusal of unsafe paths | Penalties when protected regions are violated |

**Design intent:** the signal remains **dense**, **decomposable** (multiple rubric channels), and **resistant to trivial reward hacking** without genuine repair.

---

## Training trajectory (conceptual)

| Stage | Role |
|-------|------|
| **Pretraining** | Broad language priors (out of scope to repeat inside this repository). |
| **Supervised fine-tuning** | Task-shaped behaviour from demonstrations and the notebook path. |
| **Preference alignment (often called RLHF)** | Human-aligned preferences before strict environment reinforcement. |
| **Reinforcement learning with environment** | **This submission:** the policy receives credit from **live** `step()` outcomes and long horizons. |
| **Harness** | Tools, memory, application programming interface boundaries, guardrails, and observability so learning remains credible under a production-shaped interface. |

---

## Alignment with judging criteria

| Criterion | Weight | Response in one line |
|-----------|--------|------------------------|
| Environment innovation | **40%** | Self-corrupting **service codebase** with an operations layer, not a toy grid. |
| Storytelling and presentation | **30%** | Space, this article, and **README**: problem, environment, results, and motivation. |
| Improvement in rewards | **20%** | Curves and tables under `results/`, **contrasted with** noop, random, heuristic, and supervised-fine-tuning-style baselines. |
| Reward and training pipeline | **10%** | Structured rewards; **Unsloth** and **TRL** notebook on the same code path as the server. |

---

## Narrative structure (four beats)

| Step | Beat | Summary |
|------|------|---------|
| 1 | **Problem** | Large language models need **extended** incident practice, not isolated trivia answers. |
| 2 | **Environment** | Typed world: faults, tests, vitality, graph, telemetry, standard OpenEnv surface. |
| 3 | **Results** | **Before and after** training, **contrasted with baselines**, plots committed to the repository. |
| 4 | **Motivation** | Skills learned on a controlled bruised codebase transfer toward **more complex** production systems. |

---

## Artefact locations

| Artefact | Location |
|----------|----------|
| **Live demonstration** | This Hugging Face **Space** (Docker hosts the application programming interface, Gradio, and static console) |
| **Training notebook** | `CodeOrganismVM_Training.ipynb` (Unsloth and TRL oriented) |
| **Metrics and figures** | GitHub repository: `results/` and scripts under `training/` |
| **Upstream OpenEnv** | [meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv) |
| **Manifest** | `openenv.yaml` at repository root |

---

## Engineering commitments

| Commitment | Rationale |
|------------|-----------|
| **Client and server** separation | Application clients do not depend on server-only internals. |
| **`openenv.yaml` with reset, step, and state** | External trainers attach to a **stable** contract. |
| **Docker image equals Space runtime** | What judges run is what continuous integration builds. |

---

## Closing

We ask a model to remain with a **bruised codebase** until responses stabilise. That requires **implementation**, **measured curves**, and a **concise** narrative. Thank you for reading.

*Team Autonomous SRE*
