# The bruised codebase: building an OpenEnv where the agent actually learns

*A field note—not a manual—about Autonomous SRE / CodeOrganismVM.*

---

There is a moment every engineer recognises: the dashboard is green, the pager is quiet, and you still do not trust the system. Not because the metrics lie, but because **incidents are long stories**. A typo slips in, a test flips red, someone patches in a hurry, and three days later the same ghost wears a new costume. Short-horizon “fix the line” thinking breaks there. **Super-long-horizon planning** begins there.

We built this project around that feeling—and around the hackathon’s invitation to treat **self-improvement**, **multi-agent coordination**, and **world modelling** as first-class themes, not afterthoughts.

## The world we put in front of the model

Instead of asking a model to recite how *Kubernetes* works, we give it a **small, hostile codebase** that refuses to stay healthy. Faults arrive on a schedule you did not choose. Tests fail for reasons that rhyme with real outages—imports that rot, assertions that invert, vitality that drains when you panic-patch. The agent does not read a story about outages; it **lives inside one**.

That is the “cool environment for real-world issues” pitch, stripped of marketing: the world is **typed**, **resettable**, and **steppable**. `reset`, `step`, and `state` are not buzzwords here; they are the contract we use ourselves when we train, evaluate, and demo. Docker wraps the same stack you see on Hugging Face because **if it does not ship, it does not count**.

We stand on **[OpenEnv](https://github.com/meta-pytorch/OpenEnv)**—the same mental model the opening session drew on: an agent loop where **experience** flows back as **state** and **reward**, where tools and memory sit next to policy updates instead of pretending the world is a CSV. Our `openenv.yaml` and server entrypoint are how we stay inside that ecosystem instead of reinventing a private toy gym.

## World modelling, teaming, and multiple voices

A single hero agent makes a good movie poster; production is messier. We leaned into **multi-agent style interaction** in a lightweight way: signals you broadcast before you patch, subagents you delegate to when the blast radius scares you, an “expert” channel when you need a second opinion on whether a diff is sane. None of this is theatre for the UI—it feeds the same **reward and rubric** machinery the trainer sees.

We also surface structure the agent can lean on: a **dependency graph** between modules, alerts that read like incident breadcrumbs, and memory hooks on the platform side so repeat signatures do not feel like déjà vu without context. That is our slice of **world modelling**: not a perfect simulator of the internet, but enough topology that “fix the symptom” and “fix the cause” stop looking identical.

## The reward signal that actually teaches

A lot of demos reward “the model talked confidently.” We wanted something harsher and more honest: **vitality** (are you still alive as a service?), **test recovery** (did you move the needle?), **efficiency** (did you thrash?), **coordination** (did you signal before you swung the wrench?), and a little room for **novelty** so memorising one playbook does not win forever. Watchdog penalties bite when you touch ground you should not.

The ceremony slides put it better than we could: a great signal is **rich**, sometimes **clever in what it measures**, and **hard to game**. We split scoring into **composable rubrics** on purpose—so “cheat the scalar” is harder than “patch like you mean it.”

If you have walked the path from **broad pretraining** to **supervised fine-tuning**, then **preference tuning**, and finally **RL in real environments**, you know the last mile is where proxies lie. **RL + envs** is where the gradient stops flattering the model and starts repeating the word “no” until behaviour changes. Above that loop lives the **harness** everyone is now designing for real systems: tools, memory, APIs, guardrails, observability—the boring words that decide whether learning survives contact with production.

## Training logs, Hugging Face, and the story you can submit

Judges and teammates should not have to SSH into our laptops. Training artefacts—**loss and reward curves**, JSON summaries, evaluation tables, adapter bundles when we have them—belong on **Hugging Face** next to the Space. We keep plots as **checked-in images** under `results/` so a reviewer does not have to resurrect a dead notebook cell to see a line move.

The organisers framed judging roughly as: **environment ambition first**, **story second**, **visible improvement third**, **pipeline coherence last**. We took that seriously internally: the Space is the living env; the notebook (`CodeOrganismVM_Training.ipynb`) is the **Unsloth / TRL** path we want people to re-run; the repo README ties the URLs together; this post is the **human** layer—because the brief was explicit: *tell a story, not an API doc.*

In plain language, our story answers the four questions they asked every team to carry:

1. **Problem** — LLMs need practice at *long* incident response under stress, not one-shot trivia.  
2. **Environment** — The agent **sees** tests, files, vitality, graphs, and platform telemetry; it **does** patches, rollbacks, quarantines, signals, and expert calls; it **gets rewarded** for rubric-shaped outcomes, not vibes.  
3. **Results** — We compare policies and training runs against **baselines** (noop, random, heuristic, SFT-style schedules) and publish the numbers and plots we trust.  
4. **Why it matters** — If we can train models to survive a self-sabotaging codebase toy, we learn how to teach them to survive messier real systems tomorrow.

## Table stakes we still cared about

None of this works if the plumbing lies. We kept a clean **client / server** split for API users, a valid **`openenv.yaml`**, and a standard **reset → step → state** surface so the environment stays a shared contract—not a private Python object only a notebook can touch. That is the unglamorous half of “ambitious”: the half that lets someone else **actually** train on what we built.

## Why openness matters

Closed gyms train brittle champions. **Open-source environments** with clear schemas let someone else break your assumptions kindly. We want other teams to fork the pain, rename the faults, and still keep the same `step` contract—because the next wave of ideas will not arrive from a single lab.

## A closing note from us

If you have played hard games—Mario routing you into the same pit until your thumbs learn patience—you already understand the emotional shape of this work. We are asking a model to sit with **a bruised codebase** long enough to stop flinching. That takes infrastructure, logs, and a little storytelling. Thank you for reading ours.

— *Team Autonomous SRE*

---

**Where to click next**

- **Live environment:** this Hugging Face Space (Docker serves the API, Gradio UI, and static console).  
- **Training notebook:** `CodeOrganismVM_Training.ipynb` in this repo (**Unsloth** + **TRL** oriented).  
- **Plots & metrics:** GitHub `results/` and evaluation scripts under `training/`.  
- **OpenEnv upstream:** [github.com/meta-pytorch/OpenEnv](https://github.com/meta-pytorch/OpenEnv)
