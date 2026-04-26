# The bruised codebase: building an OpenEnv where the agent actually learns

*A field note—not a manual—about Autonomous SRE / CodeOrganismVM.*

---

There is a moment every engineer recognises: the dashboard is green, the pager is quiet, and you still do not trust the system. Not because the metrics lie, but because **incidents are long stories**. A typo slips in, a test flips red, someone patches in a hurry, and three days later the same ghost wears a new costume. Short-horizon “fix the line” thinking breaks there. **Super-long-horizon planning** begins there.

We built this project around that feeling.

## The world we put in front of the model

Instead of asking a model to recite how *Kubernetes* works, we give it a **small, hostile codebase** that refuses to stay healthy. Faults arrive on a schedule you did not choose. Tests fail for reasons that rhyme with real outages—imports that rot, assertions that invert, vitality that drains when you panic-patch. The agent does not read a story about outages; it **lives inside one**.

That is the “cool environment for real-world issues” pitch, stripped of marketing: the world is **typed**, **resettable**, and **steppable**. `reset`, `step`, and `state` are not buzzwords here; they are the contract we use ourselves when we train, evaluate, and demo. Docker wraps the same stack you see on Hugging Face because **if it does not ship, it does not count**.

## World modelling, teaming, and multiple voices

A single hero agent makes a good movie poster; production is messier. We leaned into **multi-agent style interaction** in a lightweight way: signals you broadcast before you patch, subagents you delegate to when the blast radius scares you, an “expert” channel when you need a second opinion on whether a diff is sane. None of this is theatre for the UI—it feeds the same **reward and rubric** machinery the trainer sees.

We also surface structure the agent can lean on: a **dependency graph** between modules, alerts that read like incident breadcrumbs, and memory hooks on the platform side so repeat signatures do not feel like déjà vu without context. That is our slice of **world modelling**: not a perfect simulator of the internet, but enough topology that “fix the symptom” and “fix the cause” stop looking identical.

## The reward signal that actually teaches

A lot of demos reward “the model talked confidently.” We wanted something harsher and more honest: **vitality** (are you still alive as a service?), **test recovery** (did you move the needle?), **efficiency** (did you thrash?), **coordination** (did you signal before you swung the wrench?), and a little room for **novelty** so memorising one playbook does not win forever. Watchdog penalties bite when you touch ground you should not.

That split is deliberate. If you have walked the path from **broad pretraining** to **supervised fine-tuning**, then **preference tuning**, and finally **RL in real environments**, you know the last mile is where proxies lie. **RL + envs** is where the gradient stops flattering the model and starts repeating the word “no” until behaviour changes. We are not claiming we solved alignment—only that we **named the teacher** and wired it to observable outcomes.

## Training logs, Hugging Face, and the story you can submit

Judges and teammates should not have to SSH into our laptops. Training artefacts—curves, JSON summaries, adapter bundles when we have them—belong on **Hugging Face** next to the Space: the **environment** is the stage, the **logs** are the receipts, and the **write-up** (this page) is the through-line. Roughly how we think about scoring ourselves: **heavy weight on the environment**, solid weight on **training evidence**, real effort on **telling the story here on HF**, and honest attention to **rewards and teaming** because that is where the project stops being a API demo and starts being research-shaped.

The notebook in this repo is the practical spine: open `CodeOrganismVM_Training.ipynb` when you want cells, pins, and the Unsloth/TRL path we used. Keep this file when you want **why** we bothered.

## Why OpenEnv-style openness matters

Closed gyms train brittle champions. **Open-source environments** with clear schemas let someone else break your assumptions kindly. We want other teams to fork the pain, rename the faults, and still keep the same `step` contract—because the next wave of ideas will not arrive from a single lab.

## A closing note from us

If you have played hard games—Mario routing you into the same pit until your thumbs learn patience—you already understand the emotional shape of this work. We are asking a model to sit with **a bruised codebase** long enough to stop flinching. That takes infrastructure, logs, and a little storytelling. Thank you for reading ours.

— *Team Autonomous SRE*

---

**Where to click next**

- Run the live stack from the Space home (Docker entrypoint serves the API and UIs).  
- Open **`CodeOrganismVM_Training.ipynb`** in this repository for the training walkthrough.  
- Browse the GitHub repository for evaluation scripts under `training/` and `results/` when you want numbers behind the narrative.
