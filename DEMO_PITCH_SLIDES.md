# Autonomous SRE Demo Pitch (90-120s)

## Slide 1: Problem (0:00-0:15)
- Modern services fail in cascading ways.
- Current automation handles shallow failures, not long-horizon incident repair.
- Goal: train an agent that survives and recovers under chaos.

## Slide 2: Environment (0:15-0:35)
- OpenEnv-powered autonomous SRE environment.
- Self-corrupting service modules + multi-phase fault injection.
- Agent can patch, rollback, quarantine, signal, and request expert feedback.

## Slide 3: Learning Signals (0:35-0:55)
- Reward components: vitality, test recovery, efficiency, coordination, generalization.
- Anti-gaming penalties for no-op/signal farming.
- Held-out seed evaluation for phase 3.

## Slide 4: Evidence (0:55-1:20)
- Show `results/baseline_vs_agent.png`.
- Show `results/survival_by_phase.png`.
- Show extracted notebook training curve: `results/notebook_training_curve.png`.

## Slide 5: Live Demo Path (1:20-1:45)
- Open UI (`/ui`), reset phase profile.
- Run baseline episode button.
- Run heuristic episode button.
- Read postmortem timeline and reward trend.

## Slide 6: Why It Matters (1:45-2:00)
- This environment targets trainable incident response behavior, not static code completion.
- Reproducible scripts and artifacts are committed for judge verification.
