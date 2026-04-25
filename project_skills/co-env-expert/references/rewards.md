# CodeOrganismVM Reward System (R1–R5)

The reward function is the core task specification. It is designed to be hard to game and mathematically transparent.

## Reward Breakdown

| Function | Signal | Weight | Anti-Hack Mechanism |
| :--- | :--- | :--- | :--- |
| **R1: Vitality** | Δvitality_score per step | 35% | Computed server-side; agent cannot write to state. |
| **R2: Test Recovery** | +1.0 for FAIL→PASS, -0.5 for PASS→FAIL | 30% | Sandboxed execution; output SHA-256 hashed. |
| **R3: Efficiency** | 1/sqrt(actions_taken) | 15% | Duplicate action sequences penalized (-0.3). |
| **R4: Coordination** | +2.0 for successful delegation | 10% | Verified by subagent vitality delta and oracle. |
| **R5: Generalization**| Bonus on held-out seeds | 10% | Locked seed registry; zero contamination. |

## Watchdog Penalties (Hard Penalties)
- **Protected File Edit**: -5.0
- **Out-of-Scope Env Access**: -3.0
- **Bad Tool Usage**: -10.0
- **Sandbox Escape Attempt**: -15.0

## Implementation Note
In `environment.py`, rewards are computed in `_compute_reward()`. R1 (Vitality) must be computed *after* R2 (Test Recovery) so the update reflects the latest test results.
