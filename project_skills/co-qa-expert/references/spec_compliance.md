# CodeOrganismVM Spec Compliance Checklist

Based on the technical specification (spec_text.txt, CodeOrganismVM_Complete_Spec.pdf).

## §4: Environment Design
- [ ] **Observation Space**: Includes `vitality_score`, `test_results`, `file_tree`, `stack_trace`, `active_checkpoints`.
- [ ] **Action Space**: 7 actions (patch_file, run_tests, spawn_subagent, quarantine, rollback, request_expert, emit_signal).
- [ ] **Reset Logic**: Generates fresh broken codebase from seeded procedural generator.

## §5: Self-* Properties
- [ ] **Self-Heals**: Measurable time-to-heal and vitality recovery.
- [ ] **Self-Corrects**: Accurate `rollback()` issuance on regressions.
- [ ] **Self-Replicates**: Successful `spawn_subagent()` delegation.

## §6: Reward Engineering
- [ ] **Weights**: R1 (35%), R2 (30%), R3 (15%), R4 (10%), R5 (10%).
- [ ] **Anti-Hacking**: Watchdog penalties for protected file edits (-5.0) and sandbox escapes.

## §9: Technical Implementation
- [ ] **OpenEnv API**: Exposes `/reset`, `/step`, `/state`, `/health` endpoints.
- [ ] **Containerization**: Isolated Docker execution per episode.
