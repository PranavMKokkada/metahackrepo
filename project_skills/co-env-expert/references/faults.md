# CodeOrganismVM Fault Catalog

Faults are injected by the `FaultInjector` in `data.py` to corrupt the codebase and challenge the organism.

## Phase-Aware Catalog

### Phase 1: Basic Faults (Single)
- `corrupted_import`: Replaces valid import with broken path.
- `flipped_assertion`: Inverts test assertion (True -> False).
- `missing_env_var`: Removes required env var.
- `null_return`: Replaces return with `None`.
- `off_by_one`: Introduces off-by-one in loops.

### Phase 2: Structural Faults (Multi)
- `dependency_cycle`: Circular imports between modules.
- `permission_revoked`: Removes read permission on config files.
- `race_condition`: Timing-dependent state mutation.
- `schema_mismatch`: Changes return type of critical functions.

### Phase 3: Adversarial Faults
- `targeted_regression`: Fault in a module the agent recently patched.
- `cascade_corruption`: Single fault triggering a chain of 3+ failures.
- `checkpoint_invalidation`: Silently corrupts an existing checkpoint.

## Injection Logic
- **Intervals**: Phase 1 (8 steps), Phase 2 (6 steps), Phase 3 (4 steps).
- **Adversarial Mode**: In Phase 3, the `FaultInjector` is reactive, targeting the agent's most recent repairs.
