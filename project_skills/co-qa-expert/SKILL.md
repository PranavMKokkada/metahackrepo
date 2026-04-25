---
name: co-qa-expert
description: Expert guidance for testing, validation, and spec compliance of the CodeOrganismVM environment. Use when running validation suites, debugging environment crashes, or ensuring alignment with hackathon requirements (§4, §5, §6).
---

# CodeOrganismVM Quality Assurance Expert

This skill ensures that the environment and training pipeline remain compliant with the technical specification.

## Core Components

- **validate.py**: Script to check environment compliance against the spec.
- **test_env.py**: Pytest suite for environment dynamics, rewards, and fault injection.
- **test_api.py**: Tests for the FastAPI server and OpenEnv endpoints.
- **debug_reset.py**: Utility for inspecting environment state after a reset.

## Key Workflows

### 1. Running Validation
- Refer to [spec_compliance.md](references/spec_compliance.md) for a checklist of mandatory requirements.
- Always run `python validate.py` after modifying `environment.py` or `data.py`.

### 2. Environment Debugging
- If the environment crashes during a `step()`, use `debug_reset.py` to check the `file_tree` and `test_results` of the initial state.
- Check the `watchdog_flags` in the observation to see if a policy violation caused the failure.

### 3. Adding New Tests
- When adding a new fault type, add a corresponding test case in `test_env.py` that verifies the fault is detectable and patchable.
- Ensure that the reward signal (R1–R5) remains within the expected range [-1.0, 1.0] for typical actions.

## Best Practices
- **No Flakiness**: Use the `flaky_test_detector` before running large batches of RL rollouts to exclude non-deterministic tests.
- **Isolation**: Always run tests in a clean environment (e.g., using the Docker container) to prevent local state pollution.
- **Spec First**: If a requested change conflicts with the tech spec (§4, §5, §6), flag the discrepancy before implementing.
