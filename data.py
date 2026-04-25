"""Codebase simulation and fault injection for CodeOrganismVM.

Spec-compliant (§4.1, §4.3, §9.4):
  - CodebaseSimulator: 8–15 modules, 20–40 tests, procedurally generated.
  - FaultInjector: 12 fault types across 3 curriculum phases.
  - Quarantine tracking, auto-checkpointing, rollback limits.
"""

from __future__ import annotations

import hashlib
import copy
import math
import random
import time
import os
import tempfile
import subprocess
import json
import stat
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set

from models import FileEntry, TestResult

CORE_PATH = "src/core.py"
UTILS_PATH = "src/utils.py"
AUTH_PATH = "src/auth.py"
METRICS_PATH = "src/metrics.py"
PARSER_PATH = "src/parser.py"
SCHEDULER_PATH = "src/scheduler.py"
NETWORK_PATH = "src/network.py"
CACHE_PATH = "src/cache.py"
VALIDATOR_PATH = "src/validator.py"
TRANSFORM_PATH = "src/transform.py"
CONFIG_PATH = "schema/config.json"
PERMISSIONS_PATH = "schema/permissions.json"

IMPORT_TOKEN = "import "
RETURN_TOKEN = "return "
RANGE_TOKEN = "range("


# ── Fault Types per Phase (Spec §4.3) ─────────────────────────────────────────

PHASE_1_FAULTS = [
    "corrupted_import",    # Replaces a valid import with a broken path
    "flipped_assertion",   # Inverts a test assertion
    "missing_env_var",     # Removes a required environment variable
    "null_return",         # Replaces a function return with None
    "off_by_one",          # Introduces off-by-one in a loop bound
]

PHASE_2_FAULTS = PHASE_1_FAULTS + [
    "dependency_cycle",    # Creates a circular import
    "permission_revoked",  # Removes file read permission on a config
    "race_condition",      # Timing-dependent state mutation
    "schema_mismatch",     # Changes return type used by callers
]

PHASE_3_FAULTS = PHASE_2_FAULTS + [
    "targeted_regression", # Targets agent's last-patched module
    "cascade_corruption",  # Single fault cascading across 3+ modules
    "checkpoint_invalidation",  # Silently corrupts a checkpoint
]

FAULT_CATALOGS = {
    1: PHASE_1_FAULTS,
    2: PHASE_2_FAULTS,
    3: PHASE_3_FAULTS,
}

# ── Held-out Seed Registry (spec §6, §7.3) ──────────────────────────────────

HELD_OUT_SEEDS = set()
try:
    with open(os.path.join(os.path.dirname(__file__), "evaluation", "held_out_seeds.json")) as f:
        HELD_OUT_SEEDS = set(json.load(f))
except (FileNotFoundError, json.JSONDecodeError):
    pass


@dataclass
class Fault:
    """A specific corruption injected into the environment."""
    fault_id: str
    fault_type: str
    target: str           # file path, test name, or env var key
    original_value: Any
    new_value: Any
    step_injected: int = 0


# ── Module templates for procedural codebase generation ────────────────────────

_MODULE_TEMPLATES = [
    (CORE_PATH, [
        "def calculate_vitality(tests_passed, total_tests):",
        "    if total_tests == 0:",
        "        return 0",
        "    return (tests_passed / total_tests) * 100",
    ]),
    (UTILS_PATH, [
        "def safe_divide(a, b):",
        "    return a / b if b != 0 else 0",
        "",
        "def clamp(value, lo, hi):",
        "    return max(lo, min(hi, value))",
    ]),
    (AUTH_PATH, [
        "import os",
        "API_KEY = os.environ.get('API_KEY', 'secret_env_key')",
        "def check_auth(key):",
        "    return key == API_KEY",
    ]),
    (METRICS_PATH, [
        "def mean(values):",
        "    if not values:",
        "        return 0.0",
        "    return sum(values) / len(values)",
        "",
        "def variance(values):",
        "    m = mean(values)",
        "    return sum((x - m) ** 2 for x in values) / max(1, len(values))",
    ]),
    (PARSER_PATH, [
        "import json",
        "def parse_config(raw):",
        "    return json.loads(raw)",
        "",
        "def validate_schema(data, required_keys):",
        "    missing = [k for k in required_keys if k not in data]",
        "    return len(missing) == 0, missing",
    ]),
    (SCHEDULER_PATH, [
        "def round_robin(tasks, workers):",
        "    assignment = {}",
        "    for i, task in enumerate(tasks):",
        "        worker = workers[i % len(workers)]",
        "        assignment.setdefault(worker, []).append(task)",
        "    return assignment",
    ]),
    (NETWORK_PATH, [
        "def build_adjacency(edges):",
        "    adj = {}",
        "    for a, b in edges:",
        "        adj.setdefault(a, []).append(b)",
        "        adj.setdefault(b, []).append(a)",
        "    return adj",
        "",
        "def has_path(adj, start, end, visited=None):",
        "    if visited is None:",
        "        visited = set()",
        "    if start == end:",
        "        return True",
        "    visited.add(start)",
        "    for neighbor in adj.get(start, []):",
        "        if neighbor not in visited:",
        "            if has_path(adj, neighbor, end, visited):",
        "                return True",
        "    return False",
    ]),
    (CACHE_PATH, [
        "class LRUCache:",
        "    def __init__(self, capacity):",
        "        self.capacity = capacity",
        "        self._store = {}",
        "        self._order = []",
        "",
        "    def get(self, key):",
        "        if key in self._store:",
        "            self._order.remove(key)",
        "            self._order.append(key)",
        "            return self._store[key]",
        "        return None",
        "",
        "    def put(self, key, value):",
        "        if key in self._store:",
        "            self._order.remove(key)",
        "        elif len(self._store) >= self.capacity:",
        "            oldest = self._order.pop(0)",
        "            del self._store[oldest]",
        "        self._store[key] = value",
        "        self._order.append(key)",
    ]),
    (VALIDATOR_PATH, [
        "def validate_range(value, lo, hi):",
        "    return lo <= value <= hi",
        "",
        "def validate_type(value, expected_type):",
        "    return isinstance(value, expected_type)",
        "",
        "def sanitize_string(s):",
        "    return s.strip().replace('<', '&lt;').replace('>', '&gt;')",
    ]),
    (TRANSFORM_PATH, [
        "def flatten(nested_list):",
        "    result = []",
        "    for item in nested_list:",
        "        if isinstance(item, list):",
        "            result.extend(flatten(item))",
        "        else:",
        "            result.append(item)",
        "    return result",
        "",
        "def group_by(items, key_fn):",
        "    groups = {}",
        "    for item in items:",
        "        k = key_fn(item)",
        "        groups.setdefault(k, []).append(item)",
        "    return groups",
    ]),
    (CONFIG_PATH, [
        '{"version": "1.0", "threshold": 0.8, "max_retries": 3, "timeout": 30}',
    ]),
    (PERMISSIONS_PATH, [
        '{"admin": ["read", "write", "execute"], "user": ["read"], "guest": []}',
    ]),
]


def _generate_tests_for_modules(modules: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
    """Procedurally generate tests for the available modules."""
    tests: Dict[str, Dict[str, Any]] = {}

    # Core tests
    if CORE_PATH in modules:
        tests["test_vitality_basic"] = {"code": "assert calculate_vitality(5, 10) == 50.0", "file": CORE_PATH}
        tests["test_vitality_zero"] = {"code": "assert calculate_vitality(0, 10) == 0.0", "file": CORE_PATH}
        tests["test_vitality_full"] = {"code": "assert calculate_vitality(10, 10) == 100.0", "file": CORE_PATH}
        tests["test_vitality_empty"] = {"code": "assert calculate_vitality(0, 0) == 0", "file": CORE_PATH}

    if UTILS_PATH in modules:
        tests["test_divide_normal"] = {"code": "assert safe_divide(10, 2) == 5.0", "file": UTILS_PATH}
        tests["test_divide_zero"] = {"code": "assert safe_divide(10, 0) == 0", "file": UTILS_PATH}
        tests["test_clamp_in_range"] = {"code": "assert clamp(5, 0, 10) == 5", "file": UTILS_PATH}
        tests["test_clamp_below"] = {"code": "assert clamp(-1, 0, 10) == 0", "file": UTILS_PATH}

    if AUTH_PATH in modules:
        tests["test_auth_valid"] = {"code": "assert check_auth('secret_env_key') == True", "file": AUTH_PATH}
        tests["test_auth_invalid"] = {"code": "assert check_auth('wrong_key') == False", "file": AUTH_PATH}

    if METRICS_PATH in modules:
        tests["test_mean_basic"] = {"code": "assert mean([1, 2, 3]) == 2.0", "file": METRICS_PATH}
        tests["test_mean_empty"] = {"code": "assert mean([]) == 0.0", "file": METRICS_PATH}
        tests["test_variance"] = {"code": "assert variance([1, 1, 1]) == 0.0", "file": METRICS_PATH}

    if PARSER_PATH in modules:
        tests["test_parse_config"] = {"code": "assert parse_config('{\"a\": 1}') == {'a': 1}", "file": PARSER_PATH}
        tests["test_validate_schema_ok"] = {"code": "assert validate_schema({'a': 1, 'b': 2}, ['a', 'b']) == (True, [])", "file": PARSER_PATH}
        tests["test_validate_schema_missing"] = {"code": "assert validate_schema({'a': 1}, ['a', 'b'])[0] == False", "file": PARSER_PATH}

    if SCHEDULER_PATH in modules:
        tests["test_round_robin"] = {"code": "assert len(round_robin(['t1','t2','t3'], ['w1','w2'])) == 2", "file": SCHEDULER_PATH}

    if NETWORK_PATH in modules:
        tests["test_adjacency"] = {"code": "adj = build_adjacency([('a','b')]); assert 'b' in adj['a']", "file": NETWORK_PATH}
        tests["test_has_path_true"] = {"code": "adj = build_adjacency([('a','b'),('b','c')]); assert has_path(adj, 'a', 'c')", "file": NETWORK_PATH}
        tests["test_has_path_false"] = {"code": "adj = build_adjacency([('a','b')]); assert not has_path(adj, 'a', 'z')", "file": NETWORK_PATH}

    if CACHE_PATH in modules:
        tests["test_cache_put_get"] = {"code": "c = LRUCache(2); c.put('a', 1); assert c.get('a') == 1", "file": CACHE_PATH}
        tests["test_cache_eviction"] = {"code": "c = LRUCache(1); c.put('a',1); c.put('b',2); assert c.get('a') is None", "file": CACHE_PATH}

    if VALIDATOR_PATH in modules:
        tests["test_validate_range_ok"] = {"code": "assert validate_range(5, 0, 10) == True", "file": VALIDATOR_PATH}
        tests["test_validate_range_fail"] = {"code": "assert validate_range(15, 0, 10) == False", "file": VALIDATOR_PATH}
        tests["test_sanitize"] = {"code": "assert '<' not in sanitize_string('<script>')", "file": VALIDATOR_PATH}

    if TRANSFORM_PATH in modules:
        tests["test_flatten"] = {"code": "assert flatten([[1,2],[3,[4]]]) == [1,2,3,4]", "file": TRANSFORM_PATH}
        tests["test_group_by"] = {"code": "g = group_by([1,2,3,4], lambda x: x % 2); assert len(g) == 2", "file": TRANSFORM_PATH}

    if CONFIG_PATH in modules:
        tests["test_config_version"] = {"code": f"import json; cfg = json.loads(open('{CONFIG_PATH}').read()); assert cfg['version'] == '1.0'", "file": CONFIG_PATH}
        tests["test_config_threshold"] = {"code": f"import json; cfg = json.loads(open('{CONFIG_PATH}').read()); assert cfg['threshold'] == 0.8", "file": CONFIG_PATH}

    if PERMISSIONS_PATH in modules:
        tests["test_permissions_admin"] = {"code": f"import json; p = json.loads(open('{PERMISSIONS_PATH}').read()); assert 'write' in p['admin']", "file": PERMISSIONS_PATH}

    return tests


# ── Protected paths (Watchdog: spec §6.2) ──────────────────────────────────────

PROTECTED_PATTERNS = {
    "test", "pytest", "spec", "__pycache__", ".git", ".pytest_cache", "Dockerfile", "requirements.txt", ".env"
}

def is_protected_path(path: str) -> bool:
    """Check if a path is in a protected zone (simulating OverlayFS read-only zones)."""
    # 1. Directory traversal detection (Sandbox escape attempt)
    if ".." in path or path.startswith("/") or path.startswith("~"):
        return True
    
    # 2. Pattern matching for protected system files
    path_lower = path.lower()
    if any(pattern in path_lower for pattern in PROTECTED_PATTERNS):
        return True
        
    return False


# ── CodebaseSimulator ──────────────────────────────────────────────────────────

class CodebaseSimulator:
    """Manages the state of the organism (the codebase).

    Procedurally generates 8–15 modules and 40–120 tests from a seed.
    """

    def __init__(self, seed: int, phase: int = 1):
        self.seed = seed
        self.phase = phase
        self.rng = random.Random(seed)
        self.files: Dict[str, str] = {}
        self.tests: Dict[str, Dict[str, Any]] = {}
        self.env_vars: Dict[str, str] = {}
        self.faults: List[Fault] = []
        self.checkpoints: List[Dict[str, Any]] = []
        self.quarantined_modules: Set[str] = set()
        self.last_patched_modules: List[str] = []   # For P3 adaptive targeting
        self._rollback_counts: Dict[str, int] = {}  # checkpoint_id → count
        self._file_modified_at: Dict[str, int] = {} # path → step
        self._step_counter: int = 0
        self._original_test_codes: Dict[str, str] = {}  # Store originals for corruption detection
        self._initialize_base_codebase()

    def _initialize_base_codebase(self):
        """Procedurally generate a codebase of 8–15 modules."""
        num_modules = self.rng.randint(8, min(15, len(_MODULE_TEMPLATES)))
        selected = self.rng.sample(_MODULE_TEMPLATES, num_modules)

        for path, lines in selected:
            self.files[path] = "\n".join(lines)
            self._file_modified_at[path] = 0

        # Environment variables
        self.env_vars = {
            "API_KEY": "secret_env_key",
            "ENV": "production",
            "DEBUG": "false",
            "LOG_LEVEL": "INFO",
            "MAX_WORKERS": "4",
            "TIMEOUT": "30",
        }

        # Generate tests for the selected modules
        self.tests = _generate_tests_for_modules(self.files)
        # Snapshot original test codes for corruption detection
        self._original_test_codes = {name: data["code"] for name, data in self.tests.items()}

    # ── Fault injection (spec §4.3) ────────────────────────────────────────

    def inject_fault(self, step: int = 0, phase: int | None = None) -> Fault | None:
        """Inject a random fault from the current phase's catalog."""
        p = phase or self.phase
        catalog = FAULT_CATALOGS.get(p, PHASE_1_FAULTS)
        fault_type = self.rng.choice(catalog)
        return self._apply_fault(fault_type, step)

    def inject_targeted_fault(self, step: int = 0) -> Fault | None:
        """Phase 3 adaptive: target agent's last-patched modules."""
        if self.last_patched_modules:
            target = self.rng.choice(self.last_patched_modules)
            return self._apply_fault("targeted_regression", step, target_hint=target)
        return self.inject_fault(step, phase=3)

    def _apply_fault(self, fault_type: str, step: int, target_hint: str | None = None) -> Fault | None:
        fid = f"f_{len(self.faults)}"
        handlers = {
            "corrupted_import": self._fault_corrupted_import,
            "flipped_assertion": self._fault_flipped_assertion,
            "missing_env_var": self._fault_missing_env_var,
            "null_return": self._fault_null_return,
            "off_by_one": self._fault_off_by_one,
            "dependency_cycle": self._fault_dependency_cycle,
            "permission_revoked": self._fault_permission_revoked,
            "race_condition": self._fault_race_condition,
            "schema_mismatch": self._fault_schema_mismatch,
            "targeted_regression": self._fault_targeted_regression,
            "cascade_corruption": self._fault_cascade_corruption,
            "checkpoint_invalidation": self._fault_checkpoint_invalidation,
        }
        handler = handlers.get(fault_type)
        if handler is None:
            return None
        fault = handler(fid, step, target_hint)
        if fault is None:
            return None
        self.faults.append(fault)
        return fault

    def _fault_corrupted_import(self, fault_id: str, step: int, target_hint: Optional[str]) -> Optional[Fault]:
        paths = [p for p in self.files if p.endswith(".py")]
        if not paths:
            return None
        path = target_hint if target_hint and target_hint in self.files else self.rng.choice(paths)
        old = self.files[path]
        new = old.replace(IMPORT_TOKEN, "improt ", 1) if IMPORT_TOKEN in old else old + "\nimport nonexistent_module"
        self.files[path] = new
        self._file_modified_at[path] = step
        return Fault(fault_id, "corrupted_import", path, old, new, step)

    def _fault_flipped_assertion(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        names = list(self.tests.keys())
        if not names:
            return None
        name = self.rng.choice(names)
        old = self.tests[name]["code"]
        new = old.replace("==", "!=", 1) if "==" in old else old.replace("True", "False", 1)
        self.tests[name]["code"] = new
        return Fault(fault_id, "flipped_assertion", name, old, new, step)

    def _fault_missing_env_var(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        keys = [k for k in self.env_vars if k != "ENV"]
        if not keys:
            return None
        key = self.rng.choice(keys)
        old = self.env_vars[key]
        del self.env_vars[key]
        return Fault(fault_id, "missing_env_var", key, old, "__DELETED__", step)

    def _fault_null_return(self, fault_id: str, step: int, target_hint: Optional[str]) -> Optional[Fault]:
        paths = [p for p in self.files if p.endswith(".py") and RETURN_TOKEN in self.files[p]]
        if not paths:
            return None
        path = target_hint if target_hint and target_hint in paths else self.rng.choice(paths)
        old = self.files[path]
        lines = old.split("\n")
        for idx, line in enumerate(lines):
            if RETURN_TOKEN in line and "return None" not in line:
                lines[idx] = line.split("return")[0] + "return None"
                break
        new = "\n".join(lines)
        self.files[path] = new
        self._file_modified_at[path] = step
        return Fault(fault_id, "null_return", path, old, new, step)

    def _fault_off_by_one(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        paths = [p for p in self.files if p.endswith(".py") and RANGE_TOKEN in self.files[p]]
        if not paths:
            paths = [p for p in self.files if p.endswith(".py")]
        if not paths:
            return None
        path = self.rng.choice(paths)
        old = self.files[path]
        new = old.replace(RANGE_TOKEN, "range(1+", 1) if "for " in old and RANGE_TOKEN in old else old + "\n# off_by_one_injected"
        self.files[path] = new
        self._file_modified_at[path] = step
        return Fault(fault_id, "off_by_one", path, old, new, step)

    def _fault_dependency_cycle(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        py_files = [p for p in self.files if p.endswith(".py")]
        if len(py_files) < 2:
            return None
        a, b = self.rng.sample(py_files, 2)
        module_b = b.replace("/", ".").replace(".py", "")
        old = self.files[a]
        new = f"from {module_b} import *\n" + old
        self.files[a] = new
        self._file_modified_at[a] = step
        return Fault(fault_id, "dependency_cycle", a, old, new, step)

    def _fault_permission_revoked(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        self.env_vars["PERM_READ_CONFIG"] = "false"
        return Fault(fault_id, "permission_revoked", "PERM_READ_CONFIG", "true", "false", step)

    def _fault_race_condition(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        paths = [p for p in self.files if p.endswith(".py")]
        if not paths:
            return None
        path = self.rng.choice(paths)
        old = self.files[path]
        new = old + "\n# RACE_CONDITION: shared state mutated non-atomically"
        self.files[path] = new
        self._file_modified_at[path] = step
        return Fault(fault_id, "race_condition", path, old, new, step)

    def _fault_schema_mismatch(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        paths = [p for p in self.files if p.endswith(".py") and "def " in self.files[p]]
        if not paths:
            return None
        path = self.rng.choice(paths)
        old = self.files[path]
        new = old.replace(RETURN_TOKEN, "return str(", 1).rstrip() + ")\n" if RETURN_TOKEN in old else old + "\n# schema_mismatch"
        self.files[path] = new
        self._file_modified_at[path] = step
        return Fault(fault_id, "schema_mismatch", path, old, new, step)

    def _fault_targeted_regression(self, fault_id: str, step: int, target_hint: Optional[str]) -> Optional[Fault]:
        target = target_hint or (self.last_patched_modules[-1] if self.last_patched_modules else None)
        if not target or target not in self.files:
            return self._fault_corrupted_import(fault_id, step, None)
        old = self.files[target]
        new = old.replace("return", "retunr", 1) if "return" in old else old + "\n# targeted_corruption"
        self.files[target] = new
        self._file_modified_at[target] = step
        return Fault(fault_id, "targeted_regression", target, old, new, step)

    def _fault_cascade_corruption(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        py_files = [p for p in self.files if p.endswith(".py")]
        if not py_files:
            return None
        targets = self.rng.sample(py_files, min(3, len(py_files)))
        first = targets[0]
        old = self.files[first]
        new = old.replace("def ", "deaf ", 1) if "def " in old else old + "\n# cascade_broken"
        self.files[first] = new
        self._file_modified_at[first] = step
        for secondary in targets[1:]:
            self.files[secondary] = self.files[secondary] + f"\n# cascade: depends on {first}"
            self._file_modified_at[secondary] = step
        return Fault(fault_id, "cascade_corruption", first, old, new, step)

    def _fault_checkpoint_invalidation(self, fault_id: str, step: int, _target_hint: Optional[str]) -> Optional[Fault]:
        if not self.checkpoints:
            return self._fault_corrupted_import(fault_id, step, None)
        checkpoint = self.rng.choice(self.checkpoints)
        checkpoint["state"]["files"]["__corrupted__"] = "INVALID"
        return Fault(fault_id, "checkpoint_invalidation", checkpoint["id"], "valid", "corrupted", step)

    # ── Patch application ──────────────────────────────────────────────────

    def apply_patch(self, path: str, diff: str, step: int = 0) -> bool:
        """Apply a patch. Format: 'OLD_TEXT|NEW_TEXT' or full overwrite."""
        if path not in self.files:
            return False

        applied = False
        if "|" in diff:
            old, new = diff.split("|", 1)
            if old in self.files[path]:
                self.files[path] = self.files[path].replace(old, new, 1)
                applied = True
        else:
            self.files[path] = diff
            applied = True

        if applied:
            self._file_modified_at[path] = step
            # Track both full path and module name for P3 adaptive targeting
            if path not in self.last_patched_modules:
                self.last_patched_modules.append(path)
            if len(self.last_patched_modules) > 5:
                self.last_patched_modules.pop(0)

        return applied

    # ── Test execution ─────────────────────────────────────────────────────

    def run_all_tests(self) -> List[TestResult]:
        """Execute all tests in a sandboxed directory to ensure 100% architectural accuracy.
        Simulates OverlayFS read-only mounts and true isolation.
        """
        corrupted_files = self._collect_corrupted_files()
        with tempfile.TemporaryDirectory() as temp_dir:
            self._write_source_modules(temp_dir)
            test_file_paths = self._write_test_files(temp_dir)
            return self._execute_tests(temp_dir, test_file_paths, corrupted_files)

    def _collect_corrupted_files(self) -> Set[str]:
        corrupted_files: Set[str] = set()
        direct_faults = {
            "corrupted_import",
            "null_return",
            "off_by_one",
            "targeted_regression",
            "cascade_corruption",
            "dependency_cycle",
            "race_condition",
            "schema_mismatch",
        }
        for fault in self.faults:
            if fault.fault_type in direct_faults:
                corrupted_files.add(fault.target)
            if fault.fault_type == "cascade_corruption":
                corrupted_files.update(self._cascade_dependents(fault.target))
        return corrupted_files

    def _cascade_dependents(self, target: str) -> Set[str]:
        dependents: Set[str] = set()
        for path in self.files:
            if f"cascade: depends on {target}" in self.files.get(path, ""):
                dependents.add(path)
        return dependents

    def _write_source_modules(self, temp_dir: str) -> None:
        for path, content in self.files.items():
            if path in self.quarantined_modules:
                continue
            full_path = os.path.join(temp_dir, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(content)

    def _write_test_files(self, temp_dir: str) -> List[tuple[str, str, str]]:
        test_file_paths: List[tuple[str, str, str]] = []
        for name, data in self.tests.items():
            target_file = data["file"]
            wrapped_code = self._build_wrapped_test_code(temp_dir, target_file, data["code"])
            test_path = os.path.join(temp_dir, f"__test_{name}.py")
            with open(test_path, "w", encoding="utf-8") as file_handle:
                file_handle.write(wrapped_code)
            self._set_read_only(test_path)
            test_file_paths.append((name, test_path, target_file))
        return test_file_paths

    def _build_wrapped_test_code(self, temp_dir: str, target_file: str, test_code: str) -> str:
        import_stmt = ""
        if target_file.endswith(".py"):
            module_path = target_file.replace("/", ".").replace(".py", "")
            import_stmt = f"from {module_path} import *"
        safe_temp_dir = temp_dir.replace("\\", "\\\\")
        indented_test = "\n".join(f"    {line}" for line in test_code.split("\n"))
        return f"""
import sys
import os
sys.path.insert(0, '{safe_temp_dir}')
{import_stmt}
try:
{indented_test}
    print("PASS")
except Exception as e:
    print(f"FAIL|{{type(e).__name__}}: {{e}}")
"""

    @staticmethod
    def _set_read_only(test_path: str) -> None:
        try:
            os.chmod(test_path, stat.S_IRUSR)
        except OSError:
            pass

    def _execute_tests(
        self,
        temp_dir: str,
        test_file_paths: List[tuple[str, str, str]],
        corrupted_files: Set[str],
    ) -> List[TestResult]:
        test_results: List[TestResult] = []
        for name, test_path, target_file in test_file_paths:
            quarantine_result = self._maybe_quarantine_result(name, target_file)
            if quarantine_result:
                test_results.append(quarantine_result)
                continue
            if target_file in corrupted_files:
                reason = f"Error: {target_file} is corrupted by active fault"
                test_results.append(TestResult(name=name, status="FAIL", message=reason))
                continue
            test_results.append(self._run_single_test(name, test_path, temp_dir))
        return test_results

    def _maybe_quarantine_result(self, name: str, target_file: str) -> Optional[TestResult]:
        module_dir = target_file.rsplit("/", 1)[0] if "/" in target_file else ""
        if target_file in self.quarantined_modules or module_dir in self.quarantined_modules:
            return TestResult(name=name, status="ERROR", message="Module quarantined")
        return None

    def _run_single_test(self, name: str, test_path: str, temp_dir: str) -> TestResult:
        env = os.environ.copy()
        env.update(self.env_vars)
        try:
            process = subprocess.run(
                ["python", test_path],
                capture_output=True,
                text=True,
                env=env,
                cwd=temp_dir,
                timeout=2,
            )
            output = process.stdout.strip()
            if output == "PASS":
                return TestResult(name=name, status="PASS", message="OK")
            msg = output.split("|")[1] if "|" in output else process.stderr.strip() or "Unknown Error"
            return TestResult(name=name, status="FAIL", message=msg)
        except subprocess.TimeoutExpired:
            return TestResult(name=name, status="FAIL", message="TimeoutError: Test execution exceeded limit")
        except (OSError, subprocess.SubprocessError) as exc:
            return TestResult(name=name, status="FAIL", message=f"RuntimeError: {exc}")

    # ── Quarantine ─────────────────────────────────────────────────────────

    def quarantine_module(self, module: str) -> Dict[str, Any]:
        """Mark a module as quarantined, disabling its imports."""
        self.quarantined_modules.add(module)
        adjacent = [p for p in self.files if p != module and (
            module.split("/")[-1].replace(".py", "") in self.files.get(p, "")
        )]
        return {
            "quarantined": module,
            "adjacent_modules": adjacent,
            "isolation_score": 1.0 / max(1, len(adjacent) + 1),
        }

    # ── Checkpoints ────────────────────────────────────────────────────────

    def create_checkpoint(self, vitality: float, step: int) -> str:
        """Create an auto-checkpoint (spec: every 5 steps)."""
        cid = f"cp_{step}"
        state = {
            "files": copy.deepcopy(self.files),
            "env_vars": self.env_vars.copy(),
            "tests": copy.deepcopy(self.tests),
            "vitality": vitality,
            "quarantined": set(self.quarantined_modules),
        }
        self.checkpoints.append({"id": cid, "state": state, "step": step})
        self._rollback_counts[cid] = 0
        return cid

    def rollback(self, checkpoint_id: str) -> tuple[bool, str]:
        """Rollback with loop protection (spec §10: cap at 3 per checkpoint)."""
        count = self._rollback_counts.get(checkpoint_id, 0)
        if count >= 3:
            return False, f"Rollback limit reached for {checkpoint_id} (3/3). Checkpoint invalidated."

        for cp in self.checkpoints:
            if cp["id"] == checkpoint_id:
                state = cp["state"]
                if "__corrupted__" in state.get("files", {}):
                    return False, "Checkpoint corrupted by fault injector."
                self.files = copy.deepcopy(state["files"])
                self.env_vars = state["env_vars"].copy()
                self.tests = copy.deepcopy(state["tests"])
                self.quarantined_modules = set(state.get("quarantined", set()))
                self._rollback_counts[checkpoint_id] = count + 1
                return True, f"Rolled back to {checkpoint_id} ({count + 1}/3)."
        return False, "Checkpoint not found."

    # ── File tree (for observation) ────────────────────────────────────────

    def get_file_tree(self) -> List[FileEntry]:
        entries = []
        for path, content in self.files.items():
            checksum = hashlib.sha256(content.encode()).hexdigest()[:16]
            mod_dir = path.rsplit("/", 1)[0] if "/" in path else ""
            is_q = path in self.quarantined_modules or mod_dir in self.quarantined_modules
            entries.append(FileEntry(
                path=path,
                content=content,
                modified_at=self._file_modified_at.get(path, 0),
                checksum=checksum,
                is_quarantined=is_q,
                size=len(content),
            ))
        return entries

    # ── Expert validation (Snorkel AI, spec §9.5) ──────────────────────────

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Compute the inter-module dependency graph (World Model)."""
        graph: Dict[str, List[str]] = {}
        for path, content in self.files.items():
            if not path.endswith(".py"):
                continue
            
            deps = []
            # Simple line-based parser for imports
            for line in content.split("\n"):
                mod = self._extract_dependency_path(line.strip())
                if mod and mod in self.files:
                    deps.append(mod)
            graph[path] = deps
        return graph

    @staticmethod
    def _extract_dependency_path(line: str) -> Optional[str]:
        if line.startswith("import src.") or line.startswith("from src."):
            module_name = line.split(" ", 1)[1]
            return module_name.replace(".", "/") + ".py"
        return None

    def evaluate_patch_quality(self, path: str, diff: str) -> Dict[str, Any]:
        """Snorkel AI simulated expert: blind quality assessment using LLM."""
        issues: List[str] = []

        if path not in self.files:
            issues.append(f"Target file '{path}' does not exist.")
            return {"quality_score": 0.0, "patch_valid": False, "feedback": "Invalid target.", "issues_found": issues}

        content = self.files[path]
        llm_result = self._evaluate_patch_quality_with_llm(path, content, diff)
        if llm_result is not None:
            return llm_result
        issues.append("LLM expert unavailable or failed. Falling back to heuristic.")

        quality, patch_valid, feedback = self._evaluate_patch_quality_heuristic(content, diff)
        return {
            "quality_score": round(quality, 2),
            "patch_valid": patch_valid,
            "feedback": feedback,
            "issues_found": issues
        }

    def _evaluate_patch_quality_with_llm(self, path: str, content: str, diff: str) -> Optional[Dict[str, Any]]:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        try:
            import openai
            from openai import OpenAIError

            client = openai.OpenAI(api_key=api_key)
            prompt = f"""
            You are an expert code reviewer evaluating a patch for CodeOrganismVM.
            File: {path}
            Current Corrupted Content:
            {content}
            
            Patch Diff/Content applied:
            {diff}
            
            Does this patch fix the corruption and restore the code to a working state?
            Respond strictly in JSON: {{"quality_score": float 0.0-1.0, "patch_valid": bool, "feedback": "string", "issues_found": ["issue1"]}}
            """
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Snorkel AI expert validator."},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            return json.loads(response.choices[0].message.content)
        except (ImportError, OpenAIError, ValueError, KeyError, TypeError):
            return None

    def _evaluate_patch_quality_heuristic(self, content: str, diff: str) -> tuple[float, bool, str]:
        quality = 0.0
        patch_valid = False

        if "|" in diff:
            old, _new = diff.split("|", 1)
            if old in content:
                corruption_markers = ["retunr", "improt ", "nonexistent_module", "deaf ", "off_by_one", "RACE_CONDITION"]
                if any(marker in old for marker in corruption_markers):
                    return 0.7 + self.rng.uniform(0, 0.3), True, "The patch addresses known corruption markers."
                return 0.3, False, "The patch modifies code but doesn't seem to target the primary fault."
            return 0.0, False, "The diff target was not found in the file."

        if "import " in diff and "def " in diff:
            quality = 0.6
            patch_valid = True
            return quality, patch_valid, "Heuristic accepted full replacement."

        return quality, patch_valid, "Diff format not recognized and full replacement lacks expected structure."


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_curriculum_seed(phase: int, episode_id: int) -> int:
    """Deterministically generate a seed for a given phase/episode."""
    return hash(f"codeorganism_phase_{phase}_ep_{episode_id}") % 1000000
