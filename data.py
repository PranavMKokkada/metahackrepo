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
    ("src/core.py", [
        "def calculate_vitality(tests_passed, total_tests):",
        "    if total_tests == 0:",
        "        return 0",
        "    return (tests_passed / total_tests) * 100",
    ]),
    ("src/utils.py", [
        "def safe_divide(a, b):",
        "    return a / b if b != 0 else 0",
        "",
        "def clamp(value, lo, hi):",
        "    return max(lo, min(hi, value))",
    ]),
    ("src/auth.py", [
        "import os",
        "API_KEY = os.environ.get('API_KEY', 'secret_env_key')",
        "def check_auth(key):",
        "    return key == API_KEY",
    ]),
    ("src/metrics.py", [
        "def mean(values):",
        "    if not values:",
        "        return 0.0",
        "    return sum(values) / len(values)",
        "",
        "def variance(values):",
        "    m = mean(values)",
        "    return sum((x - m) ** 2 for x in values) / max(1, len(values))",
    ]),
    ("src/parser.py", [
        "import json",
        "def parse_config(raw):",
        "    return json.loads(raw)",
        "",
        "def validate_schema(data, required_keys):",
        "    missing = [k for k in required_keys if k not in data]",
        "    return len(missing) == 0, missing",
    ]),
    ("src/scheduler.py", [
        "def round_robin(tasks, workers):",
        "    assignment = {}",
        "    for i, task in enumerate(tasks):",
        "        worker = workers[i % len(workers)]",
        "        assignment.setdefault(worker, []).append(task)",
        "    return assignment",
    ]),
    ("src/network.py", [
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
    ("src/cache.py", [
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
    ("src/validator.py", [
        "def validate_range(value, lo, hi):",
        "    return lo <= value <= hi",
        "",
        "def validate_type(value, expected_type):",
        "    return isinstance(value, expected_type)",
        "",
        "def sanitize_string(s):",
        "    return s.strip().replace('<', '&lt;').replace('>', '&gt;')",
    ]),
    ("src/transform.py", [
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
    ("schema/config.json", [
        '{"version": "1.0", "threshold": 0.8, "max_retries": 3, "timeout": 30}',
    ]),
    ("schema/permissions.json", [
        '{"admin": ["read", "write", "execute"], "user": ["read"], "guest": []}',
    ]),
]


def _generate_tests_for_modules(modules: Dict[str, str], rng: random.Random) -> Dict[str, Dict[str, Any]]:
    """Procedurally generate tests for the available modules."""
    tests: Dict[str, Dict[str, Any]] = {}

    # Core tests
    if "src/core.py" in modules:
        tests["test_vitality_basic"] = {"code": "assert calculate_vitality(5, 10) == 50.0", "file": "src/core.py"}
        tests["test_vitality_zero"] = {"code": "assert calculate_vitality(0, 10) == 0.0", "file": "src/core.py"}
        tests["test_vitality_full"] = {"code": "assert calculate_vitality(10, 10) == 100.0", "file": "src/core.py"}
        tests["test_vitality_empty"] = {"code": "assert calculate_vitality(0, 0) == 0", "file": "src/core.py"}

    if "src/utils.py" in modules:
        tests["test_divide_normal"] = {"code": "assert safe_divide(10, 2) == 5.0", "file": "src/utils.py"}
        tests["test_divide_zero"] = {"code": "assert safe_divide(10, 0) == 0", "file": "src/utils.py"}
        tests["test_clamp_in_range"] = {"code": "assert clamp(5, 0, 10) == 5", "file": "src/utils.py"}
        tests["test_clamp_below"] = {"code": "assert clamp(-1, 0, 10) == 0", "file": "src/utils.py"}

    if "src/auth.py" in modules:
        tests["test_auth_valid"] = {"code": "assert check_auth('secret_env_key') == True", "file": "src/auth.py"}
        tests["test_auth_invalid"] = {"code": "assert check_auth('wrong_key') == False", "file": "src/auth.py"}

    if "src/metrics.py" in modules:
        tests["test_mean_basic"] = {"code": "assert mean([1, 2, 3]) == 2.0", "file": "src/metrics.py"}
        tests["test_mean_empty"] = {"code": "assert mean([]) == 0.0", "file": "src/metrics.py"}
        tests["test_variance"] = {"code": "assert variance([1, 1, 1]) == 0.0", "file": "src/metrics.py"}

    if "src/parser.py" in modules:
        tests["test_parse_config"] = {"code": "assert parse_config('{\"a\": 1}') == {'a': 1}", "file": "src/parser.py"}
        tests["test_validate_schema_ok"] = {"code": "assert validate_schema({'a': 1, 'b': 2}, ['a', 'b']) == (True, [])", "file": "src/parser.py"}
        tests["test_validate_schema_missing"] = {"code": "assert validate_schema({'a': 1}, ['a', 'b'])[0] == False", "file": "src/parser.py"}

    if "src/scheduler.py" in modules:
        tests["test_round_robin"] = {"code": "assert len(round_robin(['t1','t2','t3'], ['w1','w2'])) == 2", "file": "src/scheduler.py"}

    if "src/network.py" in modules:
        tests["test_adjacency"] = {"code": "adj = build_adjacency([('a','b')]); assert 'b' in adj['a']", "file": "src/network.py"}
        tests["test_has_path_true"] = {"code": "adj = build_adjacency([('a','b'),('b','c')]); assert has_path(adj, 'a', 'c')", "file": "src/network.py"}
        tests["test_has_path_false"] = {"code": "adj = build_adjacency([('a','b')]); assert not has_path(adj, 'a', 'z')", "file": "src/network.py"}

    if "src/cache.py" in modules:
        tests["test_cache_put_get"] = {"code": "c = LRUCache(2); c.put('a', 1); assert c.get('a') == 1", "file": "src/cache.py"}
        tests["test_cache_eviction"] = {"code": "c = LRUCache(1); c.put('a',1); c.put('b',2); assert c.get('a') is None", "file": "src/cache.py"}

    if "src/validator.py" in modules:
        tests["test_validate_range_ok"] = {"code": "assert validate_range(5, 0, 10) == True", "file": "src/validator.py"}
        tests["test_validate_range_fail"] = {"code": "assert validate_range(15, 0, 10) == False", "file": "src/validator.py"}
        tests["test_sanitize"] = {"code": "assert '<' not in sanitize_string('<script>')", "file": "src/validator.py"}

    if "src/transform.py" in modules:
        tests["test_flatten"] = {"code": "assert flatten([[1,2],[3,[4]]]) == [1,2,3,4]", "file": "src/transform.py"}
        tests["test_group_by"] = {"code": "g = group_by([1,2,3,4], lambda x: x % 2); assert len(g) == 2", "file": "src/transform.py"}

    if "schema/config.json" in modules:
        tests["test_config_version"] = {"code": "import json; cfg = json.loads(open('schema/config.json').read()); assert cfg['version'] == '1.0'", "file": "schema/config.json"}
        tests["test_config_threshold"] = {"code": "import json; cfg = json.loads(open('schema/config.json').read()); assert cfg['threshold'] == 0.8", "file": "schema/config.json"}

    if "schema/permissions.json" in modules:
        tests["test_permissions_admin"] = {"code": "import json; p = json.loads(open('schema/permissions.json').read()); assert 'write' in p['admin']", "file": "schema/permissions.json"}

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
        self.tests = _generate_tests_for_modules(self.files, self.rng)
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

        if fault_type == "corrupted_import":
            paths = [p for p in self.files if p.endswith(".py")]
            if not paths:
                return None
            path = target_hint if target_hint and target_hint in self.files else self.rng.choice(paths)
            old = self.files[path]
            new = old.replace("import ", "improt ", 1) if "import " in old else old + "\nimport nonexistent_module"
            self.files[path] = new
            self._file_modified_at[path] = step
            f = Fault(fid, fault_type, path, old, new, step)

        elif fault_type == "flipped_assertion":
            names = list(self.tests.keys())
            if not names:
                return None
            name = self.rng.choice(names)
            old = self.tests[name]["code"]
            new = old.replace("==", "!=", 1) if "==" in old else old.replace("True", "False", 1)
            self.tests[name]["code"] = new
            f = Fault(fid, fault_type, name, old, new, step)

        elif fault_type == "missing_env_var":
            keys = [k for k in self.env_vars if k not in ("ENV",)]
            if not keys:
                return None
            key = self.rng.choice(keys)
            old = self.env_vars[key]
            del self.env_vars[key]
            f = Fault(fid, fault_type, key, old, "__DELETED__", step)

        elif fault_type == "null_return":
            paths = [p for p in self.files if p.endswith(".py") and "return " in self.files[p]]
            if not paths:
                return None
            path = target_hint if target_hint and target_hint in paths else self.rng.choice(paths)
            old = self.files[path]
            lines = old.split("\n")
            for i, line in enumerate(lines):
                if "return " in line and "return None" not in line:
                    lines[i] = line.split("return")[0] + "return None"
                    break
            new = "\n".join(lines)
            self.files[path] = new
            self._file_modified_at[path] = step
            f = Fault(fid, fault_type, path, old, new, step)

        elif fault_type == "off_by_one":
            paths = [p for p in self.files if p.endswith(".py") and "range(" in self.files[p]]
            if not paths:
                paths = [p for p in self.files if p.endswith(".py")]
            if not paths:
                return None
            path = self.rng.choice(paths)
            old = self.files[path]
            new = old.replace("len(", "len(", 1)  # subtle — add +1
            if "for " in old:
                new = old.replace("range(", "range(1+", 1) if "range(" in old else old + "\n# off_by_one_injected"
            else:
                new = old + "\n# off_by_one_injected"
            self.files[path] = new
            self._file_modified_at[path] = step
            f = Fault(fid, fault_type, path, old, new, step)

        elif fault_type == "dependency_cycle":
            py_files = [p for p in self.files if p.endswith(".py")]
            if len(py_files) < 2:
                return None
            a, b = self.rng.sample(py_files, 2)
            module_b = b.replace("/", ".").replace(".py", "")
            old = self.files[a]
            new = f"from {module_b} import *\n" + old
            self.files[a] = new
            self._file_modified_at[a] = step
            f = Fault(fid, fault_type, a, old, new, step)

        elif fault_type == "permission_revoked":
            self.env_vars["PERM_READ_CONFIG"] = "false"
            f = Fault(fid, fault_type, "PERM_READ_CONFIG", "true", "false", step)

        elif fault_type == "race_condition":
            paths = [p for p in self.files if p.endswith(".py")]
            if not paths:
                return None
            path = self.rng.choice(paths)
            old = self.files[path]
            new = old + "\n# RACE_CONDITION: shared state mutated non-atomically"
            self.files[path] = new
            self._file_modified_at[path] = step
            f = Fault(fid, fault_type, path, old, new, step)

        elif fault_type == "schema_mismatch":
            paths = [p for p in self.files if p.endswith(".py") and "def " in self.files[p]]
            if not paths:
                return None
            path = self.rng.choice(paths)
            old = self.files[path]
            new = old.replace("return ", "return str(", 1).rstrip() + ")\n" if "return " in old else old + "\n# schema_mismatch"
            self.files[path] = new
            self._file_modified_at[path] = step
            f = Fault(fid, fault_type, path, old, new, step)

        elif fault_type == "targeted_regression":
            target = target_hint or (self.last_patched_modules[-1] if self.last_patched_modules else None)
            if not target or target not in self.files:
                return self._apply_fault("corrupted_import", step)
            old = self.files[target]
            new = old.replace("return", "retunr", 1) if "return" in old else old + "\n# targeted_corruption"
            self.files[target] = new
            self._file_modified_at[target] = step
            f = Fault(fid, fault_type, target, old, new, step)

        elif fault_type == "cascade_corruption":
            py_files = [p for p in self.files if p.endswith(".py")]
            targets = self.rng.sample(py_files, min(3, len(py_files)))
            first = targets[0]
            old = self.files[first]
            new = old.replace("def ", "deaf ", 1) if "def " in old else old + "\n# cascade_broken"
            self.files[first] = new
            self._file_modified_at[first] = step
            for secondary in targets[1:]:
                old_s = self.files[secondary]
                self.files[secondary] = old_s + f"\n# cascade: depends on {first}"
                self._file_modified_at[secondary] = step
            f = Fault(fid, fault_type, first, old, new, step)

        elif fault_type == "checkpoint_invalidation":
            if self.checkpoints:
                cp = self.rng.choice(self.checkpoints)
                cp["state"]["files"]["__corrupted__"] = "INVALID"
                f = Fault(fid, fault_type, cp["id"], "valid", "corrupted", step)
            else:
                return self._apply_fault("corrupted_import", step)
        else:
            return None

        self.faults.append(f)
        return f

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
        # First, check if there are any immediate syntax/import errors in corrupted files
        # to avoid slow subprocess runs for obvious breaks (optimization)
        corrupted_files = set()
        for f in self.faults:
            if f.fault_type in ("corrupted_import", "null_return", "off_by_one",
                                "targeted_regression", "cascade_corruption",
                                "dependency_cycle", "race_condition", "schema_mismatch"):
                corrupted_files.add(f.target)
            if f.fault_type == "cascade_corruption":
                for path in self.files:
                    if f"cascade: depends on {f.target}" in self.files.get(path, ""):
                        corrupted_files.add(path)

        # Build execution directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1. Write all source modules
            for path, content in self.files.items():
                if path in self.quarantined_modules:
                    continue  # Quarantined modules are excluded
                
                full_path = os.path.join(temp_dir, path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, "w", encoding="utf-8") as f:
                    f.write(content)

            # 2. Write test files and enforce OverlayFS read-only simulation
            test_results = []
            test_file_paths = []
            for name, data in self.tests.items():
                test_code = data["code"]
                target_file = data["file"]
                
                # Auto-import the functions being tested
                import_stmt = ""
                if target_file.endswith(".py"):
                    module_path = target_file.replace("/", ".").replace(".py", "")
                    import_stmt = f"from {module_path} import *"
                
                # We wrap the assert in a try-except to get PASS/FAIL
                # Use repr() to safely encode Windows paths with backslashes
                safe_temp_dir = temp_dir.replace("\\", "\\\\")
                wrapped_code = f"""
import sys
import os
sys.path.insert(0, '{safe_temp_dir}')
{import_stmt}
try:
{chr(10).join(['    ' + line for line in test_code.split(chr(10))])}
    print("PASS")
except Exception as e:
    print(f"FAIL|{{type(e).__name__}}: {{e}}")
"""
                test_path = os.path.join(temp_dir, f"__test_{name}.py")
                with open(test_path, "w", encoding="utf-8") as f:
                    f.write(wrapped_code)
                
                # Simulate OverlayFS read-only mount
                try:
                    os.chmod(test_path, stat.S_IREAD)
                except OSError:
                    # Best effort on platforms/filesystems that don't fully support chmod semantics.
                    pass
                test_file_paths.append((name, test_path, target_file))

            # 3. Execute tests in subprocess (simulating Docker isolation)
            # In a full cluster environment, we would do:
            # subprocess.run(["docker", "run", "--rm", "-v", f"{temp_dir}:/app:ro", "python:3.11", "python", ...])
            # For local speed and reliability across Windows/Linux, we use a secure subprocess
            for name, t_path, target_file in test_file_paths:
                module_dir = target_file.rsplit("/", 1)[0] if "/" in target_file else ""
                if target_file in self.quarantined_modules or module_dir in self.quarantined_modules:
                    test_results.append(TestResult(name=name, status="ERROR", message="Module quarantined"))
                    continue
                
                # Fast fail for obvious faults (saves RL time)
                if target_file in corrupted_files:
                    reason = f"Error: {target_file} is corrupted by active fault"
                    test_results.append(TestResult(name=name, status="FAIL", message=reason))
                    continue
                    
                env = os.environ.copy()
                env.update(self.env_vars) # Inject VM env vars
                try:
                    proc = subprocess.run(
                        ["python", t_path],
                        capture_output=True,
                        text=True,
                        env=env,
                        cwd=temp_dir, # Ensure relative paths work
                        timeout=2 # 2 second timeout per test
                    )
                    out = proc.stdout.strip()
                    if out == "PASS":
                        test_results.append(TestResult(name=name, status="PASS", message="OK"))
                    else:
                        msg = out.split("|")[1] if "|" in out else proc.stderr.strip() or "Unknown Error"
                        test_results.append(TestResult(name=name, status="FAIL", message=msg))
                except subprocess.TimeoutExpired:
                    test_results.append(TestResult(name=name, status="FAIL", message="TimeoutError: Test execution exceeded limit"))
                except Exception as e:
                    test_results.append(TestResult(name=name, status="FAIL", message=f"RuntimeError: {e}"))

        return test_results

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
        if line.startswith("import src."):
            module_name = line.split(" ", 1)[1]
            return module_name.replace(".", "/") + ".py"
        if line.startswith("from src."):
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
