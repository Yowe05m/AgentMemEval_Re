"""Protect the frozen implementation, package resources and adapter behavior."""

from __future__ import annotations

import ast
import hashlib
import json
from importlib.resources import files
from pathlib import Path, PurePosixPath, PureWindowsPath

from .parity_probe import collect, storage_conflict_message, storage_relative_path

PREFIX = "agentmemeval.path1_v1"
FIXTURES = Path(__file__).parent / "fixtures"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class OriginalNamespace(ast.NodeTransformer):
    def __init__(self, routing_literal_inverse=None):
        self.routing_literal_inverse = routing_literal_inverse or {}

    def visit_Constant(self, node):
        if type(node.value) is str and node.value in self.routing_literal_inverse:
            node.value = self.routing_literal_inverse[node.value]
        return node

    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith(PREFIX + "."):
            node.module = "agentmemeval" + node.module[len(PREFIX):]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith(PREFIX + "."):
                alias.name = "agentmemeval" + alias.name[len(PREFIX):]
        return node


def stable_ast_dump(tree):
    # Python 3.12 adds empty type_params fields to otherwise unchanged 3.11 ASTs.
    # Preserve any nonempty generic parameters; only omit the version-only empty field.
    for node in ast.walk(tree):
        if "type_params" in node._fields and getattr(node, "type_params", None) == []:
            node._fields = tuple(field for field in node._fields if field != "type_params")
    return ast.dump(tree, include_attributes=False)


def test_adapters_and_scientific_outputs_match_immutable_source(tmp_path_factory):
    expected = json.loads((FIXTURES / "parity_sha256.json").read_text(encoding="utf-8"))
    # Retain the long-component fixture without adding pytest's long test name
    # to its root. The exclusive directory remains under the external basetemp.
    root = tmp_path_factory.mktemp("p")
    observed = collect(PREFIX, root, Path(__file__).parent
                       / "unit/test_path1_terminal_policy.py")
    assert {key: digest(value) for key, value in observed.items()} == expected["sections"]


def test_mechanical_modules_preserve_complete_ast_and_import_only_their_version():
    expected = json.loads((FIXTURES / "source_sha256.json").read_text(encoding="utf-8"))
    evolution = json.loads((FIXTURES / "operations_evolution.json").read_text(encoding="utf-8"))
    assert evolution["baseline_integration_commit"] == "b9885e4d75ef32a21360323784004de9919181ea"
    assert evolution["original_source_commit"] == expected["source_commit"]
    assert set(evolution["modules"]) == {
        "operations/worker.py", "operations/intent_executor.py",
        "operations/submission_process.py", "storage/snapshot_archive.py",
        "operations/contracts.py", "operations/slurm.py",
    }
    root = files(PREFIX)
    for name, record in expected["modules"].items():
        tree = ast.parse(root.joinpath(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            assert not any(value and value.startswith("agentmemeval.")
                           and not value.startswith(PREFIX + ".") for value in imports), name
        if record["mode"] in {
            "namespace_imports_only", "namespace_imports_and_reviewed_routing_literals",
        }:
            normalized = stable_ast_dump(OriginalNamespace(
                expected.get("routing_literal_inverse")).visit(tree))
            if name in evolution["modules"]:
                change = evolution["modules"][name]
                assert change["baseline_normalized_ast_sha256"] == record["ast_sha256"], name
                assert change["reason"] and change["regression_tests"], name
                assert all((Path(__file__).parents[2] / test).is_file()
                           for test in change["regression_tests"]), name
                if name == "operations/contracts.py":
                    # The execution identity builder may reuse a verified
                    # in-process model capability. Every other contract rule,
                    # hashing algorithm and scientific boundary stays frozen.
                    assert change["allowed_top_level_changes"] == ["build_runtime_identity"]
                    preserved = ast.parse(root.joinpath(name).read_text(encoding="utf-8"))
                    preserved.body = [node for node in preserved.body
                                      if getattr(node, "name", None) != "build_runtime_identity"]
                    unchanged = stable_ast_dump(OriginalNamespace(
                        expected.get("routing_literal_inverse")).visit(preserved))
                    assert hashlib.sha256(unchanged.encode()).hexdigest() == (
                        change["baseline_unchanged_top_level_ast_sha256"]
                    )
                if name == "operations/slurm.py":
                    # Only CPU/MiB/policy fields and their exact rendering/
                    # validation may evolve. Preserve every other scheduler,
                    # identity, capacity and transport rule from the baseline.
                    scopes = {
                        "SubmitSpec": {
                            "new_fields": ["cpus_per_task", "memory_mib", "memory_policy"],
                            "methods": ["__post_init__", "payload", "from_payload"],
                        },
                        "SubmissionRequest": {"new_fields": [], "methods": ["sbatch_argv"]},
                    }
                    assert change["allowed_class_changes"] == scopes
                    preserved = ast.parse(root.joinpath(name).read_text(encoding="utf-8"))
                    for node in preserved.body:
                        if isinstance(node, ast.ClassDef) and node.name in scopes:
                            scope = scopes[node.name]
                            node.body = [
                                child for child in node.body
                                if not (
                                    isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                                    and child.name in scope["methods"]
                                ) and not (
                                    isinstance(child, ast.AnnAssign)
                                    and isinstance(child.target, ast.Name)
                                    and child.target.id in scope["new_fields"]
                                )
                            ]
                    unchanged = stable_ast_dump(OriginalNamespace(
                        expected.get("routing_literal_inverse")).visit(preserved))
                    assert hashlib.sha256(unchanged.encode()).hexdigest() == (
                        change["baseline_unchanged_ast_sha256"]
                    )
                expected_ast = change["normalized_ast_sha256"]
            else:
                expected_ast = record["ast_sha256"]
            assert hashlib.sha256(normalized.encode()).hexdigest() == expected_ast, name


def test_new_and_inherited_modules_keep_namespace_import_closure():
    def walk(root):
        for child in root.iterdir():
            if child.is_dir() and child.name != "__pycache__":
                yield from walk(child)
            elif child.name.endswith(".py"):
                yield child

    for path in walk(files(PREFIX)):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            assert not any(value and value.startswith("agentmemeval.")
                           and not value.startswith(PREFIX + ".") for value in imports), str(path)


def test_packaged_assets_match_frozen_source_bytes():
    expected = json.loads((FIXTURES / "source_sha256.json").read_text(encoding="utf-8"))
    root = files(PREFIX)
    for name, identity in expected["assets"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == identity, name


def test_storage_path_fields_have_one_representation_on_both_hosts():
    results = []
    for root in (PurePosixPath('/fixture'), PureWindowsPath('A:/fixture')):
        run = root / 'files' / 'run'
        results.append({
            'directory': storage_relative_path(run / 'hero', root),
            'message': storage_conflict_message('conflict / detail: ' + str(run), run, root),
        })
    assert results[0] == results[1] == {
        'directory': 'files\\run\\hero',
        'message': 'conflict / detail: <fixture-root>\\files\\run',
    }


def test_storage_path_normalization_keeps_non_path_and_payload_differences():
    root = PurePosixPath('/fixture')
    run = root / 'files' / 'run'
    record = {
        'directory': storage_relative_path(run / 'hero', root),
        'message': storage_conflict_message('conflict / detail: ' + str(run), run, root),
        'files': {'raw.txt': 'literal/with/slash'},
    }
    variants = [
        {**record, 'directory': storage_relative_path(run / 'other', root)},
        {**record, 'message': storage_conflict_message('other / detail: ' + str(run), run, root)},
        {**record, 'files': {'raw.txt': 'literal\\with\\slash'}},
    ]
    assert all(digest(value) != digest(record) for value in variants)
