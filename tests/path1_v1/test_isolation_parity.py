"""Protect the frozen implementation, package resources and adapter behavior."""

from __future__ import annotations

import ast
import hashlib
import json
from importlib.resources import files
from pathlib import Path

from .parity_probe import collect

PREFIX = "agentmemeval.path1_v1"
FIXTURES = Path(__file__).parent / "fixtures"


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


class OriginalNamespace(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module and node.module.startswith(PREFIX + "."):
            node.module = "agentmemeval" + node.module[len(PREFIX):]
        return node

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.startswith(PREFIX + "."):
                alias.name = "agentmemeval" + alias.name[len(PREFIX):]
        return node


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
    root = files(PREFIX)
    for name, record in expected["modules"].items():
        tree = ast.parse(root.joinpath(name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            imports = ([alias.name for alias in node.names] if isinstance(node, ast.Import)
                       else [node.module] if isinstance(node, ast.ImportFrom) else [])
            assert not any(value and value.startswith("agentmemeval.")
                           and not value.startswith(PREFIX + ".") for value in imports), name
        if record["mode"] == "namespace_imports_only":
            normalized = ast.dump(OriginalNamespace().visit(tree), include_attributes=False)
            assert hashlib.sha256(normalized.encode()).hexdigest() == record["ast_sha256"], name


def test_packaged_assets_match_frozen_source_bytes():
    expected = json.loads((FIXTURES / "source_sha256.json").read_text(encoding="utf-8"))
    root = files(PREFIX)
    for name, identity in expected["assets"].items():
        assert hashlib.sha256(root.joinpath(name).read_bytes()).hexdigest() == identity, name
