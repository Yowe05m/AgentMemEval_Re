"""The optional frozen Path1 runtime must not raise main's Python floor."""

import sys


def pytest_ignore_collect(collection_path, config):
    return not (3, 11) <= sys.version_info[:2] < (3, 13)
