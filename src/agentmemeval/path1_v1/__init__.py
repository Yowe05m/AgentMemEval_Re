"""Isolated Path1 v1 library; importing it does not authorize an experiment."""

import sys

if not (3, 11) <= sys.version_info[:2] < (3, 13):
    raise RuntimeError("Path1 v1 requires its frozen Python runtime >=3.11,<3.13")
