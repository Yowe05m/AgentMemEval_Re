from __future__ import annotations

import hashlib
import random


def derive_seed(base_seed: int, *parts: object) -> int:

    text = "|".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def make_rng(base_seed: int, *parts: object) -> random.Random:

    return random.Random(derive_seed(base_seed, *parts))


def seed_global(base_seed: int) -> None:

    random.seed(base_seed)
