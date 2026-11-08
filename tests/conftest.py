"""Shared helpers: the stage scripts start with digits, so they are loaded by path."""
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "src" / "pipeline"
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT / "src"))


@lru_cache(maxsize=None)
def load_stage(name: str):
    spec = importlib.util.spec_from_file_location(f"stage_{name}", PIPELINE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def stage05():
    return load_stage("05_ranking_elicit")


@pytest.fixture(scope="session")
def stage08():
    return load_stage("08_substitution_shares")


@pytest.fixture(scope="session")
def stage10():
    return load_stage("10_assortment_experiment")
