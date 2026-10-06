"""The result tables in both READMEs are transcribed from committed JSON; this keeps them in step."""
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
READMES = ["README.md", "README.zh-CN.md"]
TACIT = ROOT / "bench/results_tacit/2026-10-05/one_forward.json"
L0 = ROOT / "bench/results_v01/2026-09-22/Qwen__Qwen3-8B.json"


def cells(line):
    return [c.strip().strip("*") for c in line.strip().strip("|").split("|")]


@pytest.mark.parametrize("readme", READMES)
def test_tacit_table_matches_its_json(readme):
    models = json.loads(TACIT.read_text())["models"]
    rows = [cells(x) for x in (ROOT / readme).read_text().splitlines() if x.startswith("| [Tacit-")]
    assert len(rows) == len(models)
    for row in rows:
        name = re.match(r"\[(Tacit-[\w.]+)\]", row[0]).group(1)
        m = models["morriszjm/" + name]
        assert row[1] == m["base_model"].split("/")[-1]
        assert row[2] == "%.3f" % m["jevbench_public"]["accuracy"]
        assert row[3] == "%.3f" % m["bev_decision_test"]["accuracy"]


@pytest.mark.parametrize("readme", READMES)
def test_raw_vs_l0_table_matches_its_json(readme):
    task = next(t for t in json.loads(L0.read_text())["tasks"] if t["task"] == "banking20")["levels"]
    text = (ROOT / readme).read_text()
    for metric, fmt in [("flip", "%.3f"), ("acc", "%.3f"), ("ece", "%.3f")]:
        want = "| %s | %s |" % (fmt % task["raw"][metric], fmt % task["L0"][metric])
        assert want in re.sub(r"\*\*", "", text), (readme, metric, want)
    cov = "| %.1f%% | %.1f%% |" % (100 * task["raw"]["cov@5%"], 100 * task["L0"]["cov@5%"])
    assert cov in re.sub(r"\*\*", "", text), (readme, cov)
