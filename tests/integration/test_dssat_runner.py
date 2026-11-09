from pathlib import Path

import pytest

from src.dssat.config import DssatConfig
from src.dssat.data.repository import get_experiment
from src.dssat.runner import run_experiment


@pytest.mark.integration
def test_real_wheat_experiment_runs_when_dssat_is_installed():
    try:
        config = DssatConfig.discover()
    except Exception as exc:
        pytest.skip(f"DSSAT not configured: {exc}")
    try:
        experiment = get_experiment(config.data_dir, "KSAS8101")
    except Exception as exc:
        pytest.skip(f"DSSAT wheat example data unavailable: {exc}")
    result = run_experiment(config, experiment, treatment=1)
    assert float(result["summary"].iloc[0]["HWAM"]) > 0
    assert result["growth"] is not None
    assert (Path(result["run_dir"]) / "metadata.json").is_file()
