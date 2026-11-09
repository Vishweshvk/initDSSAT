from pathlib import Path

from src.dssat.config import DssatConfig, _load_dotenv


def test_dssat_home_detection_uses_environment(tmp_path: Path, monkeypatch):
    executable = tmp_path / "DSCSM048.EXE"
    executable.write_bytes(b"fixture")
    monkeypatch.setenv("DSSAT_HOME", str(tmp_path))
    monkeypatch.delenv("DSSAT_EXECUTABLE", raising=False)
    monkeypatch.setenv("DSSAT_DATA", str(tmp_path))
    config = DssatConfig.discover()
    assert config.home == tmp_path.resolve()
    assert config.executable == executable.resolve()


def test_relative_project_paths_are_rooted_at_project_not_shell_cwd(tmp_path: Path, monkeypatch):
    dssat = tmp_path / "DSSAT"
    dssat.mkdir()
    (dssat / "DSCSM048.EXE").write_bytes(b"fixture")
    monkeypatch.setenv("DSSAT_HOME", str(dssat))
    monkeypatch.setenv("DSSAT_DATA", str(dssat))
    monkeypatch.setenv("PROJECT_DATA_DIR", "relative-data")
    monkeypatch.setenv("MODEL_DIR", "relative-models")
    monkeypatch.chdir(tmp_path)

    config = DssatConfig.discover()
    project_root = Path(__file__).resolve().parents[2]
    assert config.project_data_dir == (project_root / "relative-data").resolve()
    assert config.model_dir == (project_root / "relative-models").resolve()


def test_dotenv_loads_only_missing_environment_values(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PROJECT_DATA_DIR=from-file\nMODEL_DIR='quoted-models'\n", encoding="utf-8")
    monkeypatch.delenv("PROJECT_DATA_DIR", raising=False)
    monkeypatch.setenv("MODEL_DIR", "from-process")
    _load_dotenv(tmp_path)
    assert __import__("os").environ["PROJECT_DATA_DIR"] == "from-file"
    assert __import__("os").environ["MODEL_DIR"] == "from-process"


def test_blank_optional_paths_use_project_defaults(tmp_path: Path, monkeypatch):
    executable = tmp_path / "DSCSM048.EXE"
    executable.write_bytes(b"fixture")
    monkeypatch.setenv("DSSAT_HOME", str(tmp_path))
    monkeypatch.setenv("DSSAT_DATA", "")
    monkeypatch.setenv("PROJECT_DATA_DIR", "")
    monkeypatch.setenv("MODEL_DIR", "")
    config = DssatConfig.discover()
    project_root = Path(__file__).resolve().parents[2]
    assert config.data_dir == tmp_path.resolve()
    assert config.project_data_dir == (project_root / "data").resolve()
    assert config.model_dir == (project_root / "models").resolve()
