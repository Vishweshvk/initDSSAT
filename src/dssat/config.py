"""Resolve DSSAT installation and project data directories."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil

from .exceptions import DssatConfigurationError


def _load_dotenv(project_root: Path) -> None:
    """Load simple KEY=VALUE entries without overriding the process environment."""
    env_file = project_root / ".env"
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key, value = key.strip(), value.strip()
        if not key or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ[key] = value


def _resolve_path(value: str | Path, project_root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def _env_or_default(name: str, default: str | Path) -> str | Path:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def project_directories() -> tuple[Path, Path, Path]:
    """Return project root, data directory, and model directory without requiring DSSAT."""
    project_root = Path(__file__).resolve().parents[2]
    _load_dotenv(project_root)
    project_data = _resolve_path(_env_or_default("PROJECT_DATA_DIR", "data"), project_root)
    model_dir = _resolve_path(_env_or_default("MODEL_DIR", "models"), project_root)
    return project_root, project_data, model_dir


@dataclass(frozen=True)
class DssatConfig:
    home: Path
    executable: Path
    data_dir: Path
    project_data_dir: Path
    model_dir: Path

    @classmethod
    def discover(cls) -> "DssatConfig":
        project_root, project_data, model_dir = project_directories()
        explicit_executable = _env_or_default("DSSAT_EXECUTABLE", "")
        home_env = _env_or_default("DSSAT_HOME", "")
        candidates = [_resolve_path(home_env, project_root)] if home_env else []
        # Prefer the project-bundled DSSAT installation when it is present.
        # This makes a copied DSSAT48 folder work without machine-specific
        # environment variables, while explicit settings still take priority.
        candidates.append(project_root / "DSSAT48")
        candidates.extend(Path(p) for p in (r"C:\DSSAT48", r"C:\Program Files\DSSAT48"))
        home = next((p for p in candidates if (p / "DSCSM048.EXE").is_file()), None)
        executable = _resolve_path(explicit_executable, project_root) if explicit_executable else (home / "DSCSM048.EXE" if home else None)
        if executable is None:
            found = shutil.which("DSCSM048.EXE")
            executable = Path(found) if found else None
        if executable is not None and home is None:
            home = executable.parent
        if home is None or executable is None or not executable.is_file():
            expected = "Set DSSAT_HOME to the DSSAT v4.8.6 installation directory or DSSAT_EXECUTABLE to DSCSM048.EXE."
            raise DssatConfigurationError(expected)
        data_dir = _resolve_path(_env_or_default("DSSAT_DATA", home), project_root)
        return cls(home.resolve(), executable.resolve(), data_dir.resolve(), project_data.resolve(), model_dir.resolve())

    @property
    def wheat_dir(self) -> Path:
        return self.data_dir / "Wheat"

    @property
    def weather_dir(self) -> Path:
        return self.data_dir / "Weather"

    @property
    def soil_dir(self) -> Path:
        return self.data_dir / "Soil"

    @property
    def genotype_dir(self) -> Path:
        return self.data_dir / "Genotype"

    def version(self) -> str:
        log_file = self.home / "dssat48.LOG"
        if log_file.is_file():
            for line in log_file.read_text(errors="replace").splitlines():
                if "Initializing DSSAT Version" in line:
                    return line.split("Initializing DSSAT Version", 1)[1].strip()
        return "unknown (check DSSAT output header)"
