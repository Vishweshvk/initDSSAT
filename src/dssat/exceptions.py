"""Domain exceptions for DSSAT setup and execution."""


class DssatError(RuntimeError):
    """Base class for actionable DSSAT errors."""


class DssatConfigurationError(DssatError):
    """Raised when DSSAT installation or data paths are unavailable."""


class ExperimentValidationError(DssatError):
    """Raised when a DSSAT experiment has missing or invalid linked inputs."""


class DssatRunError(DssatError):
    """Raised when DSSAT fails or does not produce parseable output."""
