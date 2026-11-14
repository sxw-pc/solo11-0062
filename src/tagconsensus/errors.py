"""Stable, typed errors used across the command line pipeline."""


class TagConsensusError(Exception):
    """Base error carrying a stable machine-readable code and field path."""

    def __init__(self, code: str, message: str, field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "field": self.field}


class ConfigError(TagConsensusError):
    """The YAML plan is invalid."""


class DataError(TagConsensusError):
    """The input file cannot be read or parsed at all."""


class MetricError(TagConsensusError):
    """A metric is not defined for the supplied data shape."""


class AdjudicationError(TagConsensusError):
    """An adjudication file is malformed or inconsistent with the dataset."""


class LedgerError(TagConsensusError):
    """The local SQLite ledger could not be read or updated."""
