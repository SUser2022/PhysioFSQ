"""Public PhysioFSQ model API, loaded lazily to keep imports focused."""

from __future__ import annotations
from importlib import import_module

_EXPORTS = {
    "FrequencyFSQTokenizer": ".frequency_fsq",
    "FrequencyFSQStudent": ".frequency_student",
    "FrequencyStudentBackbone": ".frequency_student",
    "FrequencyDownstreamModel": ".frequency_downstream",
    "load_frequency_backbone": ".frequency_student",
}
__all__ = list(_EXPORTS)

def __getattr__(name: str):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
