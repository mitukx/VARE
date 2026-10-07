"""Lazy public API: standalone evaluation commands also run on Python 3.9+."""
from importlib import import_module

__version__ = '0.4.0'
_EXPORTS = {'EngineConfig': 'config', 'LagConfig': 'config', 'PromotionConfig': 'config',
            'ReplayConfig': 'config', 'CapabilityLoop': 'engine', 'LoopHooks': 'engine',
            'Attempt': 'types', 'EvaluationReport': 'types', 'Task': 'types', 'Verification': 'types'}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    return getattr(import_module('.' + _EXPORTS[name], __package__), name)
