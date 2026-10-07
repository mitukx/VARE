"""Checkout bootstrap; all implementation lives in src/vare."""
from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / 'src' / 'vare')]
from ._public import __all__, __getattr__, __version__
