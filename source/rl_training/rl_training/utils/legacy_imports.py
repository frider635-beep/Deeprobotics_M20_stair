"""Lazy aliases for pre-reorganization checkpoint and configuration imports."""
import importlib
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
import sys

from .module_layout import LEGACY_MODULES


class _AliasLoader(Loader):
    def __init__(self, target):
        self.target = target

    def create_module(self, spec):
        return importlib.import_module(self.target)

    def exec_module(self, module):
        pass


class _LegacyFinder(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        prefix = 'rl_training.'
        if fullname.startswith(prefix):
            name = fullname[len(prefix):]
            if name in LEGACY_MODULES:
                return spec_from_loader(fullname, _AliasLoader(prefix + LEGACY_MODULES[name]))
        return None


def install():
    """Register aliases before IsaacLab discovers task configurations."""
    if not any(isinstance(finder, _LegacyFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _LegacyFinder())
