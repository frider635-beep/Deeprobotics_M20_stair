"""Load the training implementation for CPU/MuJoCo tools without Isaac/Kit.

This module is loaded by file path. Its isolated namespace preserves canonical
relative imports while deliberately skipping task registration initializers.
Normal IsaacLab entry points continue to import ``rl_training`` normally.
"""
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_NAMESPACE = '_m20_training_cpu'


def _namespace():
    if _NAMESPACE in sys.modules:
        return
    paths = {_PACKAGE_ROOT}
    # All package ancestors needed by the relocated pure modules are installed
    # before imports; importing a parent must never start Isaac task discovery.
    for relative in ('models', 'algorithms/initialization', 'utils',
                     'tasks/manager_based/locomotion/velocity/mdp/ascent',
                     'tasks/manager_based/locomotion/velocity/mdp/stop',
                     'tasks/manager_based/locomotion/velocity/config/wheeled/deeprobotics_m20/profiles'):
        parent = _PACKAGE_ROOT / relative
        while parent != _PACKAGE_ROOT:
            paths.add(parent)
            parent = parent.parent
    for path in sorted(paths, key=lambda p: len(p.parts)):
        suffix = '.'.join(path.relative_to(_PACKAGE_ROOT).parts)
        name = _NAMESPACE + ('.' + suffix if suffix else '')
        package = ModuleType(name)
        package.__path__ = [str(path)]
        package.__package__ = name
        package.__spec__ = importlib.util.spec_from_loader(name, loader=None, is_package=True)
        sys.modules[name] = package
        if suffix:
            parent_name, child = name.rsplit('.', 1)
            setattr(sys.modules[parent_name], child, package)


def load_module(name):
    """Load one historical short name or a canonical relative module name."""
    _namespace()
    layout = importlib.import_module(_NAMESPACE + '.utils.module_layout')
    target = layout.LEGACY_MODULES.get(name, name)
    return importlib.import_module(_NAMESPACE + '.' + target)


def source_path(name):
    _namespace()
    return importlib.import_module(_NAMESPACE + '.utils.module_layout').source_path(name)
