#!/usr/bin/env python3
"""Check relocated imports, old checkpoint aliases and resource paths on CPU.

Task discovery is replaced by an empty module in this check. Isaac/PhysX startup
is intentionally outside its scope; training and play keep normal registration.
"""
import argparse
import ast
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'source/rl_training/rl_training'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    assert sorted(p.name for p in SOURCE.glob('*.py')) == ['__init__.py']
    # Exercise the real root initializer and compatibility hook while avoiding
    # the simulator's task initializer. All model/algorithm modules are real.
    sys.path.insert(0, str(SOURCE.parent))
    tasks = ModuleType('rl_training.tasks')
    tasks.__path__ = [str(SOURCE / 'tasks')]
    sys.modules[tasks.__name__] = tasks
    import rl_training
    from rl_training.utils.module_layout import LEGACY_MODULES, REPO_ROOT, source_path
    assert REPO_ROOT == ROOT
    for target in LEGACY_MODULES.values():
        parts = target.split('.')
        for length in range(1, len(parts)):
            name = 'rl_training.' + '.'.join(parts[:length])
            if name in sys.modules:
                continue
            package = ModuleType(name)
            package.__path__ = [str(SOURCE.joinpath(*parts[:length]))]
            sys.modules[name] = package
    checked = []
    for old, target in LEGACY_MODULES.items():
        assert source_path(old + '.py').is_file(), old
        # Environment/config classes require Isaac's runtime. Their paths and
        # relative imports are checked below, without pretending to launch it.
        if old == 'ascent_entry_env' or old.endswith('_profile') and old != 'v1_stop_profile':
            continue
        canonical = importlib.import_module('rl_training.' + target)
        legacy = importlib.import_module('rl_training.' + old)
        assert legacy is canonical, old
        assert importlib.import_module('rl_training.' + target) is canonical
        checked.append(old)
    from rl_training import ascent_sequence_actor
    assert ascent_sequence_actor is importlib.import_module('rl_training.models.ascent_sequence_actor')
    from rl_training.utils.module_loader import load_module
    assert load_module('ascent_adaptive_actor') is load_module('models.ascent_adaptive_actor')
    assert 'isaaclab' not in sys.modules and 'isaacsim' not in sys.modules
    paths_checked = 0
    for path in SOURCE.rglob('*.py'):
        relative = path.relative_to(SOURCE)
        package = 'rl_training.' + '.'.join(relative.parent.parts)
        package = package.rstrip('.')
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom):
                continue
            target = node.module or ''
            if node.level:
                target = importlib.util.resolve_name('.' * node.level + target, package)
            if target == 'rl_training' or target.startswith('rl_training.'):
                parts = target.split('.')[1:]
                base = SOURCE.joinpath(*parts)
                assert base.with_suffix('.py').is_file() or (base / '__init__.py').is_file(), (path, target)
                paths_checked += 1
    for name in ('ascent_reference.json', 'v1_stop_baseline.json'):
        assert isinstance(json.loads(source_path(name).read_text()), dict)
    from rl_training.tasks.manager_based.locomotion.velocity.config.wheeled.deeprobotics_m20.profiles import v1_stop_profile
    assert isinstance(v1_stop_profile.baseline(), dict)
    result = dict(root_python_files=['__init__.py'], module_destinations=len(LEGACY_MODULES),
                  legacy_alias_identity_checked=len(checked), internal_import_paths_checked=paths_checked,
                  resources_readable=True, cpu_loader_identity=True,
                  isaac_started=False, training_started=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
