"""Run the full suite and CLI replay from a wheel outside the checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['forestlab', 'forestlab.__main__', 'forestlab.controls',
           'forestlab.core', 'forestlab.dataset', 'forestlab.experiment',
           'forestlab.export', 'forestlab.permutation', 'forestlab.proximity',
           'forestlab.proximity_export']
COMMANDS = [
    ['report'], ['audit'], ['witness', '--group', '750', '--tree', '6'],
    ['predict', '--input', 'samples.json'], ['fit', '--output', 'fresh.json'],
    ['audit', '--report', 'fresh.json'], ['export', '--output', 'web'],
    ['permutation'], ['permutation', '--fresh', '--output', 'permutation.json'],
    ['permutation', '--input', 'permutation.json', '--audit'],
    ['permutation', '--export', 'permutation-web'],
    ['proximity', '--input', 'examples/proximity.json'], ['proximity-controls'],
    ['proximity-export', '--output', 'proximity-web'],
]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    # The tests read these archive paths. No library Python files are copied.
    shutil.copytree(ROOT / 'forestlab/artifacts', outside / 'forestlab/artifacts')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    shutil.copytree(ROOT / 'examples', reference / 'examples')
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    samples = json.dumps({'features': [[0, 0, 0, 0], [1, 2, 3, 4]]})
    for folder in [reference, outside]:
        (folder / 'samples.json').write_text(samples)
    reports = 0
    for args in COMMANDS:
        expected = json.loads(call([sys.executable, '-m', 'forestlab'] + args,
                                   reference, source_env))
        actual = json.loads(call([py, '-I', '-m', 'forestlab'] + args, outside))
        assert actual == expected, args
        reports += 1
    for name in ['fresh.json', 'permutation.json']:
        assert json.loads((outside / name).read_text()) == json.loads((reference / name).read_text()), name
    exports = 0
    for name in ['web', 'permutation-web', 'proximity-web']:
        actual = outside / name
        expected = reference / name
        assert {p.name for p in actual.iterdir()} == {p.name for p in expected.iterdir()}
        for path in actual.iterdir():
            assert path.read_bytes() == (expected / path.name).read_bytes(), path
            exports += 1
    print(json.dumps(dict(complete_reports=reports, command_kinds=10,
                          complete_output_files=2, fresh_models=19,
                          complete_export_files=exports, entrypoint='python -m')))
