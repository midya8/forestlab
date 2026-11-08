#!/usr/bin/env bash
set -euo pipefail
installed=$1
source=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
import importlib,pkgutil,forestlab
mods=list(pkgutil.walk_packages(forestlab.__path__,forestlab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==7,[m.name for m in mods]
print('all eight modules including root imported outside checkout')
PY
python -m forestlab report > report.json
python -m forestlab audit > audit.json
python -m forestlab witness --group 750 --tree 6 > witness.json
printf '%s\n' '{"features":[[0,0,0,0],[1,2,3,4]]}' > input.json
python -m forestlab predict --input input.json > predictions.json
python -m forestlab fit --output fresh.json > summary.json
python -m forestlab audit --report fresh.json > fresh-audit.json
python -m forestlab export --output web > export.json
python - <<'PY'
import json
from forestlab.experiment import saved,equivalent
assert equivalent(json.load(open('fresh.json')),saved())
assert len(json.load(open('predictions.json'))['forest']['votes'])==2
assert len(json.load(open('fresh-audit.json'))['forests'])==8
assert json.load(open('export.json'))['group_evaluations']==10784
print('all six commands, nineteen fresh fits and complete audits pass outside checkout')
PY
cmp web/data.js "$source/docs/data.js"
python - "$source/docs/reference.json" <<'PY'
import json,sys
from forestlab.experiment import equivalent
actual=json.load(open('web/reference.json'));expected=json.load(open(sys.argv[1]))
assert actual.keys()==expected.keys()
assert actual['predictions']==expected['predictions']
assert actual['paths']==expected['paths']
assert equivalent(actual['controls'],expected['controls'])
PY
for n in 0 1 2 3; do cmp "web/archive-$n.js" "$source/docs/archive-$n.js"; done
printf '%s\n' 'archive exports, discrete predictions and paths exact; finite control-score references agree outside checkout'

python -m forestlab permutation --fresh --output permutation.json > permutation-summary.json
python -m forestlab permutation --input permutation.json --audit > permutation-audit.json
python -m forestlab permutation --export perm-web > permutation-export.json
python - <<'PY2'
import json
from forestlab.permutation import saved
assert json.load(open('permutation.json'))==saved()
assert json.load(open('permutation-audit.json'))['model_row_evaluations']==129600
PY2
cmp perm-web/permutation-data.js "$source/docs/permutation-data.js"
cmp perm-web/permutation-archive-0.js "$source/docs/permutation-archive-0.js"
printf '%s\n' 'seventh command: all240 post-fit cases/129600 model-row evaluations, fresh and archived audits and two exact exports PASS'
