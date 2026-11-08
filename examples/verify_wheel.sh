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
assert len(mods)==6,[m.name for m in mods]
print('all seven modules including root imported outside checkout')
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
cmp web/reference.json "$source/docs/reference.json"
for n in 0 1 2 3; do cmp "web/archive-$n.js" "$source/docs/archive-$n.js"; done
printf '%s\n' 'both exports reproduce byte-exact outside checkout'
