#!/usr/bin/env bash
set -euo pipefail
python -m forestlab report
python -m forestlab audit
python -m forestlab witness --group 750 --tree 6
python -m forestlab proximity --input examples/proximity.json
python -m forestlab proximity-controls
