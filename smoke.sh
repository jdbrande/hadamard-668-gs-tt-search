#!/usr/bin/env bash
# 3-worker orchestration smoke test on toy orders (~1 minute).
set -eu
rm -rf smoke hadamard_44.csv hadamard_52.csv
python3 ctl.py launch --workers 3 --tt-share 0.34 --gs-n 13 --tt-m 4 \
        --pool-cap 200 --workdir smoke --wait
ORDER=$(python3 -c "import json;print(json.load(open('smoke/SOLUTION.json'))['order'])")
python3 ctl.py verify "hadamard_${ORDER}.csv"
python3 ctl.py status --workdir smoke | tail -2
echo "smoke test passed (order ${ORDER})"
