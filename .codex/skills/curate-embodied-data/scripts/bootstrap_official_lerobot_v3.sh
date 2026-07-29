#!/usr/bin/env bash
set -euo pipefail

target=${1:-/tmp/rds-official-lerobot-0.6.0}
python -m venv --system-site-packages "$target"
"$target/bin/pip" install --disable-pip-version-check \
  'lerobot==0.6.0' \
  'draccus==0.10.0' \
  'setuptools==80.9.0' \
  'transformers==5.4.0' \
  'tokenizers==0.22.2' \
  'huggingface-hub==1.25.1' \
  'gymnasium==1.2.2'
"$target/bin/python" - <<'PY'
import importlib.metadata as metadata
from lerobot.datasets.lerobot_dataset import LeRobotDataset
assert LeRobotDataset.__module__ == "lerobot.datasets.lerobot_dataset"
print(f"official lerobot={metadata.version('lerobot')}")
PY
