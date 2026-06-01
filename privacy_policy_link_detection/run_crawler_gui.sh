#!/bin/bash

TRANCO_FILE="$1"

if [ -z "$TRANCO_FILE" ]; then
    echo "No Tranco file selected."
    exit 1
fi

echo "Using Tranco list: $TRANCO_FILE"

set -e

echo "=== System Info ==="
lscpu
free -h

echo "=== Clean possible inherited venv variables ==="
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH

echo "=== Activate conda environment: openwpm ==="
source ~/anaconda3/etc/profile.d/conda.sh
conda activate openwpm

echo "Python used after conda activate:"
which python
python --version

cd ~/OpenWPM

echo "=== Start Xvfb ==="

pkill -f "Xvfb :99" || true

Xvfb :99 -screen 0 1980x1020x24 &
XVFB_PID=$!

cleanup() {
    echo "=== Cleanup Xvfb ==="
    kill "$XVFB_PID" 2>/dev/null || true
}

trap cleanup EXIT

export DISPLAY=:99

echo "=== Start crawler ==="

/home/openwpm/anaconda3/envs/openwpm/bin/python demo_privacy_policy_download.py --tranco-file "$TRANCO_FILE"

echo "=== Crawler finished successfully ==="
