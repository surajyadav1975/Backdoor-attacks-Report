#!/usr/bin/env bash
# Run the full PSBD pipeline against all three attacks (Linux / macOS).
# Use from the project root:
#   bash scripts/run_all_attacks.sh

set -euo pipefail

echo "==> BadNets"
python main.py --attack badnets

echo "==> Blended"
python main.py --attack blended

echo "==> WaNets"
python main.py --attack wanets

echo "All experiments finished. See ./results for JSON reports."
