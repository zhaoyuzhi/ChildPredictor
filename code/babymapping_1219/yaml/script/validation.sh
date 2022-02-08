#!/bin/bash
date=`date +%y%m%d`
script_name=`basename "$0"`
script_name="${script_name%.*}"
IP=$(hostname -I)

echo "[DATE] $date"
echo "[ IP ] $IP"
echo "[YAML] $script_name"

python -u main.py \
--config ./yaml/yaml/${script_name}.yaml \
--mode validation
