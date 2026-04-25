#!/bin/bash
cd /lihongliang/fangzl/ckpt-compress/code/checkpoint_compress
mkdir -p logs
LOGFILE="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
nohup bash scripts/overnight_ft_batch.sh > "$LOGFILE" 2>&1 &
PID=$!
echo "Started PID=$PID, log=$LOGFILE"
sleep 2
ps -p $PID -o pid,stat,cmd --no-headers
