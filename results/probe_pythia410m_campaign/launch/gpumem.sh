#!/usr/bin/env bash
# Peak per-PID memory on GPU 7 from the campaign samplers, excluding the known 32B job PID.
cd /home/gridtest/dacp_probe/campaign
for f in */gpu7_mem.log; do
  awk -v f="$f" -F', ' '{split($1,a," "); pid=$2; mem=$3+0; if (pid!="2999137" && mem>max[pid]) max[pid]=mem} END {for (p in max) print f, "pid", p, "peak_MiB", max[p]}' "$f"
done
