#!/usr/bin/env bash
# Usage: pack.sh <suite> [...] -- tar.gz of campaign suite dirs (no checkpoints) to /tmp/dacp_pack.tgz
cd /home/gridtest/dacp_probe/campaign
tar czf /tmp/dacp_pack.tgz --exclude='*.pt' --exclude='*.bin' --exclude='*.safetensors' "$@" queue.log queue_*.txt
ls -la /tmp/dacp_pack.tgz
du -sh "$@"
