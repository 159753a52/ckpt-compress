#!/usr/bin/env bash
# Usage: deploy.sh <base_dir_name> <new_sha> <bundle_path>
set -euo pipefail
base=/home/gridtest/dacp_probe_code/$1
sha=$2
short=${sha:0:7}
new=/home/gridtest/dacp_probe_code/$short
if [[ -e "$new" ]]; then echo "exists $new"; exit 1; fi
cp -a "$base" "$new"
cd "$new"
git bundle verify "$3" >/dev/null
git fetch -q "$3" HEAD
git reset -q --hard "$sha"
echo "$sha" > DACP_CODE_COMMIT
git log --oneline -3
echo "status:"; git status --short | head
