#!/usr/bin/env bash
# Build an offline Python 3.10 venv on a host whose only torch install is a
# Python 3.8 conda env (the repository requires Python >= 3.10).
#
# Usage: bash build_py310_env.sh <venv_dir> <wheel_dir> <py38_site_packages>
#   <wheel_dir> holds cp310 wheels for the compiled dependencies (numpy, scipy,
#   pyarrow, pandas, pillow, regex, PyYAML, xxhash, aiohttp, multidict, yarl,
#   frozenlist, propcache, MarkupSafe, charset-normalizer, tokenizers) and
#   torch/torchvision overlays (see below). Pure-Python packages and the nvidia CUDA
#   library wheels are symlinked from the Python 3.8 site-packages; compiled
#   Python 3.8 extensions without a cp310 replacement are left out.
set -euo pipefail

venv="$1"
wheels="$2"
s38="$3"
python3_bin="${PYTHON3:-/usr/bin/python3}"

"$python3_bin" -m venv --without-pip "$venv"
sp="$("$venv/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"

# Overlays (*_overlay.tgz in <wheel_dir>): for torch and torchvision only the
# files whose wheel RECORD hash differs between the cp38 and cp310 builds of the
# same release are shipped; every other file is copied from the cp38 install.
for overlay in "$wheels"/*_overlay.tgz; do
  [[ -f "$overlay" ]] || continue
  for top in $(tar -tzf "$overlay" | cut -d/ -f1 | sort -u); do
    if [[ -d "$s38/$top" && ! -e "$sp/$top" ]]; then
      cp -a "$s38/$top" "$sp/$top"
      find "$sp/$top" -name '*.cpython-38-*.so' -delete
      find "$sp/$top" -name '__pycache__' -type d -prune -exec rm -rf {} +
    fi
  done
  tar -xzf "$overlay" -C "$sp"
done

for wheel in "$wheels"/*.whl; do
  "$venv/bin/python" -m zipfile -e "$wheel" "$sp/"
done
# Wheel .data/{purelib,platlib} trees belong in site-packages.
for data_dir in "$sp"/*.data; do
  [[ -d "$data_dir" ]] || continue
  for sub in purelib platlib; do
    if [[ -d "$data_dir/$sub" ]]; then
      cp -a "$data_dir/$sub/." "$sp/"
    fi
  done
  rm -rf "$data_dir"
done

# Compiled Python 3.8 packages that are neither replaced nor needed.
exclude=" backports cffi _cffi_backend contourpy coverage cupy cupy_backends cupyx fastrlock
 fontTools greenlet grpc h5py h5py.libs httptools jiter kiwisolver matplotlib mpl_toolkits
 msgpack msgspec numba llvmlite nvtx orjson pydantic pydantic_core ray rpds sentencepiece
 sklearn scikit_learn.libs sqlalchemy tiktoken tiktoken_ext uvloop watchfiles websockets zmq
 pyzmq.libs triton xformers vllm torchaudio torio cv2 opencv_python_headless.libs pymupdf fitz
 flash_attention __pycache__ "
exclude="$(echo "$exclude" | tr '\n' ' ') "

normalize() {
  echo "$1" | sed -E 's/(-[0-9][^-]*)?\.(dist|egg)-info$//' | tr 'A-Z-' 'a-z_'
}
installed_projects=" "
for info in "$sp"/*.dist-info; do
  installed_projects+="$(normalize "$(basename "$info")") "
done

for entry in "$s38"/*; do
  name="$(basename "$entry")"
  [[ -e "$sp/$name" ]] && continue
  case "$name" in
    *.pth|*.cpython-38-*.so) continue ;;
  esac
  [[ "$exclude" == *" $name "* ]] && continue
  if [[ "$name" == *.dist-info || "$name" == *.egg-info ]]; then
    project="$(normalize "$name")"
    [[ "$installed_projects" == *" $project "* ]] && continue
    [[ "$exclude" == *" $project "* ]] && continue
  fi
  ln -s "$entry" "$sp/$name"
done

"$venv/bin/python" - <<'EOF'
import sys
import numpy, scipy, pandas, pyarrow, yaml, regex, tokenizers, safetensors
import torch, torchvision, transformers, datasets
print(sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "transformers", transformers.__version__, "datasets", datasets.__version__,
      "numpy", numpy.__version__)
print("cuda available:", torch.cuda.is_available())
EOF
