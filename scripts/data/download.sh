#!/usr/bin/env bash
# Downloads the official benchmark files used here into $GPO_DATA/raw (default ./datasets/raw).
# Advection and the notched Darcy problem are generated: see data/README.md.
set -eu
ROOT=${GPO_DATA:-$(cd "$(dirname "$0")/../.." && pwd)/datasets}
RAW=$ROOT/raw
mkdir -p "$RAW/caltech" "$RAW/fno/burgers" "$RAW/fno/darcy421" "$RAW/hf_mutransfer"

get() { curl -sSL -C - --retry 8 -o "$1" "$2"; }

for f in NavierStokes_inputs NavierStokes_outputs; do
    get "$RAW/caltech/$f.npy" "https://data.caltech.edu/api/records/fp3ds-kej20/files/$f.npy/content"
done
HF=https://huggingface.co/datasets/LDA1020/muTransfer-FNO-data/resolve/main
get "$RAW/fno/burgers/burgers_data_R10.mat" "$HF/burgers/burgers_data_R10.mat"
get "$RAW/hf_mutransfer/piececonst_r421_N1024_smooth1.mat" "$HF/darcy/piececonst_r421_N1024_smooth1.mat"
get "$RAW/fno/darcy421/piececonst_r421_N1024_smooth2.mat" "$HF/darcy/piececonst_r421_N1024_smooth2.mat"
echo done
