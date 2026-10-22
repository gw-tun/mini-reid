#!/usr/bin/env bash
# Download the official model assets (macOS / Linux counterpart of download_models.ps1).
#
#   scripts/download_models.sh                 # everything missing
#   scripts/download_models.sh --only yolo     # yolo | person | vehicle
#   scripts/download_models.sh --verify        # check files + print SHA-256, no download
#   scripts/download_models.sh --force         # replace existing files
#
# Resumes interrupted downloads (<file>.download). Also fetches the two FastReID
# base configs the class configs inherit from:
#   configs/msmt17_sbs_R50-IBN.yml -> ../Base-Strongerbaseline.yml -> Base-bagtricks.yml
# Without them the configs don't load (docs/VIP-9-PERSON-PLAN.md §5, issue 2).
set -euo pipefail
cd "$(dirname "$0")/.."

FASTREID=https://github.com/JDAI-CV/fast-reid
CFG=https://raw.githubusercontent.com/JDAI-CV/fast-reid/v0.1.1/configs

# group | url | path
ASSETS=(
  "yolo|https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m.pt|models/yolo/yolo11m.pt"
  "person|$FASTREID/releases/download/v0.1.1/msmt_sbs_R50-ibn.pth|models/fastreid/msmt17_sbs_R50-IBN.pth"
  "person|$CFG/MSMT17/sbs_R50-ibn.yml|models/fastreid/configs/msmt17_sbs_R50-IBN.yml"
  "vehicle|$FASTREID/releases/download/v0.1.1/veri_sbs_R50-ibn.pth|models/fastreid/veri_sbs_R50-IBN.pth"
  "vehicle|$CFG/VeRi/sbs_R50-ibn.yml|models/fastreid/configs/veri_sbs_R50-IBN.yml"
  "person|$CFG/Base-Strongerbaseline.yml|models/fastreid/Base-Strongerbaseline.yml"
  "person|$CFG/Base-bagtricks.yml|models/fastreid/Base-bagtricks.yml"
  "vehicle|$CFG/Base-Strongerbaseline.yml|models/fastreid/Base-Strongerbaseline.yml"
  "vehicle|$CFG/Base-bagtricks.yml|models/fastreid/Base-bagtricks.yml"
)

only="" verify=0 force=0
while [[ $# -gt 0 ]]; do
  case $1 in
    --only) only=$2; shift 2 ;;
    --verify) verify=1; shift ;;
    --force) force=1; shift ;;
    *) echo "usage: $0 [--only yolo|person|vehicle] [--verify] [--force]" >&2; exit 2 ;;
  esac
done

missing=0
seen=" "  # paths already handled (bash 3.2 on macOS has no associative arrays)
for entry in "${ASSETS[@]}"; do
  IFS='|' read -r group url path <<<"$entry"
  [[ -n $only && $group != "$only" ]] && continue
  [[ $seen == *" $path "* ]] && continue
  seen="$seen$path "
  if [[ $verify == 1 ]]; then
    if [[ -f $path ]]; then
      printf '%s  %s\n' "$(shasum -a 256 "$path" | cut -d' ' -f1)" "$path"
    else
      echo "MISSING  $path"; missing=1
    fi
    continue
  fi
  if [[ -f $path && $force == 0 ]]; then
    echo "skip     $path (exists)"
    continue
  fi
  mkdir -p "$(dirname "$path")"
  echo "download $path"
  curl -fL --retry 10 --retry-delay 5 -C - -o "$path.download" "$url"
  mv "$path.download" "$path"
  printf '%s  %s\n' "$(shasum -a 256 "$path" | cut -d' ' -f1)" "$path"
done
exit $missing
