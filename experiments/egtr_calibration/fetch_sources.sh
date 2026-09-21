#!/usr/bin/env bash
# Fetch pinned upstream source without overwriting existing checkouts.
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p third_party
fetch_source() {
  local destination="$1" url="$2" revision="$3"
  if [[ -e "$destination" ]]; then
    if [[ ! -d "$destination/.git" ]] || [[ "$(git -C "$destination" rev-parse HEAD)" != "$revision" ]]; then
      echo "Existing $destination differs from pinned source; inspect it manually." >&2
      return 1
    fi
    echo "Using existing pinned checkout: $destination"
  else
    git clone "$url" "$destination"
    git -C "$destination" checkout --detach "$revision"
  fi
}
fetch_source third_party/egtr https://github.com/naver-ai/egtr.git 7f87450f32758ed8583948847a8186f2ee8b21e3
fetch_source third_party/CubiGraph5K https://github.com/luyueheng/CubiGraph5K.git a07dfc37cbc33731efdd0c7dc09fded46a5f6957
