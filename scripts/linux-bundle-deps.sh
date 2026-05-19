#!/usr/bin/env bash
#
# linux-bundle-deps.sh — collect shared-library dependencies into a
# self-contained ParFlow install tree and set RPATH so the result is
# relocatable (no references outside the tree except glibc and core libs).
#
# Usage:
#   ./scripts/linux-bundle-deps.sh <install-prefix> [extra-search-dir ...]
#
# Requires: patchelf, file, readelf, ldd

set -euo pipefail

PREFIX="$1"
shift
SEARCH_DIRS=("$@")

LIB_DIR="${PREFIX}/lib"
BIN_DIR="${PREFIX}/bin"

mkdir -p "${LIB_DIR}"

if ! command -v patchelf >/dev/null 2>&1; then
  echo "ERROR: patchelf is required" >&2
  exit 1
fi

is_elf() {
  file "$1" 2>/dev/null | grep -q 'ELF'
}

# Core C library / dynamic linker — never bundle (by soname basename).
is_system_lib() {
  local base
  base=$(basename "$1")
  case "$base" in
    ld-linux-*|linux-vdso.so.*) return 0 ;;
    libc.so*|libm.so*|libdl.so*|libpthread.so*|librt.so*|libresolv.so*|libutil.so*|libnsl.so*|libcrypt.so*) return 0 ;;
    libnss_*.so*|libtinfo.so*|libselinux.so*|libpcre2-*.so*|libpcre.so*|libcom_err.so*|libkeyutils.so*) return 0 ;;
  esac
  return 1
}

is_inside_prefix() {
  [[ "$1" == "${PREFIX}"/* ]]
}

resolve_lib() {
  local ref="$1"
  if [[ -f "$ref" ]]; then
    echo "$ref"
    return 0
  fi
  local base
  base=$(basename "$ref")
  if [[ -f "${LIB_DIR}/${base}" ]]; then
    echo "${LIB_DIR}/${base}"
    return 0
  fi
  for d in "${SEARCH_DIRS[@]+"${SEARCH_DIRS[@]}"}"; do
    if [[ -f "${d}/${base}" ]]; then
      echo "${d}/${base}"
      return 0
    fi
    if [[ -f "${d}/lib/${base}" ]]; then
      echo "${d}/lib/${base}"
      return 0
    fi
    local found
    found=$(find "$d" -name "$base" -type f 2>/dev/null | head -1)
    if [[ -n "$found" ]]; then
      echo "$found"
      return 0
    fi
  done
  return 1
}

collect_elfs() {
  local -a dirs=()
  [[ -d "${BIN_DIR}" ]] && dirs+=("${BIN_DIR}")
  [[ -d "${LIB_DIR}" ]] && dirs+=("${LIB_DIR}")
  [[ ${#dirs[@]} -eq 0 ]] && return 0
  find "${dirs[@]}" -type f 2>/dev/null | while read -r f; do
    is_elf "$f" && echo "$f"
  done
}

ensure_in_prefix() {
  local src="$1"
  local base
  base=$(basename "$src")
  local dst="${LIB_DIR}/${base}"

  if [[ ! -f "$dst" ]]; then
    cp -L "$src" "$dst"
    chmod u+w "$dst"
    patchelf --set-soname "$base" "$dst" 2>/dev/null || true
  fi
  echo "$dst"
}

set_rpath() {
  local elf="$1"
  local rpath="$2"
  local current
  current=$(patchelf --print-rpath "$elf" 2>/dev/null || true)
  if [[ "$current" != "$rpath" ]]; then
    patchelf --set-rpath "$rpath" "$elf" 2>/dev/null || true
  fi
}

rewrite_needed() {
  local elf="$1" old="$2" newname="$3"
  if patchelf --print-needed "$elf" 2>/dev/null | grep -qF "$old"; then
    patchelf --replace-needed "$old" "$newname" "$elf" 2>/dev/null || true
  fi
}

echo "=== ParFlow Linux dependency bundler ==="
echo "PREFIX: ${PREFIX}"
echo "SEARCH_DIRS: ${SEARCH_DIRS[*]+"${SEARCH_DIRS[*]}"}"
echo

# Phase 1 — copy missing shared libraries and fix NEEDED entries
echo "--- Phase 1: copying dependency libraries into prefix ---"
CHANGED_FLAG=$(mktemp)
echo 1 > "$CHANGED_FLAG"
PASS=0
while [[ "$(cat "$CHANGED_FLAG")" != "0" ]]; do
  echo 0 > "$CHANGED_FLAG"
  PASS=$((PASS + 1))
  echo "  pass ${PASS}"
  while IFS= read -r elf; do
    while IFS= read -r line; do
      case "$line" in
        *"not found"*|*" => "*) ;;
        *) continue ;;
      esac
      soname="${line%% =>*}"
      soname="${soname%% *}"

      ref=""
      if [[ "$line" == *"not found"* ]]; then
        ref="$soname"
      else
        ref=$(echo "$line" | awk '{print $3}')
      fi

      [[ "$soname" == linux-vdso* ]] && continue
      [[ "$soname" == ld-linux-* ]] && continue
      is_system_lib "$ref" && continue
      is_inside_prefix "$ref" && continue

      resolved=""
      if [[ -f "$ref" ]]; then
        resolved="$ref"
      elif resolved=$(resolve_lib "$ref" 2>/dev/null); then
        :
      elif resolved=$(resolve_lib "$soname" 2>/dev/null); then
        :
      else
        echo "    WARNING: cannot resolve ${soname} (from $(basename "$elf"))"
        continue
      fi

      ensure_in_prefix "$resolved" >/dev/null
      if [[ "$line" != *"not found"* && "$ref" != "$soname" ]]; then
        rewrite_needed "$elf" "$ref" "$soname"
      fi
      echo 1 > "$CHANGED_FLAG"
    done < <(ldd "$elf" 2>/dev/null || true)
  done < <(collect_elfs)
done
rm -f "$CHANGED_FLAG"

# Phase 2 — set RPATH on all ELF files
echo "--- Phase 2: setting RPATH ---"
while IFS= read -r elf; do
  case "$elf" in
    "${BIN_DIR}"/*) set_rpath "$elf" '$ORIGIN/../lib' ;;
    "${LIB_DIR}"/*) set_rpath "$elf" '$ORIGIN' ;;
    *) set_rpath "$elf" '$ORIGIN/../lib' ;;
  esac
done < <(collect_elfs)

# Phase 3 — verify
echo
echo "=== Verification ==="
PROBLEMS_FLAG=$(mktemp)
echo 0 > "$PROBLEMS_FLAG"
while IFS= read -r elf; do
  ldd "$elf" 2>/dev/null | grep -q 'not found' && {
    echo "  UNRESOLVED: $(basename "$elf")"
    ldd "$elf" 2>/dev/null | grep 'not found' | sed 's/^/    /'
    echo 1 > "$PROBLEMS_FLAG"
  }
done < <(collect_elfs)

if [[ "$(cat "$PROBLEMS_FLAG")" == "0" ]]; then
  echo "  All non-system dependencies resolve."
else
  echo "  Some dependencies are still missing (see above)."
  exit 1
fi
rm -f "$PROBLEMS_FLAG"

echo
echo "=== Done ==="
