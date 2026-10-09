#!/usr/bin/env bash
#
# linux-bundle-deps.sh — collect shared-library dependencies into a
# self-contained ParFlow install tree and set RPATH so the result is
# relocatable (no references outside the tree except glibc and core libs).
#
# Usage:
#   ./scripts/linux-bundle-deps.sh <install-prefix> [extra-search-dir ...]
#
# Requires: patchelf, file

set -euo pipefail

PREFIX="$1"
shift
SEARCH_DIRS=("$@")

LIB_DIR="${PREFIX}/lib"
BIN_DIR="${PREFIX}/bin"
LIBEXEC_DIR="${PREFIX}/libexec"
MAX_PASSES=32

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

is_system_soname() {
  is_system_lib "$1"
}

is_inside_prefix() {
  [[ "$1" == "${PREFIX}"/* ]]
}

is_absolute_path() {
  [[ "$1" == /* ]]
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
  [[ -d "${LIBEXEC_DIR}" ]] && dirs+=("${LIBEXEC_DIR}")
  [[ ${#dirs[@]} -eq 0 ]] && return 0
  find "${dirs[@]}" -type f 2>/dev/null | while read -r f; do
    is_elf "$f" && echo "$f"
  done
}

# Copy library into LIB_DIR. Prints "new" or "existing".
ensure_in_prefix() {
  local src="$1"
  local base
  base=$(basename "$src")
  local dst="${LIB_DIR}/${base}"

  if [[ ! -f "$dst" ]]; then
    cp -L "$src" "$dst"
    chmod u+w "$dst"
    patchelf --set-soname "$base" "$dst" 2>/dev/null || true
    echo "new"
  else
    echo "existing"
  fi
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

# RPATH for ELFs under install/: bin/ -> ../lib; lib/*.so -> $ORIGIN;
# lib/openmpi/mca/... and libexec/openmpi/orted -> prefix-relative path to lib/.
compute_install_rpath() {
  local elf="$1"
  local elf_dir
  elf_dir=$(dirname "$elf")

  if [[ "$elf" == "${BIN_DIR}"/* ]]; then
    echo '$ORIGIN/../lib'
    return
  fi

  if [[ "$elf_dir" == "${LIB_DIR}" ]]; then
    echo '$ORIGIN'
    return
  fi

  if [[ "$elf" != "${PREFIX}"/* ]]; then
    echo '$ORIGIN/../lib'
    return
  fi

  local rel="${elf_dir#"${PREFIX}/"}"
  local n
  n=$(awk -F/ '{print NF}' <<< "$rel")
  local up='$ORIGIN'
  local i
  for ((i = 0; i < n; i++)); do
    up="${up}/.."
  done
  echo "${up}/lib:\$ORIGIN"
}

set_install_rpath() {
  set_rpath "$1" "$(compute_install_rpath "$1")"
}

# Replace a NEEDED entry; return 0 if the entry was updated.
rewrite_needed() {
  local elf="$1" old="$2" newname="$3"
  if [[ "$old" == "$newname" ]]; then
    return 1
  fi
  if patchelf --print-needed "$elf" 2>/dev/null | grep -qF "$old"; then
    patchelf --replace-needed "$old" "$newname" "$elf" 2>/dev/null
    return 0
  fi
  return 1
}

echo "=== ParFlow Linux dependency bundler ==="
echo "PREFIX: ${PREFIX}"
echo "SEARCH_DIRS: ${SEARCH_DIRS[*]+"${SEARCH_DIRS[*]}"}"
echo

# Phase 0 — RPATH first so ldd behaves consistently in later phases.
echo "--- Phase 0: setting RPATH ---"
while IFS= read -r elf; do
  set_install_rpath "$elf"
done < <(collect_elfs)

# Phase 1 — Copy external libraries; rewrite NEEDED to soname immediately.
# CHANGED only when a new .so is copied (macOS-style convergence).
echo "--- Phase 1: copying dependency libraries into prefix ---"
CHANGED_FLAG=$(mktemp)
echo 1 > "$CHANGED_FLAG"
PASS=0
while [[ "$(cat "$CHANGED_FLAG")" != "0" ]]; do
  if [[ $PASS -ge $MAX_PASSES ]]; then
    echo "ERROR: exceeded ${MAX_PASSES} passes in phase 1; bundling is not converging" >&2
    rm -f "$CHANGED_FLAG"
    exit 1
  fi
  echo 0 > "$CHANGED_FLAG"
  PASS=$((PASS + 1))
  echo "  pass ${PASS}"

  while IFS= read -r elf; do
    while IFS= read -r needed; do
      [[ -z "$needed" ]] && continue
      # Already a bare soname and present in lib/
      if ! is_absolute_path "$needed"; then
        if [[ -f "${LIB_DIR}/${needed}" ]] || is_system_soname "$needed"; then
          continue
        fi
      fi

      [[ "$needed" == linux-vdso* ]] && continue
      [[ "$needed" == ld-linux-* ]] && continue
      is_system_lib "$needed" && continue
      is_inside_prefix "$needed" && continue

      _base=$(basename "$needed")
      _soname="$_base"
      _resolved=""

      if [[ -f "$needed" ]]; then
        _resolved="$needed"
      elif _resolved=$(resolve_lib "$needed" 2>/dev/null); then
        :
      elif _resolved=$(resolve_lib "$_soname" 2>/dev/null); then
        :
      else
        echo "    WARNING: cannot resolve ${needed} (from $(basename "$elf"))"
        continue
      fi

      _status=$(ensure_in_prefix "$_resolved")
      if [[ "$_status" == "new" ]]; then
        echo 1 > "$CHANGED_FLAG"
      fi
      if rewrite_needed "$elf" "$needed" "$_soname"; then
        echo 1 > "$CHANGED_FLAG"
      fi
    done < <(patchelf --print-needed "$elf" 2>/dev/null || true)
  done < <(collect_elfs)
done
rm -f "$CHANGED_FLAG"

# Phase 2 — Bulk rewrite: any remaining absolute NEEDED -> soname if in lib/.
echo "--- Phase 2: rewriting remaining absolute references ---"
while IFS= read -r elf; do
  while IFS= read -r needed; do
    [[ -z "$needed" ]] && continue
    is_absolute_path "$needed" || continue
    is_system_lib "$needed" && continue
    is_inside_prefix "$needed" && continue
    base=$(basename "$needed")
    if [[ -f "${LIB_DIR}/${base}" ]]; then
      rewrite_needed "$elf" "$needed" "$base" || true
    fi
  done < <(patchelf --print-needed "$elf" 2>/dev/null || true)
done < <(collect_elfs)

# Phase 3 — Re-apply RPATH after patchelf edits.
echo "--- Phase 3: refreshing RPATH ---"
while IFS= read -r elf; do
  set_install_rpath "$elf"
done < <(collect_elfs)

# Phase 4 — Verify with ldd.
echo
echo "=== Verification ==="
PROBLEMS_FLAG=$(mktemp)
echo 0 > "$PROBLEMS_FLAG"
while IFS= read -r elf; do
  while IFS= read -r line; do
    case "$line" in
      *"not found"*)
        echo "  UNRESOLVED: $(basename "$elf")"
        echo "    ${line}"
        echo 1 > "$PROBLEMS_FLAG"
        ;;
    esac
  done < <(ldd "$elf" 2>/dev/null || true)
done < <(collect_elfs)

if [[ "$(cat "$PROBLEMS_FLAG")" == "0" ]]; then
  echo "  All non-system dependencies resolve."
else
  echo "  Some dependencies are still missing (see above)."
  rm -f "$PROBLEMS_FLAG"
  exit 1
fi
rm -f "$PROBLEMS_FLAG"

echo
echo "=== Done (${PASS} pass(es) in phase 1) ==="
