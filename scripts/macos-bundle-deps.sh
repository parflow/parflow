#!/usr/bin/env bash
#
# macos-bundle-deps.sh — collect shared-library dependencies into a
# self-contained ParFlow install tree and rewrite load paths so the
# result is relocatable (no references outside the tree).
#
# Usage:
#   ./scripts/macos-bundle-deps.sh <install-prefix> [dep-dir ...]
#
# <install-prefix>  The CMAKE_INSTALL_PREFIX (e.g. ~/install).
# [dep-dir ...]     Optional extra directories whose libs are already
#                   "ours" and should be copied in (e.g. ~/depend).
#
# After running, every Mach-O binary and dylib under <install-prefix>
# will load its peers via @rpath, and the executables set
# @executable_path/../lib as their RPATH.

set -euo pipefail

PREFIX="$1"; shift
DEP_DIRS=("$@")

LIB_DIR="${PREFIX}/lib"
BIN_DIR="${PREFIX}/bin"

mkdir -p "${LIB_DIR}"

is_macho() {
  file "$1" | grep -q 'Mach-O'
}

# Paths considered "system" — we never bundle these.
is_system_lib() {
  case "$1" in
    /usr/lib/*|/System/*) return 0 ;;
    *) return 1 ;;
  esac
}

# Return 0 if the path is already inside our prefix.
is_inside_prefix() {
  [[ "$1" == "${PREFIX}"/* ]]
}

# Return 0 if the path lives in one of the supplied dep dirs.
is_in_dep_dirs() {
  for d in "${DEP_DIRS[@]+"${DEP_DIRS[@]}"}"; do
    [[ "$1" == "$d"/* ]] && return 0
  done
  return 1
}

# Collect every Mach-O file under prefix.
collect_machos() {
  find "${PREFIX}" -type f | while read -r f; do
    is_macho "$f" && echo "$f"
  done
}

# Copy a library into LIB_DIR if it isn't there yet. Returns the
# destination path.
ensure_in_prefix() {
  local src="$1"
  local base
  base=$(basename "$src")
  local dst="${LIB_DIR}/${base}"

  if [[ ! -f "$dst" ]]; then
    cp -L "$src" "$dst"
    chmod u+w "$dst"
    install_name_tool -id "@rpath/${base}" "$dst" 2>/dev/null || true
  fi
  echo "$dst"
}

# Rewrite one reference inside a Mach-O file.
rewrite_ref() {
  local macho="$1" old="$2" new_name="$3"
  install_name_tool -change "$old" "$new_name" "$macho" 2>/dev/null || true
}

# Make sure executables have the right RPATH.
set_exe_rpath() {
  local exe="$1"
  local desired="@executable_path/../lib"
  local existing
  existing=$(otool -l "$exe" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || true)
  if ! echo "$existing" | grep -qF "$desired"; then
    install_name_tool -add_rpath "$desired" "$exe" 2>/dev/null || true
  fi
}

# Make sure dylibs in lib/ have a loader-relative RPATH.
set_lib_rpath() {
  local lib="$1"
  local desired="@loader_path"
  local existing
  existing=$(otool -l "$lib" 2>/dev/null | grep -A2 LC_RPATH | grep path | awk '{print $2}' || true)
  if ! echo "$existing" | grep -qF "$desired"; then
    install_name_tool -add_rpath "$desired" "$lib" 2>/dev/null || true
  fi
}

echo "=== ParFlow macOS dependency bundler ==="
echo "PREFIX : ${PREFIX}"
echo "DEP_DIRS: ${DEP_DIRS[*]+"${DEP_DIRS[*]}"}"
echo

# Phase 1 — Copy dep-dir libraries that are referenced but not yet in prefix.
# We iterate until no new libraries are pulled in (newly copied libs may
# themselves reference further libs).  A flag file is used because the inner
# loops run in subshells where variable changes don't propagate.
echo "--- Phase 1: copying dependency libraries into prefix ---"
CHANGED_FLAG=$(mktemp)
echo 1 > "$CHANGED_FLAG"
PASS=0
while [[ "$(cat "$CHANGED_FLAG")" != "0" ]]; do
  echo 0 > "$CHANGED_FLAG"
  PASS=$((PASS + 1))
  echo "  pass ${PASS}"
  while IFS= read -r macho; do
    otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r ref; do
      [[ "$ref" == @* ]] && continue
      is_system_lib "$ref" && continue
      is_inside_prefix "$ref" && continue

      if [[ -f "$ref" ]]; then
        dst=$(ensure_in_prefix "$ref")
        base=$(basename "$ref")
        rewrite_ref "$macho" "$ref" "@rpath/${base}"
        echo 1 > "$CHANGED_FLAG"
      elif is_in_dep_dirs "$ref"; then
        echo "    WARNING: referenced dep not found: $ref (from $macho)"
      fi
    done
  done < <(collect_machos)
done
rm -f "$CHANGED_FLAG"

# Phase 1b — Resolve @rpath references by searching dep dirs for the library.
CHANGED_FLAG=$(mktemp)
# Libraries like libgfortran reference libgcc_s via @rpath, which the bundler
# can't resolve from the path alone. We search all dep dirs for a match.
echo "--- Phase 1b: resolving @rpath references ---"
echo 1 > "$CHANGED_FLAG"
while [[ "$(cat "$CHANGED_FLAG")" != "0" ]]; do
  echo 0 > "$CHANGED_FLAG"
  while IFS= read -r macho; do
    otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r ref; do
      [[ "$ref" != @rpath/* ]] && continue
      base="${ref#@rpath/}"
      # Already in our lib dir? Skip.
      [[ -f "${LIB_DIR}/${base}" ]] && continue

      # Search dep dirs and common Homebrew/GCC paths for this library.
      found=""
      for search_dir in "${DEP_DIRS[@]+"${DEP_DIRS[@]}"}" \
                        /opt/homebrew/lib \
                        /opt/homebrew/opt/gcc/lib/gcc/current; do
        candidate=$(find "$search_dir" -name "$base" -type f 2>/dev/null | head -1)
        if [[ -n "$candidate" ]]; then
          found="$candidate"
          break
        fi
      done

      if [[ -n "$found" ]]; then
        echo "    Resolved @rpath/${base} -> ${found}"
        ensure_in_prefix "$found" > /dev/null
        echo 1 > "$CHANGED_FLAG"
      fi
    done
  done < <(collect_machos)
done
rm -f "$CHANGED_FLAG"

# Phase 2 — Fix install names of all dylibs in lib/ to @rpath/basename.
echo "--- Phase 2: normalising install names ---"
find "${LIB_DIR}" -name '*.dylib' -type f | while read -r lib; do
  base=$(basename "$lib")
  current_id=$(otool -D "$lib" 2>/dev/null | tail -1)
  if [[ "$current_id" != "@rpath/${base}" ]]; then
    install_name_tool -id "@rpath/${base}" "$lib" 2>/dev/null || true
  fi
done

# Phase 3 — Rewrite all remaining absolute refs inside prefix to @rpath.
echo "--- Phase 3: rewriting remaining absolute references ---"
while IFS= read -r macho; do
  otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r ref; do
    [[ "$ref" == @* ]] && continue
    is_system_lib "$ref" && continue
    base=$(basename "$ref")
    target="${LIB_DIR}/${base}"
    if [[ -f "$target" ]]; then
      rewrite_ref "$macho" "$ref" "@rpath/${base}"
    fi
  done
done < <(collect_machos)

# Phase 4 — Set RPATHs on executables and libraries.
echo "--- Phase 4: setting RPATHs ---"
find "${BIN_DIR}" -type f | while read -r f; do
  is_macho "$f" && set_exe_rpath "$f"
done
find "${LIB_DIR}" -type f -name '*.dylib' | while read -r f; do
  set_lib_rpath "$f"
done

# Phase 5 — Verify: report any remaining non-system absolute references.
echo
echo "=== Verification ==="
PROBLEMS_FLAG=$(mktemp)
echo 0 > "$PROBLEMS_FLAG"
while IFS= read -r macho; do
  otool -L "$macho" 2>/dev/null | tail -n +2 | awk '{print $1}' | while read -r ref; do
    [[ "$ref" == @* ]] && continue
    is_system_lib "$ref" && continue
    echo "  UNBUNDLED: ${ref}  (in $(basename "$macho"))"
    echo 1 > "$PROBLEMS_FLAG"
  done
done < <(collect_machos)

if [[ "$(cat "$PROBLEMS_FLAG")" == "0" ]]; then
  echo "  All references are bundled or system-provided."
fi
rm -f "$PROBLEMS_FLAG"

# Phase 6 — Re-sign Mach-O files after install_name_tool.
# Ad-hoc signing satisfies the build runner; release builds use Xcode
# cloud-managed Developer ID signing (set PARFLOW_ADHOC_SIGN=0 to skip).
echo
if [[ "${PARFLOW_ADHOC_SIGN:-1}" == "0" ]]; then
  echo "--- Phase 6: skipping ad-hoc code signing (Xcode will sign) ---"
else
  echo "--- Phase 6: ad-hoc code signing ---"
  while IFS= read -r macho; do
    codesign --force --sign - "$macho" 2>/dev/null || true
  done < <(collect_machos)
fi

echo
echo "=== Done ==="
