#!/bin/bash
# Guard script: Fail if any legacy prototype references appear in the codebase

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Patterns to search for (intentionally obfuscated to avoid self-matching)
SEARCH_WORD="mist"
PATTERNS="${SEARCH_WORD}|Mist|MIST|${SEARCH_WORD}::|\.\./${SEARCH_WORD}|/${SEARCH_WORD}/|lib${SEARCH_WORD}"

# Directories to exclude
EXCLUDE_DIRS=(
    "build"
    ".cache"
    ".venv"
    "__pycache__"
    ".git"
    "*.dSYM"
    "tools"
)

# File types to exclude
EXCLUDE_FILES=(
    "*.dat"
    "*.png"
    "*.jpg"
    "*.pdf"
    "*.o"
    "*.a"
    "*.so"
    "*.dylib"
)

echo "Checking for legacy prototype references in nest repository..."

# Build exclude args
EXCLUDE_ARGS=""
for dir in "${EXCLUDE_DIRS[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude-dir=$dir"
done
for file in "${EXCLUDE_FILES[@]}"; do
    EXCLUDE_ARGS="$EXCLUDE_ARGS --exclude=$file"
done

# Search using grep (always available)
MATCHES=$(grep -rn -E "$PATTERNS" . $EXCLUDE_ARGS 2>/dev/null || true)

if [ -n "$MATCHES" ]; then
    echo "❌ FAILED: Found references to legacy prototype in the codebase:"
    echo ""
    echo "$MATCHES"
    echo ""
    echo "Please remove all legacy prototype references to maintain project independence."
    exit 1
fi

echo "✅ PASSED: No legacy prototype references found"
exit 0

