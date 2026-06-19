#!/bin/bash
# Runs all built tests and reports pass/fail summary.
# Run from the pstl-bench repo root: ./tests/run_tests.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="$SCRIPT_DIR/bin"

total=0
passed=0

for bin in "$BIN_DIR"/test_*; do
    name=$(basename "$bin")
    total=$((total+1))
    if "$bin" > /tmp/test_output_$$.log 2>&1; then
        if grep -q "ALL.*PASS" /tmp/test_output_$$.log; then
            echo "PASS  $name"
            passed=$((passed+1))
        else
            echo "FAIL  $name (exited 0 but no PASS summary found)"
        fi
    else
        echo "FAIL  $name (non-zero exit)"
    fi
    rm -f /tmp/test_output_$$.log
done

echo ""
echo "===================================="
echo "$passed / $total test binaries passed"
