#!/bin/bash
# Builds all standalone correctness tests across 7 datatypes for all 5 algorithms.
# Run from the pstl-bench repo root: ./tests/build_tests.sh

set -e
source ~/setup_env.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
BIN_DIR="$SCRIPT_DIR/bin"
INCLUDE_DIR="$SCRIPT_DIR/../include"

mkdir -p "$BIN_DIR"

ALGOS=(find for_each reduce scan sort)
TYPES=(char short int long "long long" float double)

for algo in "${ALGOS[@]}"; do
    echo "=== Building $algo (default double) ==="
    ~/adaptivecpp/bin/acpp -I "$INCLUDE_DIR" -I"$TBB_OLD/include" \
        -DACPP_TARGETS=cuda:sm_70 -O2 \
        "$SRC_DIR/test_${algo}.cpp" -o "$BIN_DIR/test_${algo}"

    for type in "${TYPES[@]}"; do
        safe_name=$(echo "$type" | tr ' ' '_')
        echo "=== Building $algo for type=$type ==="
        ~/adaptivecpp/bin/acpp -I "$INCLUDE_DIR" -I"$TBB_OLD/include" \
            -DACPP_TARGETS=cuda:sm_70 -DPSTL_ELEM_T="$type" -O2 \
            "$SRC_DIR/test_${algo}.cpp" -o "$BIN_DIR/test_${algo}_${safe_name}"
    done

    echo "=== Building $algo cache test ==="
    ~/adaptivecpp/bin/acpp -I "$INCLUDE_DIR" -I"$TBB_OLD/include" \
        -DACPP_TARGETS=cuda:sm_70 -O2 \
        "$SRC_DIR/test_${algo}_cache.cpp" -o "$BIN_DIR/test_${algo}_cache"
done

echo ""
echo "All tests built into $BIN_DIR"
