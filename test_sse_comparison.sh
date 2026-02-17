#!/bin/bash
set -euo pipefail

NON_SSE_BIN="test_sse_correctness_no_sse"
SSE_BIN="test_sse_correctness_sse"
NON_SSE_OUT="test_sse_correctness_no_sse.log"
SSE_OUT="test_sse_correctness_sse.log"
DIFF_OUT="test_sse_correctness.diff"

# Cleanup previous runs
rm -f "$NON_SSE_BIN" "$SSE_BIN" "$NON_SSE_OUT" "$SSE_OUT" "$DIFF_OUT"

echo "--- Building Non-SSE version ---"
make "$NON_SSE_BIN"

echo "--- Building SSE version ---"
make "$SSE_BIN"

echo "--- Running Non-SSE tests and capturing output ---"
./"$NON_SSE_BIN" > "$NON_SSE_OUT"

echo "--- Running SSE tests and capturing output ---"
./"$SSE_BIN" > "$SSE_OUT"

echo "--- Comparing outputs ---"
# Compare outputs line by line, allowing for floating point differences
# Using awk to compare floats with a tolerance
DIFF_COUNT=0
paste "$NON_SSE_OUT" "$SSE_OUT" | while IFS=$'\t' read -r line_non_sse line_sse; do
    # Skip lines that are not float arrays (e.g., TEST headers)
    if [[ ! "$line_non_sse" =~ "]: [" ]]; then
        if [[ "$line_non_sse" != "$line_sse" ]]; then
            echo "Non-float line mismatch: '$line_non_sse' vs '$line_sse'"
            DIFF_COUNT=$((DIFF_COUNT+1))
        fi
        continue
    fi

    # Extract float values from lines
    IFS='[]' read -r -a non_sse_parts <<< "$line_non_sse"
    IFS=', ' read -r -a non_sse_vals <<< "${non_sse_parts[1]}"
    
    IFS='[]' read -r -a sse_parts <<< "$line_sse"
    IFS=', ' read -r -a sse_vals <<< "${sse_parts[1]}"

    if [[ "${#non_sse_vals[@]}" != "${#sse_vals[@]}" ]]; then
        echo "Value count mismatch for line: '$line_non_sse' vs '$line_sse'"
        DIFF_COUNT=$((DIFF_COUNT+1))
        continue
    fi

    for (( i=0; i<${#non_sse_vals[@]}; i++ )); do
        val_non_sse="${non_sse_vals[i]}"
        val_sse="${sse_vals[i]}"
        
        # Using awk for floating point comparison with tolerance
        if ! awk -v a="$val_non_sse" -v b="$val_sse" 'BEGIN { exit ( (a-b < 0 ? b-a : a-b) > 0.0001 ) }'; then # Tolerance 0.0001
            echo "Mismatch found for element $i: Non-SSE='$(printf "%.6f" "$val_non_sse")', SSE='$(printf "%.6f" "$val_sse")'"
            echo "  Non-SSE: $line_non_sse"
            echo "  SSE: $line_sse"
            DIFF_COUNT=$((DIFF_COUNT+1))
        fi
    done
done

if [[ "$DIFF_COUNT" -eq 0 ]]; then
    echo "SSE Correctness Test PASSED!"
    exit 0
else
    echo "SSE Correctness Test FAILED! Found $DIFF_COUNT mismatches."
    exit 1
fi
