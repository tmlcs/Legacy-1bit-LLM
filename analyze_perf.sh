#!/bin/bash

# Function to process a single log file
process_log() {
    LOG_FILE=$1
    BUILD_TYPE=$2
    echo "--- $BUILD_TYPE Performance Summary ---"
    
    # Get unique function names
    FUNCTIONS=$(grep -oP "PERF: \K[a-zA-Z_]+" "$LOG_FILE" | sort | uniq)
    
    TOTAL_EXECUTION_TIME=0
    
    for FUNC in $FUNCTIONS; do
        # Sum up all execution times for the current function
        SUM_TIME=$(grep "PERF: $FUNC took" "$LOG_FILE" | awk '{sum+=$4} END {print sum}')
        COUNT=$(grep -c "PERF: $FUNC took" "$LOG_FILE")
        
        if [ -n "$SUM_TIME" ]; then
            echo "$FUNC: Total Time = ${SUM_TIME} ms, Calls = ${COUNT}"
            TOTAL_EXECUTION_TIME=$(awk "BEGIN {print $TOTAL_EXECUTION_TIME + $SUM_TIME}")
        fi
    done
    echo "Overall Total Execution Time: ${TOTAL_EXECUTION_TIME} ms"
    echo ""
}

process_log "sse_perf_log.txt" "SSE Build"
process_log "no_sse_perf_log.txt" "Non-SSE Build"