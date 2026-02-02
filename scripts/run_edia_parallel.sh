#!/bin/bash
# run_edia_parallel.sh
# Runs EDIA (density-fitness) in parallel across all PDBs in water_pdbs.txt
# Usage: ./run_edia_parallel.sh [num_jobs]

set -euo pipefail

# Source sbgrid for density-fitness (EDIA)
# Temporarily disable 'nounset' (-u) and 'errexit' (-e) because sbgrid.shrc
# uses unset variables and may return non-zero exit status
set +eu
source /programs/sbgrid.shrc
set -eu

# Configuration
PARENT_DIR="/sb/wankowicz_lab/data/srivasv/pdb_redo_data"
OUTPUT_DIR="/sb/wankowicz_lab/data/srivasv/edia_results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FULL_ID_FILE="$SCRIPT_DIR/../splits/water_pdbs.txt"
PARSER="$SCRIPT_DIR/parse_edia_json.py"

# Number of parallel jobs (default: 32, or pass as argument)
NUM_JOBS="${1:-32}"

# Validate input file exists
if [[ ! -f "$FULL_ID_FILE" ]]; then
    echo "Error: Input file not found: $FULL_ID_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Count total PDBs (excluding comments and empty lines)
TOTAL_PDBS=$(grep -cvE '^\s*$|^\s*#' "$FULL_ID_FILE" || echo "0")

echo "=============================================="
echo "EDIA (density-fitness) Parallel Pipeline"
echo "=============================================="
echo "Data directory:   $PARENT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Input file:       $FULL_ID_FILE"
echo "Total PDBs:       $TOTAL_PDBS"
echo "Parallel jobs:    $NUM_JOBS"
echo "CPUs available:   $(nproc)"
echo "=============================================="
echo ""

# Export variables for use in subshells
export PARENT_DIR OUTPUT_DIR PARSER

# Define the worker function
process_pdb() {
    local pdb_line="$1"

    # Strip whitespace, remove .pdb extension, remove _final or _final_X suffix
    local pdb=$(echo "$pdb_line" | tr -d '[:space:]' | sed -E 's/\.pdb$//; s/_final(_[A-Za-z0-9])?$//')

    # Skip empty lines or comments
    [[ -z "$pdb" || "${pdb:0:1}" == "#" ]] && return 0

    # Set up paths
    local pdb_lower=$(echo "$pdb" | tr '[:upper:]' '[:lower:]')
    local pdb_path="$PARENT_DIR/$pdb_lower"

    # Check if directory exists
    if [[ ! -d "$pdb_path" ]]; then
        echo "[SKIP] Path not found for $pdb_lower"
        return 0
    fi

    # Define input file paths
    local FINAL_MTZ_FILE="$pdb_path/${pdb_lower}_final.mtz"
    local FINAL_PDB_FILE="$pdb_path/${pdb_lower}_final.pdb"

    # Define output file paths
    local output_path="$OUTPUT_DIR/$pdb_lower"
    local FINAL_OUTPUT_FILE="$output_path/${pdb_lower}_edia.json"
    local FINAL_CSV_FILE="$output_path/${pdb_lower}_residue_stats.csv"

    # Check for required files
    if [[ ! -f "$FINAL_PDB_FILE" ]]; then
        echo "[SKIP] PDB not found: $FINAL_PDB_FILE"
        return 0
    fi

    if [[ ! -f "$FINAL_MTZ_FILE" ]]; then
        echo "[SKIP] MTZ not found: $FINAL_MTZ_FILE"
        return 0
    fi

    # Skip if already processed (check for CSV as final output)
    if [[ -f "$FINAL_CSV_FILE" ]]; then
        echo "[SKIP] Already processed: $pdb_lower"
        return 0
    fi

    # Create output directory for this PDB
    mkdir -p "$output_path"

    # Run density-fitness (EDIA)
    if density-fitness "$FINAL_MTZ_FILE" "$FINAL_PDB_FILE" -o "$FINAL_OUTPUT_FILE" 2>/dev/null; then
        # Parse output to CSV
        if [[ -f "$FINAL_OUTPUT_FILE" ]]; then
            if uv run "$PARSER" "$FINAL_OUTPUT_FILE" "$FINAL_CSV_FILE" 2>/dev/null; then
                echo "[OK] $pdb_lower"
            else
                echo "[OK-NOPARSED] $pdb_lower"
            fi
        else
            echo "[OK-NOFILE] $pdb_lower"
        fi
    else
        echo "[FAIL] $pdb_lower"
        return 0
    fi
}

# Export function for parallel
export -f process_pdb

# Run with GNU Parallel if available
if command -v parallel &> /dev/null; then
    echo "Using GNU Parallel..."
    echo ""

    grep -vE '^\s*$|^\s*#' "$FULL_ID_FILE" | \
        parallel --bar \
                 --jobs "$NUM_JOBS" \
                 --joblog "$OUTPUT_DIR/edia_joblog_$(date +%Y%m%d_%H%M%S).txt" \
                 process_pdb {}
else
    echo "GNU Parallel not found. Using xargs instead..."
    echo "(Install GNU Parallel for better progress tracking)"
    echo ""

    grep -vE '^\s*$|^\s*#' "$FULL_ID_FILE" | \
        xargs -P "$NUM_JOBS" -I {} bash -c 'process_pdb "$@"' _ {}
fi

echo ""
echo "=============================================="
echo "EDIA Pipeline complete!"
echo "Results in: $OUTPUT_DIR"
echo "=============================================="
