#!/bin/bash
# =============================================================================
# WaterFlow Docker Entrypoint
# Unified CLI that dispatches commands to the appropriate scripts
# =============================================================================

set -e

# Show help if no arguments
show_help() {
    cat << EOF
WaterFlow Docker Container

Usage: docker run waterflow <command> [options]

Commands:
  train           Run training pipeline
  inference       Run inference on PDB files
  generate-esm    Generate ESM3 embeddings for PDB files
  python          Run arbitrary Python script or command
  --help          Show this help message

For interactive shell: docker run --entrypoint /bin/bash -it waterflow

Examples:
  # Training
  docker run --gpus all -v /path/to/data:/data waterflow train \\
      --train_list /data/splits/train.txt \\
      --val_list /data/splits/val.txt \\
      --encoder_type esm \\
      --epochs 200

  # Generate ESM embeddings
  docker run --gpus all -v /path/to/data:/data waterflow generate-esm \\
      --split_file /data/splits/train.txt

  # Inference
  docker run --gpus all -v /path/to/data:/data waterflow inference \\
      --run_dir /data/checkpoints/run_name \\
      --pdb_list /data/splits/test.txt \\
      --output_dir /data/outputs

Environment Variables:
  WATERFLOW_PDB_DIR         Base directory for PDB files (default: /data/pdb)
  WATERFLOW_CACHE_DIR       Cache directory for embeddings (default: /data/cache)
  WATERFLOW_CHECKPOINT_DIR  Checkpoint directory (default: /data/checkpoints)
  WATERFLOW_OUTPUT_DIR      Output directory (default: /data/outputs)
  WATERFLOW_LOG_DIR         Log directory (default: /data/logs)
  WATERFLOW_SPLITS_DIR      Split/list files directory (default: /data/splits)
  WANDB_API_KEY             Weights & Biases API key (set via -e or docker-compose)
  WANDB_PROJECT             W&B project name (set via -e or docker-compose)

EOF
}

# Get the command
COMMAND="${1:-}"

case "$COMMAND" in
    train)
        shift
        exec python /app/scripts/train.py \
            --base_pdb_dir "${WATERFLOW_PDB_DIR}" \
            --processed_dir "${WATERFLOW_CACHE_DIR}" \
            --save_dir "${WATERFLOW_CHECKPOINT_DIR}" \
            --wandb_dir "${WATERFLOW_LOG_DIR}" \
            "$@"
        ;;

    inference)
        shift
        exec python /app/scripts/inference.py \
            --base_pdb_dir "${WATERFLOW_PDB_DIR}" \
            --processed_dir "${WATERFLOW_CACHE_DIR}" \
            --output_dir "${WATERFLOW_OUTPUT_DIR}" \
            "$@"
        ;;

    generate-esm)
        shift
        exec python /app/scripts/generate_esm_embeddings.py \
            --base_pdb_dir "${WATERFLOW_PDB_DIR}" \
            --cache_dir "${WATERFLOW_CACHE_DIR}" \
            "$@"
        ;;

    python)
        shift
        exec python "$@"
        ;;

    --help|-h|help)
        show_help
        exit 0
        ;;

    "")
        show_help
        exit 0
        ;;

    *)
        echo "Error: Unknown command '$COMMAND'"
        echo ""
        show_help
        exit 1
        ;;
esac
