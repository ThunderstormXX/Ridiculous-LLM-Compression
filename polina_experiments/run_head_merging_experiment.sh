#!/bin/bash

# Скрипт для запуска полного эксперимента по объединению attention-голов

set -e  # Остановить при ошибке

# Параметры по умолчанию
MODEL_PATH="src/checkpoints/llama3.1-8b"
THRESHOLD=0.99
OUTPUT_DIR="polina_experiments/results"
DEVICE="cuda:3"

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_path PATH    Path to model checkpoint (default: src/checkpoints/llama3.1-8b)"
            echo "  --threshold FLOAT    Cosine similarity threshold (default: 0.99)"
            echo "  --output_dir PATH    Output directory (default: polina_experiments/results)"
            echo "  --device DEVICE      Device to use (default: cpu)"
            echo "  -h, --help          Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting head merging experiment..."
echo "Model path: $MODEL_PATH"
echo "Threshold: $THRESHOLD"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo ""

# Создаем выходную директорию
mkdir -p "$OUTPUT_DIR"

# Файлы для промежуточных результатов
SIMILARITY_FILE="$OUTPUT_DIR/similar_heads_threshold_${THRESHOLD}.json"
MERGED_MODEL_PATH="$OUTPUT_DIR/merged_model_threshold_${THRESHOLD}"

echo "Step 1: Finding similar attention heads..."
python polina_experiments/step1_find_similar_heads.py \
    --model_path "$MODEL_PATH" \
    --threshold "$THRESHOLD" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

if [ ! -f "$SIMILARITY_FILE" ]; then
    echo "Error: Similarity analysis failed. File $SIMILARITY_FILE not found."
    exit 1
fi

echo ""
echo "Step 2: Merging similar heads..."
python polina_experiments/step2_merge_heads.py \
    --model_path "$MODEL_PATH" \
    --similarity_file "$SIMILARITY_FILE" \
    --output_model_path "$MERGED_MODEL_PATH" \
    --device "$DEVICE"

if [ ! -d "$MERGED_MODEL_PATH" ]; then
    echo "Error: Head merging failed. Directory $MERGED_MODEL_PATH not found."
    exit 1
fi

echo ""
echo "Step 3: Evaluating and visualizing results..."
python polina_experiments/step3_evaluate_and_visualize.py \
    --original_model_path "$MODEL_PATH" \
    --merged_model_path "$MERGED_MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE"

echo ""
echo "Experiment completed successfully!"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Generated files:"
echo "  - $SIMILARITY_FILE (similarity analysis)"
echo "  - $MERGED_MODEL_PATH/ (merged model)"
echo "  - $OUTPUT_DIR/evaluation_results.json (evaluation results)"
echo "  - $OUTPUT_DIR/metrics_comparison.png (metrics comparison plot)"
echo "  - $OUTPUT_DIR/merged_pairs_per_layer.png (merged pairs per layer plot)"
echo "  - $OUTPUT_DIR/metrics_change.png (metrics change plot)"