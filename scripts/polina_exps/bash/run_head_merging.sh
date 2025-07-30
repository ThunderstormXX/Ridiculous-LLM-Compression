#!/bin/bash

# Скрипт для запуска экспериментов по объединению attention-голов с разными порогами

set -e

# Параметры по умолчанию
MODEL_PATH="src/checkpoints/llama3.1-8b"
OUTPUT_DIR="results/head_merging"
DEVICE="cuda"
THRESHOLDS=(0.98 0.99 0.995)

# Парсинг аргументов
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
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
        --thresholds)
            IFS=',' read -ra THRESHOLDS <<< "$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model_path PATH     Path to model checkpoint (default: src/checkpoints/llama3.1-8b)"
            echo "  --output_dir DIR      Output directory (default: results/head_merging)"
            echo "  --device DEVICE       Device to use (default: cuda)"
            echo "  --thresholds LIST     Comma-separated list of thresholds (default: 0.98,0.99,0.995)"
            echo "  -h, --help           Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Starting head merging experiments..."
echo "Model path: $MODEL_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Thresholds: ${THRESHOLDS[*]}"

# Создаем выходную директорию
mkdir -p "$OUTPUT_DIR"

# Проверяем существование модели
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist"
    exit 1
fi

# Запускаем эксперименты для каждого порога
for threshold in "${THRESHOLDS[@]}"; do
    echo ""
    echo "=" * 60
    echo "Running experiment with threshold: $threshold"
    echo "=" * 60
    
    python scripts/merge_attention_heads.py \
        --model_path "$MODEL_PATH" \
        --threshold "$threshold" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"
    
    if [ $? -ne 0 ]; then
        echo "Error: Experiment with threshold $threshold failed"
        exit 1
    fi
    
    echo "Completed experiment with threshold: $threshold"
done

echo ""
echo "All experiments completed successfully!"
echo ""

# Запускаем анализ результатов
echo "Running analysis..."
python scripts/analyze_head_merging.py \
    --results_dir "$OUTPUT_DIR" \
    --output_dir "$OUTPUT_DIR/analysis"

if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo "Analysis saved to: $OUTPUT_DIR/analysis"
else
    echo "Warning: Analysis failed, but experiment results are available"
fi

echo ""
echo "Experiment summary:"
echo "- Model: $MODEL_PATH"
echo "- Thresholds tested: ${THRESHOLDS[*]}"
echo "- Results directory: $OUTPUT_DIR"
echo ""
echo "To view results, check the JSON files and PNG plots in $OUTPUT_DIR"