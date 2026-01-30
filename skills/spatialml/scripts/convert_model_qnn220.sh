#!/bin/bash

set -euo pipefail

# Check if Docker is running
check_docker() {
    docker info >/dev/null 2>&1
}

# Check if Docker image exists
image_exists() {
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^ghcr.io/pico-developer/securemr_tool:v0$"
}

# Pull Docker image
pull_image() {
    echo "Pulling Docker image ghcr.io/pico-developer/securemr_tool:v0 ..."
    docker pull --platform=linux/amd64 ghcr.io/pico-developer/securemr_tool:v0
}

# Convert model inside container
convert_model() {
    local input_model="$1"
    local custom_io="$2"
    local model_name=$(basename -- "$input_model")
    local model_ext="${model_name##*.}"
    local custom_io_arg=""

    local model_name_wo_ext="${model_name%.*}"
    local output_name="${model_name%.*}_output"

    # Prepare custom_io argument if provided
    if [[ -n "$custom_io" ]]; then
        custom_io_arg="-i /app/$custom_io"
    fi

    # Update the docker run command in convert_model function
    case "$model_ext" in
        onnx|tflite|pb|pth|pt)
            echo "Running $model_ext model conversion inside Docker ..."
            docker run --rm --platform=linux/amd64 -v "$(pwd):/app" ghcr.io/pico-developer/securemr_tool:v0 \
                bash -c "source /opt/qnn/2.29.0.241129/bin/envsetup.sh && \
                         source /opt/qnn/2.29.0.241129/qnn229_env/bin/activate && \
                         /opt/scripts/compile.sh -m /app/$input_model $custom_io_arg && \
                         qnn-context-binary-utility --context_binary /app/${output_name}/${model_name_wo_ext}.serialized.bin --json_file /app/${output_name}/${model_name_wo_ext}.serialized.json "
            ;;
        *)
            echo "Unsupported model format: $model_ext"
            echo "Supported formats: .onnx, .tflite, .pb, .pth, .pt"
            exit 1
            ;;
    esac

    echo "Conversion completed. Output saved in the current directory."
}

usage() {
    echo "Usage: $0 --input <model_file> [--custom_io <custom_io.yml>]"
    exit 1
}

INPUT_MODEL=""
CUSTOM_IO=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input)
            INPUT_MODEL="$2"
            shift 2
            ;;
        --custom_io)
            CUSTOM_IO="$2"
            shift 2
            ;;
        *)
            usage
            ;;
    esac
done

if [[ -z "$INPUT_MODEL" ]]; then
    usage
fi

if [[ -n "$CUSTOM_IO" && ! -f "$CUSTOM_IO" ]]; then
    echo "Custom IO file not found: $CUSTOM_IO"
    exit 1
fi

if ! check_docker; then
    echo "Docker is not running. Start Docker and try again."
    exit 1
fi

if [[ ! -f "$INPUT_MODEL" ]]; then
    echo "Input model not found: $INPUT_MODEL"
    exit 1
fi

if ! image_exists; then
    pull_image
fi

convert_model "$INPUT_MODEL" "$CUSTOM_IO"
