#!/bin/bash

# Function to check if Docker is running
docker_running() {
    docker info >/dev/null 2>&1
}

# Function to check if the Docker image exists
image_exists() {
    docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^ghcr.io/pico-developer/securemr_tool:v1"
}

# Function to pull Docker image from GitHub Container Registry
pull_docker_image() {
    echo "🔄 Pulling Docker image 'ghcr.io/pico-developer/securemr_tool:v1' from GitHub Container Registry..."
    docker pull --platform=linux/amd64 ghcr.io/pico-developer/securemr_tool:v1
    if [[ $? -ne 0 ]]; then
        echo "❌ Failed to pull the Docker image."
        exit 1
    fi
}

# Function to convert model inside the Docker container
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
        custom_io_arg="-i /workspace/$custom_io"
    fi

    # Update the docker run command in convert_model function
    case "$model_ext" in
        "onnx" | "tflite" | "pb" | "pth" | "pt")
            echo "🚀 Running $model_ext model conversion inside Docker..."
            docker run --rm --platform=linux/amd64 -v "$(pwd):/workspace" ghcr.io/pico-developer/securemr_tool:v1 \
                bash -c "source /opt/qnn/2.37.1.250807/bin/envsetup.sh && \
                         source /opt/qnn/2.37.1.250807/qnn237_env/bin/activate && \
                         /opt/scripts/compile.sh -m /workspace/$input_model $custom_io_arg -o /workspace/${output_name} && \
                         qnn-context-binary-utility --context_binary /workspace/${output_name}/${model_name_wo_ext}.serialized.bin --json_file /workspace/${output_name}/${model_name_wo_ext}.serialized.json"
            ;;
        *)
            echo "❌ Unsupported model format: $model_ext"
            echo "Supported formats: .onnx, .tflite, .pb (TensorFlow), .pth/.pt (PyTorch)"
            exit 1
            ;;
    esac

    if [[ $? -eq 0 ]]; then
        echo "✅ Conversion completed: Output saved in the current directory."
    else
        echo "❌ Model conversion failed."
        exit 1
    fi
}

# Check for correct arguments
function usage() {
    echo "Usage: $0 --input <model_file> [--custom_io <custom_io.yml>]"
    exit 1
}

# Update argument parsing
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

# Check required argument
if [[ -z "$INPUT_MODEL" ]]; then
    usage
fi

# Check if custom_io file exists when specified
if [[ -n "$CUSTOM_IO" && ! -f "$CUSTOM_IO" ]]; then
    echo "❌ Error: Custom IO file '$CUSTOM_IO' not found."
    exit 1
fi

# Check if Docker daemon is running
if ! docker_running; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Check if the input file exists
if [[ ! -f "$INPUT_MODEL" ]]; then
    echo "❌ Error: File '$INPUT_MODEL' not found."
    exit 1
fi

# Check if the Docker image exists; if not, pull it from Docker Hub
if ! image_exists; then
    pull_docker_image
fi

# Run model conversion inside the container
input_abs="$(realpath "$INPUT_MODEL")"
pwd_abs="$(pwd)"
model_name="$(basename -- "$INPUT_MODEL")"
model_name_safe="$(echo "$model_name" | sed 's/[^A-Za-z0-9._-]/_/g')"
if [[ "$input_abs" != "$pwd_abs/$model_name_safe" ]]; then
    mkdir -p models
    cp "$input_abs" "models/$model_name_safe"
    INPUT_MODEL="models/$model_name_safe"
fi
convert_model "$INPUT_MODEL" "$CUSTOM_IO"
