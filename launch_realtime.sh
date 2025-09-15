#!/bin/bash
# Real-Time Face Animation Application Launcher
# Optimized for RTX4060Ti 16GB

echo "======================================"
echo "Real-Time Face Animation Application"
echo "======================================"
echo "Optimized for RTX4060Ti 16GB"
echo ""

# Activate conda environment
echo "Activating FaceXHuBERT environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate FaceXHuBERT

# Check if activation was successful
if [ "$CONDA_DEFAULT_ENV" != "FaceXHuBERT" ]; then
    echo "❌ Failed to activate FaceXHuBERT environment"
    echo "Please ensure the environment exists:"
    echo "  conda env list"
    exit 1
fi

echo "✅ Environment activated: $CONDA_DEFAULT_ENV"

# Check GPU availability
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}'); print(f'✅ VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**3}GB' if torch.cuda.is_available() else '❌ No CUDA GPU')"

# Check audio system
echo ""
echo "Checking audio system..."
python -c "
try:
    import pyaudio
    p = pyaudio.PyAudio()
    info = p.get_default_input_device_info()
    print(f'✅ Audio device: {info[\"name\"]}')
    print(f'✅ Sample rate: {int(info[\"defaultSampleRate\"])}Hz')
    p.terminate()
except Exception as e:
    print(f'❌ Audio error: {e}')
"

echo ""
echo "======================================"
echo "Starting Real-Time Face Animation..."
echo "======================================"
echo ""
echo "Controls:"
echo "  'q' - Quit application"
echo "  'r' - Toggle recording"
echo ""
echo "Performance Tips:"
echo "  - Close other GPU applications"
echo "  - Ensure good lighting for webcam"
echo "  - Speak clearly into microphone"
echo ""

# Launch the application
python realtime_face_app.py "$@"

echo ""
echo "Application closed."
