#!/usr/bin/env bash
# Setup + run the easy_ViTPose webapp (pose, shuttlecock, court tracking, TrackNetV3)
# on a RunPod GPU pod. Run this INSIDE the pod's terminal after uploading/extracting
# easy_vitpose_colab_code.zip (same zip used for the Colab attempt -- it's just source
# code, works here too).
#
# Usage:
#   bash runpod_setup.sh
#
# `set -e`: stops immediately on the first real failure instead of silently
# continuing with a half-broken install (that's what made the Colab attempt
# hard to debug -- failures were easy to miss).
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
echo "==> Working in $PROJECT_DIR"

echo "==> [1/5] Python/CUDA sanity check"
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"

echo "==> [2/5] Installing Python dependencies"
pip install -q -r requirements.txt
pip install -q flask parse gdown huggingface_hub
# GPU build of onnxruntime for faster ViTPose inference; falls back to CPU build if unavailable.
pip install -q onnxruntime-gpu || pip install -q onnxruntime
apt-get -qq update && apt-get -qq install -y ffmpeg libgl1 > /dev/null

echo "==> [3/5] Downloading AI models (skips any already present)"
mkdir -p models models/tracknet
[ -f models/vitpose-l-coco.onnx ] || wget -q -O models/vitpose-l-coco.onnx https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/coco/vitpose-l-coco.onnx
[ -f models/vitpose-l-ap10k.onnx ] || wget -q -O models/vitpose-l-ap10k.onnx https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/ap10k/vitpose-l-ap10k.onnx
[ -f models/yolov8l.pt ] || wget -q -O models/yolov8l.pt https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8l.pt
if [ ! -f models/tracknet/TrackNet_best.pt ]; then
    gdown 1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA -O /tmp/TrackNetV3_ckpts.zip
    unzip -q -o /tmp/TrackNetV3_ckpts.zip -d /tmp/tracknet_ckpts_tmp
    mv /tmp/tracknet_ckpts_tmp/ckpts/*.pt models/tracknet/
    rm -rf /tmp/tracknet_ckpts_tmp /tmp/TrackNetV3_ckpts.zip
fi
echo "Models present:"
ls -lh models/*.onnx models/*.pt models/tracknet/*.pt

echo "==> [4/5] Roboflow API key"
if [ -f webapp/.env ]; then
    echo "webapp/.env already exists, leaving it as-is."
else
    read -rsp "Roboflow Private API key (blank to skip): " API_KEY
    echo
    if [ -n "$API_KEY" ]; then
        echo "ROBOFLOW_API_KEY=$API_KEY" > webapp/.env
        echo "Saved to webapp/.env"
    else
        echo "Skipped -- Shuttlecock/Gabungan/Posisi Lapangan modes will error until webapp/.env is set."
    fi
fi

echo "==> [5/5] Starting webapp on 0.0.0.0:5050"
echo "    Open this pod's HTTP port 5050 in the RunPod dashboard to get the public URL."
export HOST=0.0.0.0
export PORT=5050
python webapp/app.py
