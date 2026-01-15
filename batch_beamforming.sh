#!/bin/bash

# Batch Beamforming Script
# Process WAV files with multiple beamforming angles

set -e

# Usage check
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <left_mic.wav> <right_mic.wav> [beamformer_type]"
    echo ""
    echo "Examples:"
    echo "  $0 left.wav right.wav"
    echo "  $0 left.wav right.wav mvdr"
    echo ""
    echo "This script will process the WAV files with multiple angles:"
    echo "  -45, -30, -15, 0, 15, 30, 45 degrees"
    exit 1
fi

LEFT_MIC="$1"
RIGHT_MIC="$2"
BEAMFORMER_TYPE="${3:-delay_and_sum}"

# Check if input files exist
if [ ! -f "$LEFT_MIC" ]; then
    echo "Error: Left mic file not found: $LEFT_MIC"
    exit 1
fi

if [ ! -f "$RIGHT_MIC" ]; then
    echo "Error: Right mic file not found: $RIGHT_MIC"
    exit 1
fi

# Angles to process
ANGLES=(-45 -30 -15 0 15 30 45)

echo "=========================================="
echo "Batch Beamforming Script"
echo "=========================================="
echo "Left mic:  $LEFT_MIC"
echo "Right mic: $RIGHT_MIC"
echo "Type:      $BEAMFORMER_TYPE"
echo "Angles:    ${ANGLES[@]}"
echo "=========================================="
echo ""

# .venv activation (uncomment if using a virtual environment)
source .venv/bin/activate

# Process each angle
for angle in "${ANGLES[@]}"; do
    echo "Processing angle: ${angle}°"
    python beamformer_wav_test.py "$LEFT_MIC" "$RIGHT_MIC" \
        --type "$BEAMFORMER_TYPE" \
        --angle "$angle"
    echo ""
done

echo "=========================================="
echo "Batch processing complete!"
echo "Output directory: ./beamformed_output"
echo "=========================================="
