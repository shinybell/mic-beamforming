#!/bin/bash

# Source Separation Comparison Script
# This script runs source separation with different noise cancellation modes

set -e  # Exit on error

# Input files
LEFT_MIC="momotaro_ch1.wav"
RIGHT_MIC="momotaro_ch2.wav"

# Check if input files exist
if [ ! -f "$LEFT_MIC" ]; then
    echo "Error: $LEFT_MIC not found"
    exit 1
fi

if [ ! -f "$RIGHT_MIC" ]; then
    echo "Error: $RIGHT_MIC not found"
    exit 1
fi

# Default parameters
THRESHOLD_HIGH=2.0
THRESHOLD_LOW=0.5

# .venv activation (uncomment if using a virtual environment)
source .venv/bin/activate

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --threshold-high)
            THRESHOLD_HIGH="$2"
            shift 2
            ;;
        --threshold-low)
            THRESHOLD_LOW="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --threshold-high VALUE   Set high threshold (default: 2.0)"
            echo "  --threshold-low VALUE    Set low threshold (default: 0.5)"
            echo "  -h, --help              Show this help message"
            echo ""
            echo "Input files: $LEFT_MIC, $RIGHT_MIC"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "Source Separation Comparison"
echo "============================================================"
echo "Input files:"
echo "  Left:  $LEFT_MIC"
echo "  Right: $RIGHT_MIC"
echo ""
echo "Parameters:"
echo "  Threshold high: $THRESHOLD_HIGH"
echo "  Threshold low:  $THRESHOLD_LOW"
echo "============================================================"
echo ""

# Run separation with mode='zero'
echo "▶ Running separation with mode='zero' (noise cancellation)..."
python src/sourceseparator.py \
    "$LEFT_MIC" "$RIGHT_MIC" \
    -o "./separated_output_zero" \
    --threshold-high "$THRESHOLD_HIGH" \
    --threshold-low "$THRESHOLD_LOW" \
    --mode zero

echo ""
echo "============================================================"
echo ""

# Run separation with mode='difference'
echo "▶ Running separation with mode='difference' (quality preservation)..."
python src/sourceseparator.py \
    "$LEFT_MIC" "$RIGHT_MIC" \
    -o "./separated_output_difference" \
    --threshold-high "$THRESHOLD_HIGH" \
    --threshold-low "$THRESHOLD_LOW" \
    --mode difference

echo ""
echo "============================================================"
echo "✓ All separations complete!"
echo "============================================================"
echo ""
echo "Output directories:"
echo "  • separated_output_zero/         (noise cancellation mode)"
echo "  • separated_output_difference/   (quality preservation mode)"
echo ""
echo "Files in each directory:"
echo "  • original_stereo.wav"
echo "  • separated_amplitude_source1.wav"
echo "  • separated_amplitude_source2.wav"
echo "  • separated_phase_source1.wav"
echo "  • separated_phase_source2.wav"
echo ""
