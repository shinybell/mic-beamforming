#!/bin/bash

# Batch Mel Spectrogram Generator
# WAVファイルまたはディレクトリを指定してメルスペクトログラムを一括生成

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Show usage
show_usage() {
    echo "Usage: $0 <file_or_directory> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  file_or_directory    WAV file or directory containing WAV files"
    echo ""
    echo "Options:"
    echo "  -r, --recursive      Search for WAV files recursively in subdirectories"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Note:"
    echo "  Spectrograms are saved in the same directory as the source WAV files"
    echo ""
    echo "Examples:"
    echo "  # Process single file (output: same directory as input)"
    echo "  $0 audio.wav"
    echo ""
    echo "  # Process all WAV files in directory"
    echo "  $0 ./audio_folder/"
    echo ""
    echo "  # Process recursively"
    echo "  $0 ./audio_folder/ -r"
}

# Parse arguments
RECURSIVE=false

if [ $# -eq 0 ]; then
    show_usage
    exit 1
fi

INPUT_PATH="$1"
shift

# .venv activation (uncomment if using a virtual environment)
source .venv/bin/activate

while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--recursive)
            RECURSIVE=true
            shift
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Check if input exists
if [ ! -e "$INPUT_PATH" ]; then
    echo -e "${RED}Error: File or directory not found: $INPUT_PATH${NC}"
    exit 1
fi

echo "============================================================"
echo "Batch Mel Spectrogram Generator"
echo "============================================================"
echo "Input: $INPUT_PATH"
echo "Recursive: $RECURSIVE"
echo "Note: Spectrograms will be saved in the same directory as source files"
echo "============================================================"
echo ""

# Function to process a single WAV file
process_wav_file() {
    local wav_file="$1"
    local wav_dir=$(dirname "$wav_file")
    local base_name=$(basename "$wav_file" .wav)
    local output_file="${wav_dir}/${base_name}_melspec.png"

    echo -e "${YELLOW}▶${NC} Processing: $wav_file"

    if python wav_to_melspec.py "$wav_file" "$output_file"; then
        echo -e "${GREEN}  ✓ Success${NC}"
        return 0
    else
        echo -e "${RED}  ✗ Failed${NC}"
        return 1
    fi
}

# Process files
processed=0
failed=0

if [ -f "$INPUT_PATH" ]; then
    # Single file mode
    if [[ "$INPUT_PATH" =~ \.wav$ ]] || [[ "$INPUT_PATH" =~ \.WAV$ ]]; then
        if process_wav_file "$INPUT_PATH"; then
            ((processed++))
        else
            ((failed++))
        fi
    else
        echo -e "${RED}Error: File is not a WAV file: $INPUT_PATH${NC}"
        exit 1
    fi

elif [ -d "$INPUT_PATH" ]; then
    # Directory mode
    echo "Searching for WAV files..."
    echo ""

    if [ "$RECURSIVE" = true ]; then
        # Recursive search
        wav_files=$(find "$INPUT_PATH" -type f \( -iname "*.wav" -o -iname "*.WAV" \))
    else
        # Non-recursive search
        wav_files=$(find "$INPUT_PATH" -maxdepth 1 -type f \( -iname "*.wav" -o -iname "*.WAV" \))
    fi

    # Count total files
    total=$(echo "$wav_files" | grep -c . || echo "0")

    if [ "$total" -eq 0 ]; then
        echo -e "${YELLOW}Warning: No WAV files found in $INPUT_PATH${NC}"
        exit 0
    fi

    echo "Found $total WAV file(s)"
    echo ""

    # Process each file
    count=0
    while IFS= read -r wav_file; do
        if [ -n "$wav_file" ]; then
            ((count++))
            echo "[$count/$total]"

            if process_wav_file "$wav_file"; then
                ((processed++))
            else
                ((failed++))
            fi

            echo ""
        fi
    done <<< "$wav_files"
else
    echo -e "${RED}Error: Invalid input path: $INPUT_PATH${NC}"
    exit 1
fi

# Summary
echo "============================================================"
echo "Processing complete!"
echo "============================================================"
echo -e "Processed: ${GREEN}$processed${NC}"
if [ $failed -gt 0 ]; then
    echo -e "Failed:    ${RED}$failed${NC}"
fi
echo "Note: Spectrograms saved in the same directory as source WAV files"
echo "============================================================"

exit 0
