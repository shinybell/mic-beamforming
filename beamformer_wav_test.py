#!/usr/bin/env python3
"""
Beamforming WAV File Processor

Usage:
    python beamformer_wav_test.py <left_mic.wav> <right_mic.wav> [options]

Examples:
    # Basic usage with default angle (0°)
    python beamformer_wav_test.py left.wav right.wav

    # Specify beamforming angle
    python beamformer_wav_test.py left.wav right.wav --angle 45

    # Use MVDR beamformer
    python beamformer_wav_test.py left.wav right.wav --type mvdr --angle -30

    # Custom output path
    python beamformer_wav_test.py left.wav right.wav --output ./output/beamformed.wav
"""

import sys
import os
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from beamformer import process_wav_file
import config


def main():
    parser = argparse.ArgumentParser(
        description='Apply beamforming to stereo WAV files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s left.wav right.wav
  %(prog)s left.wav right.wav --angle 45
  %(prog)s left.wav right.wav --type mvdr --angle -30
  %(prog)s left.wav right.wav -o output.wav
        """)
    
    parser.add_argument('left_mic', type=str,
                       help='Path to left microphone WAV file')
    parser.add_argument('right_mic', type=str,
                       help='Path to right microphone WAV file')
    parser.add_argument('-o', '--output', type=str, default=None,
                       help='Output WAV file path (default: auto-generated in beamformed_output/)')
    parser.add_argument('-a', '--angle', type=float, 
                       default=config.DEFAULT_BEAMFORMING_ANGLE,
                       help=f'Beamforming angle in degrees (default: {config.DEFAULT_BEAMFORMING_ANGLE}°, 0=front, positive=right)')
    parser.add_argument('-t', '--type', type=str, 
                       choices=['delay_and_sum', 'mvdr'],
                       default='delay_and_sum',
                       help='Beamformer type (default: delay_and_sum)')
    parser.add_argument('--chunk-size', type=int, 
                       default=config.CHUNK_SIZE,
                       help=f'Chunk size for processing (default: {config.CHUNK_SIZE})')
    
    args = parser.parse_args()
    
    # Check input files exist
    if not os.path.exists(args.left_mic):
        print(f"Error: Left mic file not found: {args.left_mic}")
        sys.exit(1)
    if not os.path.exists(args.right_mic):
        print(f"Error: Right mic file not found: {args.right_mic}")
        sys.exit(1)
    
    # Generate output path if not specified
    if args.output is None:
        # Create output directory
        output_dir = config.DEFAULT_WAV_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename based on input and parameters
        left_basename = os.path.splitext(os.path.basename(args.left_mic))[0]
        output_filename = f"{left_basename}_beamformed_{args.type}_angle{args.angle:.0f}.wav"
        args.output = os.path.join(output_dir, output_filename)
    
    # Run beamforming
    print("=" * 60)
    print("Beamforming WAV File Processor")
    print("=" * 60)
    
    try:
        process_wav_file(
            left_mic_path=args.left_mic,
            right_mic_path=args.right_mic,
            output_path=args.output,
            beamformer_type=args.type,
            angle_deg=args.angle,
            chunk_size=args.chunk_size
        )
        print("=" * 60)
        print("SUCCESS!")
        print("=" * 60)
    except Exception as e:
        print("=" * 60)
        print(f"ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
