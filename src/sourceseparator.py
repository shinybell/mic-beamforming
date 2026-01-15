import numpy as np
import scipy.signal as sp
import scipy.io.wavfile as wavfile
import config as config
from abc import ABC, abstractmethod
import os


class SourceSeparatorBase(ABC):
    """
    Abstract base class for source separation algorithms.
    Unlike beamformers that output a single channel, separators output multiple sources.
    """

    def __init__(
        self,
        mic_positions=config.MIC_POSITIONS,
        sample_rate=config.SAMPLE_RATE,
        chunk_size=config.CHUNK_SIZE,
    ):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_mics = len(mic_positions)

        # Pre-compute frequency bins for FFT
        # rfft returns N//2 + 1 bins
        self.freqs = np.fft.rfftfreq(chunk_size, d=1.0 / sample_rate)
        self.num_bins = len(self.freqs)

    @abstractmethod
    def apply(self, multichannel_chunk):
        """
        Apply source separation to input audio chunk.

        Args:
            multichannel_chunk: np.ndarray shape (chunk_size, num_mics)

        Returns:
            List[np.ndarray]: List of separated sources, each with shape (chunk_size,)
        """
        pass

    def update_geometry(self, spacing):
        """
        Update the microphone positions based on a new spacing (in meters).
        """
        # Calculate new positions (centered linear array)
        if self.num_mics == 2:
            self.mic_positions = np.array(
                [[-spacing / 2, 0.0, 0.0], [spacing / 2, 0.0, 0.0]]
            )
        else:
            # Fallback for >2 mics (assume uniform linear array)
            indices = np.arange(self.num_mics) - (self.num_mics - 1) / 2
            xs = indices * spacing
            self.mic_positions = np.zeros((self.num_mics, 3))
            self.mic_positions[:, 0] = xs


class AmplitudeRatioSeparation(SourceSeparatorBase):
    """
    Two-microphone sparse separation using amplitude ratio or phase difference.

    This algorithm separates two sources by analyzing the amplitude ratio or phase difference
    between two microphones. Based on the original code from:
    execute_two_microphone_sparse_separation()

    Args:
        use_amplitude: If True, use amplitude ratio for separation (amp_left/amp_right).
                      If False, use phase difference (angle(x_left/x_right)).
        amp_threshold_high: Amplitude ratio above this value assigns to left source (default: 2.0)
        amp_threshold_low: Amplitude ratio below this value assigns to right source (default: 0.5)
        noise_cancel_mode: How to handle amplitude ratios within threshold range:
                          'zero' - Set to 0 (complete noise cancellation)
                          'difference' - Use amplitude difference (preserve audio quality)
    """

    def __init__(
        self,
        mic_positions=config.MIC_POSITIONS,
        sample_rate=config.SAMPLE_RATE,
        chunk_size=config.CHUNK_SIZE,
        use_amplitude=False,
        amp_threshold_high=2.0,
        amp_threshold_low=0.5,
        noise_cancel_mode="zero",
    ):
        super().__init__(mic_positions, sample_rate, chunk_size)

        if self.num_mics != 2:
            raise ValueError(
                f"AmplitudeRatioSeparation requires exactly 2 microphones, got {self.num_mics}"
            )

        self.use_amplitude = use_amplitude
        self.amp_threshold_high = amp_threshold_high
        self.amp_threshold_low = amp_threshold_low
        self.noise_cancel_mode = noise_cancel_mode

        # STFT parameters (matching the original code's window and nperseg)
        self.window = "hann"
        self.nperseg = chunk_size  # Use same as chunk_size for consistency

    def update_separation_params(
        self,
        use_amplitude=None,
        amp_threshold_high=None,
        amp_threshold_low=None,
        noise_cancel_mode=None,
    ):
        """
        Update separation parameters dynamically.

        Args:
            use_amplitude: If True, use amplitude ratio; if False, use phase difference
            amp_threshold_high: Upper threshold for amplitude ratio
            amp_threshold_low: Lower threshold for amplitude ratio
            noise_cancel_mode: 'zero' or 'difference' for handling mid-range ratios
        """
        if use_amplitude is not None:
            self.use_amplitude = use_amplitude
        if amp_threshold_high is not None:
            self.amp_threshold_high = amp_threshold_high
        if amp_threshold_low is not None:
            self.amp_threshold_low = amp_threshold_low
        if noise_cancel_mode is not None:
            self.noise_cancel_mode = noise_cancel_mode

    def apply(self, multichannel_chunk):
        """
        Apply sparse separation to extract two sources.

        Args:
            multichannel_chunk: np.ndarray shape (chunk_size, 2) - exactly 2 microphones

        Returns:
            List[np.ndarray]: [source_left, source_right], each with shape (chunk_size,)
        """
        # Extract left and right microphone signals
        # Shape: (chunk_size,) for each
        mic_left = multichannel_chunk[:, 0]
        mic_right = multichannel_chunk[:, 1]

        # Short-time Fourier Transform
        # f: frequency bins, t: time frames, stft_data: complex spectrum
        f, t, stft_left = sp.stft(
            mic_left, fs=self.sample_rate, window=self.window, nperseg=self.nperseg
        )
        f, t, stft_right = sp.stft(
            mic_right, fs=self.sample_rate, window=self.window, nperseg=self.nperseg
        )

        # Apply sparse separation in frequency domain
        if self.use_amplitude:
            # Amplitude ratio-based separation with threshold
            # amp_ratio = |x_left| / |x_right|
            amp_ratio = np.abs(stft_left) / np.maximum(np.abs(stft_right), 1e-18)

            # Multi-level mask with noise cancellation
            mask_left = np.zeros_like(amp_ratio, dtype=np.float32)
            mask_right = np.zeros_like(amp_ratio, dtype=np.float32)

            # Strong left source: amp_ratio > threshold_high
            mask_left[amp_ratio > self.amp_threshold_high] = 1.0

            # Strong right source: amp_ratio < threshold_low
            mask_right[amp_ratio < self.amp_threshold_low] = 1.0

            # Mid-range: threshold_low <= amp_ratio <= threshold_high
            if self.noise_cancel_mode == "difference":
                # Use amplitude difference for smoother separation
                mid_range = (amp_ratio >= self.amp_threshold_low) & (
                    amp_ratio <= self.amp_threshold_high
                )
                # Normalize the ratio in mid-range to [0, 1]
                normalized_ratio = (amp_ratio - 1.0) / (
                    self.amp_threshold_high - self.amp_threshold_low
                )
                normalized_ratio = np.clip(normalized_ratio, 0.0, 1.0)

                mask_left[mid_range] = normalized_ratio[mid_range]
                mask_right[mid_range] = 1.0 - normalized_ratio[mid_range]
            # else: noise_cancel_mode == 'zero', masks remain 0 for mid-range
        else:
            # Phase difference-based separation
            # phase_difference = angle(x_left / x_right)
            phase_difference = np.angle(
                stft_left
                / np.maximum(np.abs(stft_right), 1e-18)
                * np.exp(1j * np.angle(stft_right))
            )
            # Equivalent to: np.angle(stft_left) - np.angle(stft_right)
            # But more numerically stable with division
            phase_difference = np.angle(stft_left / (stft_right + 1e-18))

            # Binary mask: if phase_difference > 0, assign to left source; otherwise to right
            mask_left = (phase_difference > 0.0).astype(np.float32)
            mask_right = (phase_difference <= 0.0).astype(np.float32)

        # Apply masks to separate sources
        stft_source_left = mask_left * stft_left
        stft_source_right = mask_right * stft_right

        # Inverse STFT to convert back to time domain
        t, source_left = sp.istft(
            stft_source_left,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.nperseg,
        )
        t, source_right = sp.istft(
            stft_source_right,
            fs=self.sample_rate,
            window=self.window,
            nperseg=self.nperseg,
        )

        # Ensure output length matches input
        source_left = source_left[: self.chunk_size].astype(np.float32)
        source_right = source_right[: self.chunk_size].astype(np.float32)

        return [source_left, source_right]


def load_stereo_wav(file_path_left, file_path_right):
    """
    Load two mono WAV files (left and right microphone recordings).

    Args:
        file_path_left: Path to left microphone WAV file
        file_path_right: Path to right microphone WAV file

    Returns:
        multichannel_data: np.ndarray shape (samples, 2), normalized to [-1, 1]
        sample_rate: Sample rate of the audio files
    """
    # Read left channel
    sr_left, data_left = wavfile.read(file_path_left)
    # Read right channel
    sr_right, data_right = wavfile.read(file_path_right)

    if sr_left != sr_right:
        raise ValueError(f"Sample rates don't match: {sr_left} vs {sr_right}")

    # Normalize to [-1, 1] based on dtype
    if data_left.dtype == np.int16:
        data_left = data_left.astype(np.float32) / 32768.0
        data_right = data_right.astype(np.float32) / 32768.0
    elif data_left.dtype == np.int32:
        data_left = data_left.astype(np.float32) / 2147483648.0
        data_right = data_right.astype(np.float32) / 2147483648.0
    else:
        data_left = data_left.astype(np.float32)
        data_right = data_right.astype(np.float32)

    # Ensure same length
    min_len = min(len(data_left), len(data_right))
    data_left = data_left[:min_len]
    data_right = data_right[:min_len]

    # Stack into multichannel format
    multichannel_data = np.stack([data_left, data_right], axis=1)

    return multichannel_data, sr_left


def save_wav(signal, file_path, sample_rate):
    """
    Save a time-domain signal to WAV file.

    Args:
        signal: np.ndarray, time-domain signal (float, normalized to [-1, 1])
        file_path: Output file path
        sample_rate: Sample rate
    """
    # Convert to int16
    signal_int16 = (signal * 32767.0).astype(np.int16)
    wavfile.write(file_path, sample_rate, signal_int16)
    print(f"Saved: {file_path}")


def test_with_wav_files(
    left_mic_path,
    right_mic_path,
    output_dir="./separated_output",
    amp_threshold_high=2.0,
    amp_threshold_low=0.5,
    noise_cancel_mode="zero",
):
    """
    Test source separation with real WAV files.

    Args:
        left_mic_path: Path to left microphone recording
        right_mic_path: Path to right microphone recording
        output_dir: Directory to save separated outputs
        amp_threshold_high: Upper threshold for amplitude ratio
        amp_threshold_low: Lower threshold for amplitude ratio
        noise_cancel_mode: 'zero' or 'difference' for handling mid-range ratios
    """
    print(f"\n{'=' * 60}")
    print(f"Testing with WAV files:")
    print(f"  Left mic:  {left_mic_path}")
    print(f"  Right mic: {right_mic_path}")
    print(f"{'=' * 60}\n")

    # Load audio files
    try:
        multichannel_data, sample_rate = load_stereo_wav(left_mic_path, right_mic_path)
        print(f"✓ Loaded audio files")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Total samples: {len(multichannel_data)}")
        print(f"  Duration: {len(multichannel_data) / sample_rate:.2f} seconds")
    except Exception as e:
        print(f"✗ Error loading files: {e}")
        return

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save original stereo audio (before separation)
    original_stereo_path = os.path.join(output_dir, "original_stereo.wav")
    # Convert multichannel data to int16 for stereo WAV
    stereo_int16 = (multichannel_data * 32767.0).astype(np.int16)
    wavfile.write(original_stereo_path, sample_rate, stereo_int16)
    print(f"✓ Saved original stereo audio: {original_stereo_path}\n")

    # Process in chunks (if audio is longer than chunk_size)
    chunk_size = config.CHUNK_SIZE
    num_samples = len(multichannel_data)

    # For simplicity, if audio is longer than one chunk, we'll process the whole thing
    # by padding or processing in overlapping chunks

    if num_samples <= chunk_size:
        # Pad if necessary
        if num_samples < chunk_size:
            padding = chunk_size - num_samples
            multichannel_data = np.pad(
                multichannel_data, ((0, padding), (0, 0)), mode="constant"
            )
        chunks_to_process = [multichannel_data]
    else:
        # Process entire audio by splitting into chunks
        chunks_to_process = []
        hop_size = chunk_size // 2  # 50% overlap
        for start in range(0, num_samples, hop_size):
            end = min(start + chunk_size, num_samples)
            chunk = multichannel_data[start:end]
            if len(chunk) < chunk_size:
                # Pad last chunk
                padding = chunk_size - len(chunk)
                chunk = np.pad(chunk, ((0, padding), (0, 0)), mode="constant")
            chunks_to_process.append((chunk, start, end))

    # Test both amplitude and phase-based separation
    for use_amp, method_name in [(True, "amplitude"), (False, "phase")]:
        print(f"\n--- {method_name.upper()}-based separation ---")

        if use_amp:
            print(f"  Threshold high: {amp_threshold_high}")
            print(f"  Threshold low: {amp_threshold_low}")
            print(f"  Noise cancel mode: {noise_cancel_mode}")

        separator = AmplitudeRatioSeparation(
            sample_rate=sample_rate,
            chunk_size=chunk_size,
            use_amplitude=use_amp,
            amp_threshold_high=amp_threshold_high,
            amp_threshold_low=amp_threshold_low,
            noise_cancel_mode=noise_cancel_mode,
        )

        if num_samples <= chunk_size:
            # Single chunk processing
            separated = separator.apply(chunks_to_process[0])
            source1_full = separated[0][:num_samples]
            source2_full = separated[1][:num_samples]
        else:
            # Multiple chunks with overlap-add
            source1_full = np.zeros(num_samples, dtype=np.float32)
            source2_full = np.zeros(num_samples, dtype=np.float32)
            window = np.hanning(chunk_size).astype(np.float32)

            for chunk, start, end in chunks_to_process:
                separated = separator.apply(chunk)
                actual_len = end - start

                # Apply window and add to output
                source1_full[start:end] += (
                    separated[0][:actual_len] * window[:actual_len]
                )
                source2_full[start:end] += (
                    separated[1][:actual_len] * window[:actual_len]
                )

        # Calculate statistics
        rms1 = np.sqrt(np.mean(source1_full**2))
        rms2 = np.sqrt(np.mean(source2_full**2))

        print(f"  Source 1 RMS: {rms1:.4f}")
        print(f"  Source 2 RMS: {rms2:.4f}")

        # Save separated sources
        output1 = os.path.join(output_dir, f"separated_{method_name}_source1.wav")
        output2 = os.path.join(output_dir, f"separated_{method_name}_source2.wav")

        save_wav(source1_full, output1, sample_rate)
        save_wav(source2_full, output2, sample_rate)

    print(f"\n{'=' * 60}")
    print(f"✓ Separation complete! Check output directory: {output_dir}")
    print(f"{'=' * 60}\n")


def create_test_audio_files(
    left_mic_path="test_audio_left.wav",
    right_mic_path="test_audio_right.wav",
    duration=3.0,
    sample_rate=44100,
):
    """
    Create test audio files with two sine waves mixed differently in each channel.

    Args:
        left_mic_path: Path for left microphone test file
        right_mic_path: Path for right microphone test file
        duration: Duration of test audio in seconds
        sample_rate: Sample rate in Hz
    """
    print(f"Creating test audio files...")
    print(f"  Duration: {duration} seconds")
    print(f"  Sample rate: {sample_rate} Hz")

    t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

    # Source 1: 440 Hz sine wave (A4)
    source1 = np.sin(2 * np.pi * 440 * t)

    # Source 2: 523.251 Hz sine wave (C5)
    source2 = np.sin(2 * np.pi * 523.251 * t)

    # Simulate microphone signals (simple mixing)
    # Left mic: source1 is louder, Right mic: source2 is louder
    mic_left = 0.8 * source1 + 0.2 * source2
    mic_right = 0.2 * source1 + 0.8 * source2

    # Save as WAV files
    save_wav(mic_left, left_mic_path, sample_rate)
    save_wav(mic_right, right_mic_path, sample_rate)

    print(f"✓ Test audio files created:")
    print(f"  {left_mic_path}")
    print(f"  {right_mic_path}")


# Test code
if __name__ == "__main__":
    import sys
    import argparse

    # Default test file paths
    default_left_mic = "test_audio_left.wav"
    default_right_mic = "test_audio_right.wav"
    default_output_dir = "./separated_output"

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Two-microphone source separation using amplitude ratio or phase difference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default test files
  python sourceseparator.py

  # Use custom audio files
  python sourceseparator.py left.wav right.wav

  # With custom thresholds and noise cancellation
  python sourceseparator.py left.wav right.wav -o output/ --threshold-high 3.0 --threshold-low 0.33 --mode difference
        """,
    )

    parser.add_argument(
        "left_mic",
        nargs="?",
        default=default_left_mic,
        help="Path to left microphone WAV file (default: test_audio_left.wav)",
    )
    parser.add_argument(
        "right_mic",
        nargs="?",
        default=default_right_mic,
        help="Path to right microphone WAV file (default: test_audio_right.wav)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=default_output_dir,
        help="Output directory for separated files (default: ./separated_output)",
    )
    parser.add_argument(
        "--threshold-high",
        type=float,
        default=2.0,
        help="Amplitude ratio above this assigns to left source (default: 2.0)",
    )
    parser.add_argument(
        "--threshold-low",
        type=float,
        default=0.5,
        help="Amplitude ratio below this assigns to right source (default: 0.5)",
    )
    parser.add_argument(
        "--mode",
        choices=["zero", "difference"],
        default="zero",
        help="Noise cancellation mode: 'zero' (complete removal) or 'difference' (preserve quality) (default: zero)",
    )

    args = parser.parse_args()

    left_mic_path = args.left_mic
    right_mic_path = args.right_mic
    output_dir = args.output
    amp_threshold_high = args.threshold_high
    amp_threshold_low = args.threshold_low
    noise_cancel_mode = args.mode

    # Show usage if using default files
    if left_mic_path == default_left_mic and right_mic_path == default_right_mic:
        print("=" * 60)
        print("No input files specified. Using test audio files.")
        print("=" * 60)
        print("\nFor help and usage:")
        print("  python sourceseparator.py --help")
        print("\n" + "=" * 60 + "\n")

    # Check if test files exist, if not create them
    if not os.path.exists(left_mic_path) or not os.path.exists(right_mic_path):
        print(f"\nTest audio files not found. Creating them...")
        create_test_audio_files(left_mic_path, right_mic_path)
        print()

    # Test with audio files
    test_with_wav_files(
        left_mic_path,
        right_mic_path,
        output_dir,
        amp_threshold_high,
        amp_threshold_low,
        noise_cancel_mode,
    )
