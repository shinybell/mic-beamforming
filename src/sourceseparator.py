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
    """

    def __init__(
        self,
        mic_positions=config.MIC_POSITIONS,
        sample_rate=config.SAMPLE_RATE,
        chunk_size=config.CHUNK_SIZE,
        use_amplitude=False,
    ):
        super().__init__(mic_positions, sample_rate, chunk_size)

        if self.num_mics != 2:
            raise ValueError(
                f"AmplitudeRatioSeparation requires exactly 2 microphones, got {self.num_mics}"
            )

        self.use_amplitude = use_amplitude

        # STFT parameters (matching the original code's window and nperseg)
        self.window = "hann"
        self.nperseg = chunk_size  # Use same as chunk_size for consistency

    def update_separation_params(self, use_amplitude):
        """
        Update separation parameters dynamically.

        Args:
            use_amplitude: If True, use amplitude ratio; if False, use phase difference
        """
        self.use_amplitude = use_amplitude

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
            # Amplitude ratio-based separation
            # amp_ratio = |x_left| / |x_right|
            amp_ratio = np.abs(stft_left) / np.maximum(np.abs(stft_right), 1e-18)

            # Binary mask: if amp_ratio > 1, assign to left source; otherwise to right
            mask_left = (amp_ratio > 1.0).astype(np.float32)
            mask_right = (amp_ratio <= 1.0).astype(np.float32)
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


def test_with_wav_files(left_mic_path, right_mic_path, output_dir="./separated_output"):
    """
    Test source separation with real WAV files.

    Args:
        left_mic_path: Path to left microphone recording
        right_mic_path: Path to right microphone recording
        output_dir: Directory to save separated outputs
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

        separator = AmplitudeRatioSeparation(
            sample_rate=sample_rate, chunk_size=chunk_size, use_amplitude=use_amp
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


# Test code
if __name__ == "__main__":
    import sys

    # Check if WAV file paths are provided as command-line arguments
    if len(sys.argv) >= 3:
        left_mic_path = sys.argv[1]
        right_mic_path = sys.argv[2]
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./separated_output"

        # Test with real audio files
        test_with_wav_files(left_mic_path, right_mic_path, output_dir)
    else:
        # Run synthetic test
        print("=" * 60)
        print("Running synthetic signal test...")
        print("=" * 60)
        print("\nUsage for real audio files:")
        print("  python sourceseparator.py <left_mic.wav> <right_mic.wav> [output_dir]")
        print("\n" + "=" * 60 + "\n")

        print("Testing AmplitudeRatioSeparation...")

        # Create a simple test signal
        duration = config.CHUNK_SIZE / config.SAMPLE_RATE  # seconds
        t = np.linspace(0, duration, config.CHUNK_SIZE, endpoint=False)

        # Source 1: 440 Hz sine wave (A4)
        source1 = np.sin(2 * np.pi * 440 * t)

        # Source 2: 880 Hz sine wave (A5)
        source2 = np.sin(2 * np.pi * 880 * t)

        # Simulate microphone signals (simple mixing)
        # Left mic: source1 is louder, Right mic: source2 is louder
        mic_left = 0.8 * source1 + 0.2 * source2
        mic_right = 0.2 * source1 + 0.8 * source2

        # Stack into multichannel format: (chunk_size, 2)
        multichannel_chunk = np.stack([mic_left, mic_right], axis=1)

        # Test amplitude-based separation
        print("\n--- Amplitude-based separation ---")
        separator_amp = AmplitudeRatioSeparation(use_amplitude=True)
        separated_amp = separator_amp.apply(multichannel_chunk)
        print(f"Number of separated sources: {len(separated_amp)}")
        print(
            f"Source 1 shape: {separated_amp[0].shape}, dtype: {separated_amp[0].dtype}"
        )
        print(
            f"Source 2 shape: {separated_amp[1].shape}, dtype: {separated_amp[1].dtype}"
        )
        print(f"Source 1 RMS: {np.sqrt(np.mean(separated_amp[0] ** 2)):.4f}")
        print(f"Source 2 RMS: {np.sqrt(np.mean(separated_amp[1] ** 2)):.4f}")

        # Test phase-based separation
        print("\n--- Phase-based separation ---")
        separator_phase = AmplitudeRatioSeparation(use_amplitude=False)
        separated_phase = separator_phase.apply(multichannel_chunk)
        print(f"Number of separated sources: {len(separated_phase)}")
        print(
            f"Source 1 shape: {separated_phase[0].shape}, dtype: {separated_phase[0].dtype}"
        )
        print(
            f"Source 2 shape: {separated_phase[1].shape}, dtype: {separated_phase[1].dtype}"
        )
        print(f"Source 1 RMS: {np.sqrt(np.mean(separated_phase[0] ** 2)):.4f}")
        print(f"Source 2 RMS: {np.sqrt(np.mean(separated_phase[1] ** 2)):.4f}")

        # Test dynamic parameter update
        print("\n--- Testing parameter update ---")
        separator_phase.update_separation_params(use_amplitude=True)
        separated_updated = separator_phase.apply(multichannel_chunk)
        print(f"After switching to amplitude mode:")
        print(f"Source 1 RMS: {np.sqrt(np.mean(separated_updated[0] ** 2)):.4f}")
        print(f"Source 2 RMS: {np.sqrt(np.mean(separated_updated[1] ** 2)):.4f}")

        print("\n✓ All tests completed successfully!")
