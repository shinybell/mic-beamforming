import numpy as np
import src.config as config

class FreqDomainBeamformer:
    def __init__(self, mic_positions=config.MIC_POSITIONS, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_mics = len(mic_positions)
        
        # Pre-compute frequency bins for FFT
        # rfft returns N//2 + 1 bins
        self.freqs = np.fft.rfftfreq(chunk_size, d=1.0/sample_rate)
        self.num_bins = len(self.freqs)
        
        # Steering vector (weights)
        # Shape: (num_bins, num_mics)
        self.steering_vector = np.ones((self.num_bins, self.num_mics), dtype=np.complex64)
        
        # Initialize with look direction 0 degrees (broadside)
        self.update_steering_vector(0.0)

    def update_steering_vector(self, theta_deg):
        """
        Update delays for a specific angle theta (degrees).
        Theta is the angle relative to the 'front' (Y-axis) or 'side' (X-axis) depending on convention.
        Here, we assume:
        - Linear array along X-axis
        - 0 degrees = Front (positive Y direction, perpendicular to array)
        - 90 degrees = Right (positive X direction, endfire)
        - -90 degrees = Left (negative X direction)
        
        Wave vector k points TOWARDS the source.
        """
        theta_rad = np.deg2rad(theta_deg)
        
        # Direction vector of the incoming plane wave
        # Note: If source is at angle theta, the wave travels in direction -u
        # But we usually calculate delay based on projection of mic positions on direction vector.
        # Let's define direction vector u pointing TO the source.
        # u = (sin(theta), cos(theta), 0) for 0 deg = front, 90 deg = right
        ux = np.sin(theta_rad)
        uy = np.cos(theta_rad)
        uz = 0.0
        
        # Calculate time delays relative to the center (0,0,0)
        # distance = dot(mic_pos, u)
        # If mic is closer to source, it receives signal EARLIER.
        # We want to delay the earlier signals to match the latest one.
        # Or, mathematically simpler: Align phases to the center.
        # Phase shift = -omega * tau
        # tau = (d . u) / c  (time difference relative to origin)
        
        delays = np.dot(self.mic_positions, np.array([ux, uy, uz])) / config.SPEED_OF_SOUND
        
        # delays shape: (num_mics,)
        # Calculate phase shifts for each frequency
        # omega = 2 * pi * f
        # phase = -2 * pi * f * tau
        
        # Outer product to get (num_freqs, num_mics) matrix
        omega = 2 * np.pi * self.freqs
        phase_shifts = -1j * np.outer(omega, delays)
        
        # Steering vector should COMPENSATE for the delay, so we use exp(-phase_shift) or exp(+...) 
        # depending on sign convention.
        # Received signal X(f) = S(f) * exp(-j * omega * tau)
        # We want to recover S(f), so we multiply by exp(+j * omega * tau)
        self.steering_vector = np.exp(-phase_shifts) # Wait, let's verify sign. 
        # If mic is closer (tau < 0 relative), it sees signal earlier (phase advanced). 
        # We need to delay it (phase retard). 
        # Standard Beamforming: w = exp(-j * k * r). 
        # Let's stick to: Output = Sum ( W_m * X_m )
        # To align, W_m should cancel the propagation phase.
        # Propagation phase ~ exp(-j k r). Cancel with exp(+j k r)?
        # Let's try exp(-1j * ...) as compensation typically aligns phases.
        # Actually, simpler thought:
        # We want to shift time t -> t - tau.
        # Time shift property: f(t - t0) <-> F(w) exp(-j w t0).
        # We want to adding a delay corresponding to the travel time difference.
        # Let's assume we want to align everything to the origin.
        # Signal at mic m: x_m(t) = s(t - tau_m)
        # F{x_m} = S(w) exp(-j w tau_m)
        # To get S(w), we multiply by exp(+j w tau_m).
        # So we use +1j * omega * delays.
        
        self.steering_vector = np.exp(1j * np.outer(omega, delays))
        
        # Normalize weights? Standard Delay-and-Sum is 1/M Sum.
        self.steering_vector /= self.num_mics

    def apply(self, multichannel_chunk):
        """
        Apply beamforming to a chunk of multichannel audio.
        input shape: (chunk_size, num_mics)
        output shape: (chunk_size,)
        """
        # 1. FFT
        # axis 0 is time/freq, axis 1 is channel
        # rfft along axis 0
        spectrum = np.fft.rfft(multichannel_chunk, axis=0)
        
        # 2. Apply weights (Element-wise multiplication and Sum)
        # spectrum: (num_bins, num_mics)
        # steering_vector: (num_bins, num_mics)
        # Sum across channels (axis 1)
        beamformed_spectrum = np.sum(spectrum * self.steering_vector, axis=1)
        
        # 3. IFFT
        beamformed_chunk = np.fft.irfft(beamformed_spectrum, n=self.chunk_size)
        
        return beamformed_chunk.astype(np.float32)
