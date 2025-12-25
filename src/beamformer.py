import numpy as np
import config as config
from abc import ABC, abstractmethod

class BeamformerBase(ABC):
    def __init__(self, mic_positions=config.MIC_POSITIONS, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE):
        self.mic_positions = mic_positions
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.num_mics = len(mic_positions)
        
        # Pre-compute frequency bins for FFT
        # rfft returns N//2 + 1 bins
        self.freqs = np.fft.rfftfreq(chunk_size, d=1.0/sample_rate)
        self.num_bins = len(self.freqs)
        
        # Current look direction
        self.current_angle = 0.0

    @abstractmethod
    def update_steering_vector(self, theta_deg):
        pass

    @abstractmethod
    def apply(self, multichannel_chunk):
        pass

    def update_geometry(self, spacing):
        """
        Update the microphone positions based on a new spacing (in meters).
        """
        # Calculate new positions (centered linear array)
        if self.num_mics == 2:
            self.mic_positions = np.array([
                [-spacing / 2, 0.0, 0.0],
                [ spacing / 2, 0.0, 0.0]
            ])
        else:
            # Fallback for >2 mics (assume uniform linear array)
            indices = np.arange(self.num_mics) - (self.num_mics - 1) / 2
            xs = indices * spacing
            self.mic_positions = np.zeros((self.num_mics, 3))
            self.mic_positions[:, 0] = xs
            
        # Recalculate steering vector for current angle with new geometry
        self.update_steering_vector(self.current_angle)
    
    def _calculate_delays(self, theta_deg):
        """Helper to calculate delays for a given angle."""
        theta_rad = np.deg2rad(theta_deg)
        ux = np.sin(theta_rad)
        uy = np.cos(theta_rad)
        uz = 0.0
        
        # delays shape: (num_mics,)
        # distance = dot(mic_pos, u)
        return np.dot(self.mic_positions, np.array([ux, uy, uz])) / config.SPEED_OF_SOUND


class DelayAndSumBeamformer(BeamformerBase):
    def __init__(self, mic_positions=config.MIC_POSITIONS, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE):
        super().__init__(mic_positions, sample_rate, chunk_size)
        
        # Steering vector (weights)
        # Shape: (num_bins, num_mics)
        self.steering_vector = np.ones((self.num_bins, self.num_mics), dtype=np.complex64)
        
        # Initialize
        self.update_steering_vector(self.current_angle)

    def update_steering_vector(self, theta_deg):
        self.current_angle = theta_deg
        delays = self._calculate_delays(theta_deg)
        
        # Calculate phase shifts
        omega = 2 * np.pi * self.freqs
        
        # We want to align phases to the center.
        # Original logic: self.steering_vector = np.exp(1j * np.outer(omega, delays))
        # This adds a phase shift to cancel the delay.
        self.steering_vector = np.exp(1j * np.outer(omega, delays))
        
        # Normalize weights (1/M)
        self.steering_vector /= self.num_mics

    def apply(self, multichannel_chunk):
        # 1. FFT
        spectrum = np.fft.rfft(multichannel_chunk, axis=0)
        
        # 2. Apply weights
        # spectrum: (num_bins, num_mics)
        # steering_vector: (num_bins, num_mics)
        beamformed_spectrum = np.sum(spectrum * self.steering_vector, axis=1)
        
        # 3. IFFT
        beamformed_chunk = np.fft.irfft(beamformed_spectrum, n=self.chunk_size)
        
        return beamformed_chunk.astype(np.float32)


class MVDRBeamformer(BeamformerBase):
    def __init__(self, mic_positions=config.MIC_POSITIONS, sample_rate=config.SAMPLE_RATE, chunk_size=config.CHUNK_SIZE, alpha=0.95):
        super().__init__(mic_positions, sample_rate, chunk_size)
        self.alpha = alpha # Forgetting factor for recursive covariance update
        
        # Covariance matrices: (num_bins, num_mics, num_mics)
        # Initialize with Identity (uncorrelated noise assumption)
        self.R = np.zeros((self.num_bins, self.num_mics, self.num_mics), dtype=np.complex64)
        for k in range(self.num_bins):
            self.R[k] = np.eye(self.num_mics, dtype=np.complex64) * 1e-3 # Small init
            
        # Steering vector a(f, theta) - pure delay vector, NOT weights
        self.steering_vector_a = np.ones((self.num_bins, self.num_mics), dtype=np.complex64)
        
        self.update_steering_vector(self.current_angle)

    def update_steering_vector(self, theta_deg):
        self.current_angle = theta_deg
        delays = self._calculate_delays(theta_deg)
        omega = 2 * np.pi * self.freqs
        
        # This is the "steering vector" a(theta) in MVDR literature.
        # It represents the transfer function from source to mics.
        # x = a * s + n
        # Similar to DS, we model it as phase shifts.
        # Note: In DS code we used exp(1j * ...). 
        # Here a should match the physics.
        # If signal arrives at mic m with delay tau_m, X_m = S * exp(-j w tau_m).
        # So a_m = exp(-j w tau_m).
        # The DS weights were w = 1/M * a^H = 1/M * exp(+j w tau_m).
        # So here we want a to be exp(-j w tau_m).
        
        # Let's verify standard definition:
        # w_mvdr = (R^-1 a) / (a^H R^-1 a)
        # If R = I, w_mvdr = a / (a^H a) = a / M.
        # If we use w^H x as output? Or w^T x?
        # Usually y = w^H x = sum(conj(w_i) * x_i).
        # If we want y ~ s, and x = a s, then w^H a s = s  => w^H a = 1.
        # (a/M)^H a = a^H a / M = M/M = 1. Correct.
        
        # So if we apply weights as sum(w * x), that is w^T x (elementwise mult then sum without conj).
        # Then we need w_apply such that w_apply * a = 1.
        # If w_apply = conj(w_mvdr) ... this gets confusing with apply() implementation.
        
        # Let's stick to the implementation in apply():
        # output = sum(spectrum * weights)
        # This is effectively w^T x.
        # For DelayAndSum, weights = exp(+j w tau_m) / M.
        # And x = exp(-j w tau_m) * S.
        # sum(weights * x) = sum(1/M) * S = S. Correct.
        
        # So for MVDR, we want weights 'w' such that sum(w * x) = S.
        # If x = a * S, then sum(w * a) = w^T a = 1.
        # Linear constraint: w^T a = 1.
        # Minimize E[|y|^2] = E[|w^T x|^2] = w^T R w* (sort of)
        # Standard derivation usually uses y = w^H x.
        # Let's stick to y = w^H x convention for derivation, then convert.
        # y = w^H x. Constraint w^H a = 1.
        # Solution: w = (R^-1 a) / (a^H R^-1 a).
        # Our code does: y = sum(W_code * X). So W_code corresponds to conj(w).
        # So W_code = conj( (R^-1 a) / (a^H R^-1 a) )
        #           = (R^-1* a*) / (a^T R^-1* a*)  Assuming R is Hermitian symmetric?
        # R is E[x x^H]. R is Hermitian. R^-1 is Hermitian.
        # let's just compute w_mvdr (standard) and then use conjugated version for 'weights'.
        
        # 'a' vector formulation:
        # X = a * S.
        # If mic is delayed by tau, phase is -omega * tau.
        # So a = exp(-1j * omega * delays).
        
        self.steering_vector_a = np.exp(-1j * np.outer(omega, delays))

    def apply(self, multichannel_chunk):
        # 1. FFT
        # spectrum shape: (num_bins, num_mics)
        # This is our 'x' vector for each frequency.
        spectrum = np.fft.rfft(multichannel_chunk, axis=0)
        
        num_bins, num_mics = spectrum.shape
        
        # 2. Update Covariance Matrix R recursively
        # R[k] = alpha * R[k] + (1-alpha) * x[k] * x[k]^H
        # x[k] is (num_mics,)
        # We need outer product for each bin.
        
        # Vectorized update?
        # x: (B, M). x[:, :, None] * x[:, None, :].conj() -> (B, M, M)
        X = spectrum[:, :, np.newaxis] # (B, M, 1)
        X_H = spectrum[:, np.newaxis, :].conj() # (B, 1, M)
        
        # Instantaneous covariance
        R_inst = np.matmul(X, X_H) # (B, M, M)
        
        self.R = self.alpha * self.R + (1 - self.alpha) * R_inst
        
        # 3. Compute MVDR weights
        # w = (R^-1 a) / (a^H R^-1 a)
        # Invert R.
        # To ensure robust inversion, add small diagonal loading if needed, 
        # but with alpha < 1 and noise it should remain invertible.
        # For safety, let's load diagonal slightly.
        R_loaded = self.R + np.eye(num_mics, dtype=np.complex64)[np.newaxis, :, :] * 1e-6
        
        inv_R = np.linalg.inv(R_loaded) # (B, M, M)
        
        # a: (B, M)
        a = self.steering_vector_a # (B, M)
        a_vec = a[:, :, np.newaxis] # (B, M, 1)
        a_H = a[:, np.newaxis, :].conj() # (B, 1, M)
        
        # numerator = R^-1 a
        # (B, M, M) @ (B, M, 1) -> (B, M, 1)
        num = np.matmul(inv_R, a_vec) 
        
        # denominator = a^H R^-1 a
        # (B, 1, M) @ (B, M, 1) -> (B, 1, 1)
        denom = np.matmul(a_H, num)
        
        # Avoid division by zero (unlikely)
        denom = np.maximum(np.real(denom), 1e-12) + 1j * np.imag(denom)
        
        w_mvdr = num / denom # (B, M, 1)
        w_mvdr = w_mvdr.squeeze(axis=2) # (B, M)
        
        # NOTE: w_mvdr is designed for y = w^H x.
        # Our apply uses sum(weight * x) = weight^T x.
        # So we need weight = conj(w_mvdr).
        weights_for_code = np.conj(w_mvdr)
        
        # 4. Apply
        beamformed_spectrum = np.sum(spectrum * weights_for_code, axis=1)
        
        # 5. IFFT
        beamformed_chunk = np.fft.irfft(beamformed_spectrum, n=self.chunk_size)
        
        return beamformed_chunk.astype(np.float32)

