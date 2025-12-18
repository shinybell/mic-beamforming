import numpy as np

# Sampling Rate
SAMPLE_RATE = 10000

# Chunk Size
CHUNK_SIZE = 2048

# Microphone Channels (DAQ)
# Ensure these match your actual hardware connection
MIC_CHANNELS = [
    "Dev10/ai0",
    "Dev10/ai1"
]

# Speed of Sound (m/s)
SPEED_OF_SOUND = 343.0

# Microphone Positions (x, y, z) in meters
# 2 Microphones, 76cm (0.76m) spacing
# Center is at (0, 0, 0)
# Mic 0: -38cm (-0.38m)
# Mic 1: +38cm (+0.38m)
MIC_POSITIONS = np.array([
    [-0.38, 0.0, 0.0],
    [ 0.38, 0.0, 0.0]
])
