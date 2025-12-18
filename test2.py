import nidaqmx
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 44100
CHUNK_SIZE = 2048

# PCのスピーカー出力を準備
stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1)
stream.start()

with nidaqmx.Task() as task:
    task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
    task.timing.cfg_samp_clk_timing(SAMPLE_RATE, samps_per_chan=CHUNK_SIZE * 10)

    print("PCスピーカーから出力中...")
    try:
        while True:
            data = task.read(number_of_samples_per_channel=CHUNK_SIZE)
            # numpy配列に変換してPCのスピーカーへ
            stream.write(np.array(data, dtype=np.float32))
    except KeyboardInterrupt:
        pass

stream.stop()
