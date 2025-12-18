"""
データを連続収集してグラフ表示
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import nidaqmx as ni
from nidaqmx.constants import AcquisitionType, TerminalConfiguration


# Configure NI DAQmx settings
DAQ_IO_ai_0 = 'Dev10/ai0'
DAQ_IO_ai_1 = 'Dev10/ai1'

sampling_rate = 5000
samples = 500

fig, ax = plt.subplots()

with ni.Task() as task:
    task.ai_channels.add_ai_voltage_chan(DAQ_IO_ai_0, 
                                            min_val=-10, 
                                            max_val=10,
                                            terminal_config=TerminalConfiguration.DIFF) # 収集モード：差動
                                            # terminal_config=TerminalConfiguration.RSE) # 収集モード：RES
    
    task.ai_channels.add_ai_voltage_chan(DAQ_IO_ai_1, 
                                            min_val=-10, 
                                            max_val=10,
                                            terminal_config=TerminalConfiguration.DIFF) # 収集モード：差動
                                            # terminal_config=TerminalConfiguration.RSE) # 収集モード：RES

    task.timing.cfg_samp_clk_timing(rate=sampling_rate, 
                                       sample_mode=AcquisitionType.CONTINUOUS)
                                        # sample_mode=AcquisitionType.FINITE # バッファーのメモリが足りないときはこちらの方が良い時がある

    
    try:
        while True:
            new_data = task.read(samples)
            data1 = new_data[0]
            data2 = new_data[1]

            ax.cla()
            ax.plot(data1,'ro-')
            ax.plot(data2,'ro-')

            ax.grid()
            plt.pause(0.01)
            
        
    except KeyboardInterrupt: # Ctrl + C
        task.stop()
        plt.ioff()  # Turn off interactive mode
        print('Stop')
    
