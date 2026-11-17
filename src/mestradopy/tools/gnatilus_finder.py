"""
g.Nautilus Device Example - Real-time EEG Acquisition and Processing

This example demonstrates how to connect to and process real-time EEG data from
a g.Nautilus amplifier system. It showcases EEG signal
processing with standard filtering techniques commonly used in clinical and
research BCI applications.

This example demonstrates how to connect to and process real-time EEG data from
a g.Nautilus amplifier system. It showcases EEG signal processing with
standard filtering techniques commonly used in clinical and research BCI
applications.

What this example shows:
- Real-time data acquisition from g.Nautilus hardware
- Bandpass filtering for EEG frequency band selection
- Power line interference removal with dual notch filters
- Real-time visualization of clean EEG signals
- Hardware integration with g.Pype framework

Hardware requirements:
- g.Nautilus EEG amplifier system

Expected behavior:
When you run this example:
- Connects to g.Nautilus amplifier automatically
- Displays real-time EEG from 8 channels
- Shows clean, filtered signals suitable for analysis
- Amplitude range: ±50 µV (typical EEG range)
- Time window: 10 seconds of continuous data
- Real-time updates at amplifier sampling rate

Signal processing pipeline:
1. Raw EEG acquisition (8 channels, 250 Hz)
2. Bandpass filtering (1-30 Hz) - standard EEG band
3. 50 Hz notch filter - removes European power line noise
4. 60 Hz notch filter - removes American power line noise
5. Real-time visualization

Real-world applications:
- Clinical EEG monitoring and diagnosis
- BCI system development and testing
- Neurofeedback training applications
- Cognitive state monitoring research
- Sleep study and analysis
- Seizure detection systems
- Attention and meditation training

Usage:
    1. Mount g.Nautilus cap/electrodes
    2. Power on g.Nautilus
    4. Run: python example_devices_gnautilus.py
    5. Monitor real-time EEG signals

Note:
    This example provides the foundation for all BCI applications
    requiring real-time EEG data acquisition and processing.
"""

import os

# Substitua pelo caminho do gNEEDaccess no seu sistema
gds_path = r"C:\Program Files\gtec\gNEEDaccess" 

# Adiciona ao PATH do processo atual
os.environ["PATH"] = gds_path + os.pathsep + os.environ.get("PATH", "")

import gpype as gp
app = gp.MainApp()
p = gp.Pipeline()
source = gp.GNautilus(sampling_rate=250, channel_count=32)
scope = gp.TimeSeriesScope(amplitude_limit=50, time_window=10)
p.connect(source, scope)
app.add_widget(scope)
p.start()
app.run()
p.stop()