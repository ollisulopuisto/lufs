# LUFS Audio Normalizer

A Python tool for normalizing audio files to specific LUFS (Loudness Units relative to Full Scale) levels with true peak limiting and parallel processing capabilities.

## Features

- LUFS normalization to target loudness levels
- True peak limiting to prevent digital clipping
- Parallel processing for handling multiple audio files efficiently
- Automatic dependency installation
- Support for various audio file formats via SoundFile

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer)

### Setup

1. Clone this repository:

````
git clone https://github.com/ollisulopuisto/lufs.git cd lufs
```

2. (Optional but recommended) Create a virtual environment:

```
python3 -m venv .venv source .venv/bin/activate # On Windows: .venv\Scripts\activate3
```

3. Install dependencies:

````
pip install pyloudnorm soundfile numpy resampy
```


Note: The script will attempt to install dependencies automatically if they're missing.

## Usage

### Basic Usage
```python
from lufs import normalize_audio

# Normalize a single audio file
normalize_audio("input.wav", "output.wav", target_lufs=-16.0, true_peak=-1.0)

## Processing Multiple Files in Parallel

```
from lufs import parallel_normalize_audio

input_files = ["input1.wav", "input2.wav", "input3.wav"]
output_files = ["output1.wav", "output2.wav", "output3.wav"]

parallel_normalize_audio(input_files, output_files)
```

## Command Line Usage

```
python lufs.py
```

## Parameters

- target_lufs: Target integrated loudness level in LUFS (default: -16.0 LUFS)
- true_peak: Maximum allowed true peak level in dBTP (default: -1.0 dBTP)
- lra_max: Maximum loudness range in LU (default: 9.0 LU) - Note: Currently not fully implemented

## Technical Background

### LUFS (Loudness Units relative to Full Scale)

LUFS is an absolute loudness measurement standardized in ITU-R BS.1770. Streaming platforms and broadcast standards have specific LUFS targets:

- Spotify: -14 LUFS
- YouTube: -14 LUFS
- Apple Music: -16 LUFS
- AES streaming: -16 to -20 LUFS
- Broadcast (EBU R128): -23 LUFS

### True Peak

True peak measures the maximum signal level of an audio waveform accounting for inter-sample peaks, ensuring no clipping occurs during D/A conversion.

### LRA (Loudness Range)

LRA measures the variation of loudness in an audio file. Lower values indicate more consistent loudness (like in pop music), while higher values indicate more dynamic content (like classical music).

## Limitations and Future Improvements

- LRA adjustment is currently not fully implemented
- More sophisticated dynamic processing could be added
- Support for batch processing directories
- Command-line interface with arguments