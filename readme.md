# LUFS Audio Normalizer

A Python tool for normalizing audio files to specific LUFS (Loudness Units relative to Full Scale) levels with true peak limiting and parallel processing capabilities.

## Features

- LUFS normalization to target loudness levels
- True peak limiting to prevent digital clipping
- Parallel processing for handling multiple audio files efficiently
- Dynamic range adjustment (LRA)
- Memory-efficient streaming architecture for processing large files
- Caching system for faster repeated processing
- Automatic dependency installation
- Support for various audio file formats via SoundFile or FFmpeg

## Installation

### Prerequisites
- Python 3.6 or higher
- pip (Python package installer) or Conda package manager

### Setup with pip

1. Clone this repository:

```
git clone https://github.com/ollisulopuisto/lufs.git cd lufs
```

2. (Optional but recommended) Create a virtual environment:

```
python3 -m venv .venv source .venv/bin/activate # On Unix/MacOS
```

On Windows: 
```
.venv\Scripts\activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

### Setup with Conda

1. Clone this repository:

```
git clone https://github.com/ollisulopuisto/lufs.git cd lufs
```

2. Create a Conda environment and install dependencies:

```
conda create -n lufs python=3.9 conda activate lufs conda install numpy scipy pip install pyloudnorm soundfile resampy tqdm
```

Note: Some packages like `pyloudnorm` may not be available in Conda channels, so pip is used for those.

## Usage

### Command Line Usage

Process a single file:

```bash
python lufs.py input.wav output.wav -t -16.0 -p -1.0 -l 9.0
```

Process multiple files in batch mode:

```
python lufs.py -b input1.wav input2.wav output1.wav output2.wav -t -16.0 -p -1.0
```

## Command Line Options


File Options:
```
  input_file            Input audio file path
  output_file           Output audio file path
  -b, --batch           Process multiple files (provide space-separated input files followed by output files)
```
Normalization Settings:
```
  -t, --target_lufs     Target LUFS level (default: -16.0)
  -p, --true_peak       Maximum true peak level (default: -1.0)
  -l, --lra_max         Maximum loudness range (default: 9.0)
```  

Performance Options:
```
  -n, --num_processes   Number of processes to use (default: CPU count - 1)
  -c, --chunk_size      Size of processing chunks in seconds (default: 5.0)
  --no-cache            Disable caching of loudness analysis results
```  

## Using as a Module

```
from lufs import normalize_audio, parallel_normalize_audio, process_audio_streaming

# Basic normalization
normalize_audio("input.wav", "output.wav", target_lufs=-16.0, true_peak_limit=-1.0, lra_max=9.0)

# Processing multiple files in parallel
input_files = ["input1.wav", "input2.wav", "input3.wav"]
output_files = ["output1.wav", "output2.wav", "output3.wav"]
parallel_normalize_audio(input_files, output_files, target_lufs=-16.0, true_peak=-1.0)

# Memory-efficient streaming processing
process_audio_streaming("input.wav", "output.wav", target_lufs=-16.0, true_peak_limit=-1.0,
                       lra_max=9.0, num_processes=4, chunk_size=5.0, use_cache=True)
```

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

## LRA (Loudness Range)

LRA measures the variation of loudness in an audio file. Lower values indicate more consistent loudness (like in pop music), while higher values indicate more dynamic content (like classical music).

## Performance Optimization

The script includes several optimizations for better performance:

- Caching system: Analysis results are stored in a .lufs_cache directory to speed up repeated processing
- Streaming architecture: Processes files in chunks to minimize memory usage
- Parallel processing: Uses multiple CPU cores for faster processing
- Optimized for Apple Silicon: Detects M1/M2 chips and uses Accelerate framework

## Advanced Features
- FFmpeg integration: Can automatically convert unsupported audio formats
- Brickwall limiting: Implements a true peak limiter that preserves overall loudness
- Streaming I/O: Processes audio in chunks for minimal memory usage