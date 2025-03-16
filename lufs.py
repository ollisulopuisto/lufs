import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import os
import resampy
from multiprocessing import Pool, cpu_count, freeze_support
import subprocess
import argparse

def install_dependencies():
    """
    Installs the required dependencies if they are not already installed.
    """
    try:
        import pyloudnorm
        import soundfile
        import resampy
    except ImportError:
        print("Installing dependencies...")
        try:
            subprocess.check_call(["pip", "install", "pyloudnorm", "soundfile", "resampy", "numpy"])
            print("Dependencies installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("It's recommended to create a virtual environment (venv) and install dependencies there.")
            print("You can create a venv with: python3 -m venv .venv")
            print("And activate it with: source .venv/bin/activate")

install_dependencies()


def normalize_audio(input_file, output_file, target_lufs=-16.0, true_peak=-1.0, lra_max=9.0):
    """
    Analyzes an audio file and adjusts its loudness to the target LUFS, considering short-term LUFS, true peak, and LRA.
    """
    print(f"Normalizing audio: {input_file} -> {output_file}")
    try:
        # Load audio data
        print(f"Loading audio data from {input_file}...")
        data, rate = sf.read(input_file)
        print("Audio data loaded.")

        # Get the subtype from the input file
        with sf.SoundFile(input_file, 'r') as f:
            subtype = f.subtype
        print(f"Original subtype: {subtype}")

        # Initialize meter
        print("Initializing loudness meter...")
        meter = pyln.Meter(rate)
        print("Loudness meter initialized.")

        # Measure integrated loudness
        print("Measuring integrated loudness...")
        loudness = meter.integrated_loudness(data)
        print(f"Input integrated loudness: {loudness} LUFS")

        # Calculate loudness difference
        print("Calculating loudness difference...")
        loudness_diff = target_lufs - loudness
        print(f"Loudness difference: {loudness_diff} LUFS")
        
        # Apply gain
        print("Applying gain...")
        normalized_data = data * (10**(loudness_diff/20))

        # Apply true peak limiting
        print("Applying true peak limiting...")
        normalized_data = apply_true_peak_limiting(normalized_data, rate, true_peak)
        print("True peak limiting applied.")

        # Ensure data stays within [-1, 1] range to prevent clipping
        print("Clipping normalized data to [-1, 1]...")
        normalized_data = np.clip(normalized_data, -1.0, 1.0)
        print("Data clipped.")

        # Re-measure after normalization (optional, for verification)        
        print("Measuring integrated loudness after normalization...")
        normalized_loudness = meter.integrated_loudness(normalized_data)
        print(f"Output integrated loudness: {normalized_loudness} LUFS")

        # Write output audio
        print(f"Writing normalized audio to {output_file}...")
        sf.write(output_file, normalized_data, rate, subtype=subtype)
        print(f"Audio normalized and saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")

def apply_true_peak_limiting(data, rate, true_peak_limit=-1.0):
    """Apply true peak limiting to audio data with progress bar."""
    import resampy
    from tqdm import tqdm
    
    with tqdm(total=3, desc="Applying true peak limiting", unit="step") as pbar:
        # Upsample for true peak measurement (typically 4x)
        oversampled_data = resampy.resample(data, rate, rate * 4)
        pbar.update(1)
        
        # Find the maximum peak
        true_peak = np.max(np.abs(oversampled_data))
        true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -np.inf
        pbar.update(1)
        
        # Apply gain reduction if needed
        if true_peak_db > true_peak_limit:
            gain_reduction = true_peak_limit - true_peak_db
            
            # Apply gain reduction with progress bar
            for i in tqdm(range(len(data)), desc="Applying gain reduction"):
                data[i] = data[i] * (10**(gain_reduction/20))
        pbar.update(1)
    
    return data

def process_audio_files(input_output_pairs):
    """
    Processes a list of input-output file pairs using the normalize_audio function.

    Args:
        input_output_pairs (list of tuples): A list where each tuple contains the input and output file paths.
    """
    for input_file, output_file in input_output_pairs:
        normalize_audio(input_file, output_file)

def parallel_normalize_audio(input_files, output_files):
    """
    Normalizes audio files in parallel using multiprocessing.

    Args:
        input_files (list of str): List of input audio file paths.
        output_files (list of str): List of output audio file paths.
    """
    # Create pairs of input and output files
    input_output_pairs = list(zip(input_files, output_files))

    # Determine the number of processes to use (up to the number of CPU cores)
    num_processes = min(cpu_count(), len(input_files))
    print(f"Using {num_processes} processes for parallel audio normalization.")

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Wrap process_audio_files with a progress tracker
        results = pool.imap_unordered(process_wrapper, [(i, input_output_pairs[i::num_processes]) for i in range(num_processes)])
        for result in results:
            print(result)

def process_wrapper(args):
    """
    Wrapper function for process_audio_files to track progress.

    Args:
        args (tuple): A tuple containing the process ID and a list of input-output file pairs.
    """
    process_id, input_output_pairs = args
    for input_file, output_file in input_output_pairs:
        print(f"Process {process_id}: Normalizing {input_file} to {output_file}")
        normalize_audio(input_file, output_file)
        print(f"Process {process_id}: Finished normalizing {input_file} to {output_file}")
    return f"Process {process_id}: Completed its tasks."

# Create test file with a sine wave
def create_test_file(file_path, duration=5, sample_rate=44100):
    """Create a test audio file with a 440 Hz sine wave"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    data = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave at -6 dBFS
    data = data.reshape(-1, 1)  # Convert to mono format
    sf.write(file_path, data, sample_rate)
    print(f"Created test audio file: {file_path}")

def main():
    parser = argparse.ArgumentParser(description="Normalize audio files to a target LUFS level.")
    parser.add_argument("input_file", help="Path to the input audio file.")
    parser.add_argument("output_file", help="Path to the output audio file.")
    parser.add_argument("-t", "--target_lufs", type=float, default=-16.0, help="Target LUFS level (default: -16.0).")
    parser.add_argument("-p", "--true_peak", type=float, default=-1.0, help="Maximum true peak level (default: -1.0).")
    parser.add_argument("-l", "--lra_max", type=float, default=9.0, help="Maximum loudness range (default: 9.0).")

    args = parser.parse_args()

    normalize_audio(args.input_file, args.output_file, args.target_lufs, args.true_peak, args.lra_max)

if __name__ == "__main__":
    freeze_support()
    install_dependencies()
    main()
