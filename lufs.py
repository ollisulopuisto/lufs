import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import os
import resampy
from multiprocessing import Pool, cpu_count, freeze_support
import subprocess
import argparse
from tqdm import tqdm

def install_dependencies():
    """
    Installs the required dependencies if they are not already installed.
    """
    try:
        import pyloudnorm
        import soundfile
        import resampy
        import tqdm
    except ImportError:
        print("Installing dependencies...")
        try:
            subprocess.check_call(["pip", "install", "pyloudnorm", "soundfile", "resampy", "numpy", "tqdm"])
            print("Dependencies installed.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing dependencies: {e}")
            print("It's recommended to create a virtual environment (venv) and install dependencies there.")
            print("You can create a venv with: python3 -m venv .venv")
            print("And activate it with: source .venv/bin/activate")

def normalize_audio(input_file, output_file, target_lufs=-16.0, true_peak=-1.0, lra_max=9.0, num_processes=max(1, cpu_count() - 1)):
    """
    Analyzes an audio file and adjusts its loudness to the target LUFS, considering true peak limits.
    
    Args:
        input_file (str): Path to the input audio file
        output_file (str): Path to save the normalized audio
        target_lufs (float): Target LUFS level (default: -16.0)
        true_peak (float): Maximum true peak level in dBTP (default: -1.0)
        lra_max (float): Maximum loudness range - not fully implemented (default: 9.0)
        num_processes (int): Number of processes for parallel processing
    
    Returns:
        str: Success or error message
    """
    print(f"Normalizing audio: {input_file} -> {output_file}")
    try:
        # Check if input file exists
        if not os.path.isfile(input_file):
            return f"Error: Input file {input_file} does not exist"
        
        # Load audio data
        data, rate = sf.read(input_file)
        
        # Get the subtype from the input file to preserve bit depth
        with sf.SoundFile(input_file, 'r') as f:
            subtype = f.subtype
        print(f"Original format: {subtype}, {rate} Hz")

        # Initialize meter and measure loudness
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        print(f"Input integrated loudness: {loudness:.2f} LUFS")

        # Calculate loudness difference
        loudness_diff = target_lufs - loudness
        print(f"Target: {target_lufs:.2f} LUFS, difference: {loudness_diff:.2f} dB")
        
        # Apply gain
        normalized_data = data * (10**(loudness_diff/20))

        # Apply true peak limiting if needed
        print(f"Checking true peak limit ({true_peak:.2f} dBTP)...")
        normalized_data = apply_true_peak_limiting(normalized_data, rate, true_peak, num_processes)
        
        # Ensure data stays within [-1, 1] range
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        # Re-measure after normalization
        normalized_loudness = meter.integrated_loudness(normalized_data)
        print(f"Output integrated loudness: {normalized_loudness:.2f} LUFS")

        # Write output audio
        sf.write(output_file, normalized_data, rate, subtype=subtype)
        print(f"Audio normalized and saved to {output_file}")
        return f"Successfully normalized {input_file} to {output_file}"

    except Exception as e:
        return f"Error processing {input_file}: {str(e)}"

def apply_true_peak_limiting(data, rate, true_peak_limit=-1.0, num_processes=max(1, cpu_count() - 1)):
    """
    Apply true peak limiting to audio data with progress bar and multiprocessing.
    
    Args:
        data (ndarray): Audio data
        rate (int): Sample rate
        true_peak_limit (float): Maximum true peak level in dBTP
        num_processes (int): Number of processes for parallel processing
    
    Returns:
        ndarray: Processed audio data
    """
    # Upsample for true peak measurement (4x oversampling)
    print("Measuring true peak (4x oversampling)...")
    oversampled_data = resampy.resample(data, rate, rate * 4)

    # Find the maximum peak
    true_peak = np.max(np.abs(oversampled_data))
    true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -120.0
    print(f"True peak: {true_peak_db:.2f} dBTP")

    # Apply gain reduction if needed
    if true_peak_db > true_peak_limit:
        gain_reduction = true_peak_limit - true_peak_db
        print(f"Applying gain reduction of {gain_reduction:.2f} dB to meet true peak limit...")

        # For very small files, don't use multiprocessing
        if len(data) < 100000 or num_processes <= 1:
            # Apply gain reduction directly with progress bar
            for i in tqdm(range(len(data)), desc="Applying gain reduction", unit="sample"):
                data[i] = data[i] * (10**(gain_reduction/20))
        else:
            # Split data into chunks for parallel processing
            num_chunks = min(num_processes, 16)  # Limit max chunks
            chunk_size = len(data) // num_chunks
            chunks = []
            
            # Create chunks, handling the case where len(data) is not evenly divisible by num_chunks
            for i in range(num_chunks):
                start = i * chunk_size
                end = start + chunk_size if i < num_chunks - 1 else len(data)
                chunks.append(data[start:end])
            
            # Apply gain reduction to each chunk in parallel
            print(f"Processing in parallel with {num_chunks} chunks...")
            with Pool(processes=num_processes) as pool:
                processed_chunks = list(tqdm(
                    pool.imap(apply_gain_reduction_chunk, [(chunk, gain_reduction) for chunk in chunks]), 
                    total=len(chunks), 
                    desc="Applying gain reduction"
                ))
            
            # Concatenate the processed chunks
            data = np.concatenate(processed_chunks)

    return data

def apply_gain_reduction_chunk(args):
    """
    Apply gain reduction to a chunk of audio data.
    
    Args:
        args (tuple): Tuple containing (chunk, gain_reduction)
    
    Returns:
        ndarray: Processed audio chunk
    """
    chunk, gain_reduction = args
    return chunk * (10**(gain_reduction/20))  # Vectorized operation for speed

def parallel_normalize_audio(input_files, output_files, target_lufs=-16.0, true_peak=-1.0, lra_max=9.0, num_processes=None):
    """
    Normalizes multiple audio files in parallel.
    
    Args:
        input_files (list): List of input audio file paths
        output_files (list): List of output audio file paths
        target_lufs (float): Target LUFS level
        true_peak (float): Maximum true peak level in dBTP
        lra_max (float): Maximum loudness range
        num_processes (int): Number of processes to use
    """
    if len(input_files) != len(output_files):
        print("Error: Number of input files must match number of output files")
        return

    # Use min(cpu_count, number of files) processes for file-level parallelism
    if num_processes is None:
        num_processes = min(cpu_count(), len(input_files))
    
    print(f"Using {num_processes} processes for parallel audio normalization")

    # Create pairs of input and output files with parameters
    input_output_pairs = [(input_file, output_file, target_lufs, true_peak, lra_max, max(1, num_processes // len(input_files)))
                          for input_file, output_file in zip(input_files, output_files)]

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # Use starmap to apply normalize_audio to each pair in parallel
        results = list(tqdm(
            pool.starmap(normalize_audio, input_output_pairs),
            total=len(input_output_pairs),
            desc="Processing files"
        ))

    # Print the results
    for result in results:
        print(result)

def main():
    """
    Main function to handle command line arguments and run the script.
    """
    parser = argparse.ArgumentParser(description="LUFS Audio Normalizer - Adjust audio files to target loudness")
    
    # Define argument groups for better organization
    file_group = parser.add_argument_group('File Options')
    file_group.add_argument("input_file", nargs='?', help="Input audio file path")
    file_group.add_argument("output_file", nargs='?', help="Output audio file path")
    file_group.add_argument("-b", "--batch", action="store_true", help="Process multiple files (provide space-separated input files followed by output files)")
    
    # Normalization settings
    norm_group = parser.add_argument_group('Normalization Settings')
    norm_group.add_argument("-t", "--target_lufs", type=float, default=-16.0, help="Target LUFS level (default: -16.0)")
    norm_group.add_argument("-p", "--true_peak", type=float, default=-1.0, help="Maximum true peak level (default: -1.0)")
    norm_group.add_argument("-l", "--lra_max", type=float, default=9.0, help="Maximum loudness range (default: 9.0)")
    
    # Performance settings
    perf_group = parser.add_argument_group('Performance Options')
    perf_group.add_argument("-n", "--num_processes", type=int, default=max(1, cpu_count() - 1),
                          help=f"Number of processes to use (default: {max(1, cpu_count() - 1)})")
    
    # Handle the remaining arguments as lists of input/output files
    args, remaining = parser.parse_known_args()
    
    # Process the arguments based on the mode (single file vs batch)
    if args.batch or not (args.input_file and args.output_file):
        # In batch mode or if not enough arguments for single file mode
        if not remaining or len(remaining) < 2:
            parser.print_help()
            print("\nError: Batch mode requires at least one input file and one output file")
            return
            
        # Split the remaining args into input and output files
        midpoint = len(remaining) // 2
        input_files = remaining[:midpoint]
        output_files = remaining[midpoint:]
        
        if len(input_files) != len(output_files):
            print("Error: Number of input files must match number of output files")
            print(f"Input files ({len(input_files)}): {input_files}")
            print(f"Output files ({len(output_files)}): {output_files}")
            return
            
        parallel_normalize_audio(input_files, output_files, 
                                args.target_lufs, args.true_peak, 
                                args.lra_max, args.num_processes)
    else:
        # Single file mode
        normalize_audio(args.input_file, args.output_file, 
                       args.target_lufs, args.true_peak, 
                       args.lra_max, args.num_processes)

if __name__ == "__main__":
    freeze_support()
    install_dependencies()
    main()
