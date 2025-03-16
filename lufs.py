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

        # Measure initial loudness and true peak
        meter = pyln.Meter(rate)
        loudness = meter.integrated_loudness(data)
        initial_true_peak = measure_true_peak(data, rate)
        print(f"Input integrated loudness: {loudness:.2f} LUFS")
        print(f"Input true peak: {initial_true_peak:.2f} dBTP")
        
        # IMPORTANT CHANGE: First apply brickwall limiting if the true peak is already over limit
        # This preserves the average loudness while bringing peaks under control
        if initial_true_peak > true_peak:
            print(f"Input true peak {initial_true_peak:.2f} dBTP exceeds limit of {true_peak:.2f} dBTP")
            print(f"Applying brickwall limiting before normalization...")
            data = apply_brickwall_limiter(data, rate, true_peak, 0.050, num_processes)
            
            # Re-measure after brickwall limiting
            loudness = meter.integrated_loudness(data)
            initial_true_peak = measure_true_peak(data, rate)
            print(f"After brickwall limiting: loudness {loudness:.2f} LUFS, true peak {initial_true_peak:.2f} dBTP")
        
        # Calculate the maximum gain we can apply while respecting the true peak limit
        headroom = true_peak - initial_true_peak
        loudness_diff = target_lufs - loudness
        
        # If we need to increase gain but are limited by true peak
        if loudness_diff > 0 and headroom < loudness_diff:
            print(f"True peak limiting restricts gain - using {headroom:.2f} dB instead of {loudness_diff:.2f} dB")
            applied_gain = headroom
        else:
            applied_gain = loudness_diff
        
        # Apply gain for LUFS normalization - FIX HERE! Use applied_gain instead of loudness_diff
        normalized_data = data * (10**(applied_gain/20))
        
        # Apply brickwall limiting if needed (for safety, in case true peak still exceeds limit)
        final_true_peak = measure_true_peak(normalized_data, rate)
        if final_true_peak > true_peak:
            print(f"Applying brickwall limiting to bring {final_true_peak:.2f} dBTP under {true_peak:.2f} dBTP threshold...")
            normalized_data = apply_brickwall_limiter(normalized_data, rate, true_peak, 0.050, num_processes)
            
            # After brickwall limiting, check if we can add more gain to get closer to target LUFS
            limited_loudness = meter.integrated_loudness(normalized_data)
            limited_true_peak = measure_true_peak(normalized_data, rate)
            remaining_headroom = true_peak - limited_true_peak
            loudness_gap = target_lufs - limited_loudness
            
            # If we have headroom and loudness is below target, apply additional gain
            if remaining_headroom > 0.2 and loudness_gap > 0.5:  # Add small buffers for safety
                additional_gain = min(remaining_headroom - 0.1, loudness_gap)
                print(f"After limiting: loudness {limited_loudness:.2f} LUFS, true peak {limited_true_peak:.2f} dBTP")
                print(f"Applying additional gain of {additional_gain:.2f} dB to get closer to target")
                normalized_data = normalized_data * (10**(additional_gain/20))
        
        # Re-measure final loudness
        normalized_loudness = meter.integrated_loudness(normalized_data)
        print(f"Output integrated loudness: {normalized_loudness:.2f} LUFS")

        # Ensure data stays within [-1, 1] range
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        # Write output audio
        sf.write(output_file, normalized_data, rate, subtype=subtype)
        print(f"Audio normalized and saved to {output_file}")
        return f"Successfully normalized {input_file} to {output_file}"

    except Exception as e:
        return f"Error processing {input_file}: {str(e)}"

def check_true_peak(data, rate, true_peak_limit=-1.0, num_processes=max(1, cpu_count() - 1)):
    """
    Check if true peak exceeds limit and apply limiting if needed.
    
    Args:
        data (ndarray): Audio data
        rate (int): Sample rate
        true_peak_limit (float): Maximum true peak level in dBTP
        num_processes (int): Number of processes for parallel processing
    
    Returns:
        ndarray: Processed audio data
    """
    true_peak_db = measure_true_peak(data, rate)
    print(f"Original true peak: {true_peak_db:.2f} dBTP")
    
    if true_peak_db > true_peak_limit:
        return apply_true_peak_limiting(data, rate, true_peak_limit, num_processes)
    else:
        return data

def measure_true_peak(data, rate):
    """
    Measure the true peak level of audio data.
    
    Args:
        data (ndarray): Audio data
        rate (int): Sample rate
    
    Returns:
        float: True peak level in dBTP
    """
    # Upsample for true peak measurement (4x oversampling)
    oversampled_data = resampy.resample(data, rate, rate * 4)
    
    # Find the maximum peak
    true_peak = np.max(np.abs(oversampled_data))
    true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -120.0
    
    return true_peak_db

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

def apply_brickwall_limiter(data, rate, true_peak_limit=-1.0, release_time=0.050, num_processes=max(1, cpu_count() - 1)):
    """
    Apply a brickwall limiter that only affects peaks exceeding the threshold.
    
    Args:
        data (ndarray): Audio data
        rate (int): Sample rate
        true_peak_limit (float): Maximum true peak level in dBTP
        release_time (float): Release time in seconds
        num_processes (int): Number of processes for parallel processing
    
    Returns:
        ndarray: Limited audio data
    """
    from scipy import signal
    import numpy as np
    from tqdm import tqdm
    
    # Convert true peak limit from dB to linear
    threshold_linear = 10 ** (true_peak_limit / 20.0)
    
    # Upsample for true peak detection (4x oversampling)
    oversampled_data = resampy.resample(data, rate, rate * 4)
    oversampled_rate = rate * 4
    
    # Calculate the gain reduction needed at each sample
    abs_data = np.abs(oversampled_data)
    gain_reduction = np.ones_like(abs_data)
    
    # Calculate gain reduction only where the signal exceeds the threshold
    mask = abs_data > threshold_linear
    gain_reduction[mask] = threshold_linear / abs_data[mask]
    
    # Create a smoothing filter for the gain reduction (release time)
    release_samples = int(release_time * oversampled_rate)
    if release_samples > 0:
        # Create exponential release curve
        release_curve = np.exp(-np.arange(release_samples) / (release_samples / 5))
        release_curve = release_curve / np.sum(release_curve)  # Normalize
        
        # Apply the smoothing only to the gain reduction, not the original signal
        # We want to look-ahead, so we reverse, filter, then reverse again
        for channel in range(gain_reduction.shape[1] if len(gain_reduction.shape) > 1 else 1):
            if len(gain_reduction.shape) > 1:
                # Multi-channel
                gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[:, channel][::-1])[::-1]
                gain_reduction[:, channel] = np.minimum(gain_reduction[:, channel], gain_reduction_smooth)
            else:
                # Mono
                gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
                gain_reduction = np.minimum(gain_reduction, gain_reduction_smooth)
    
    # Apply the smoothed gain reduction to the oversampled audio
    limited_oversampled = oversampled_data * gain_reduction
    
    # Downsample back to original rate
    limited_data = resampy.resample(limited_oversampled, oversampled_rate, rate)
    
    return limited_data

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
