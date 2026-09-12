import pyloudnorm as pyln
import soundfile as sf
import numpy as np
import os
import resampy
from multiprocessing import Pool, cpu_count, freeze_support
import subprocess
import argparse
from tqdm import tqdm
import platform

# Define the optimization functions first
def enable_simd_optimizations(verbose=False):
    """Enable CPU SIMD vectorization optimizations"""
    import os
    
    # Use AVX2/SSE on Intel or NEON on ARM
    os.environ['NPY_ENABLE_AVX2'] = '1'
    os.environ['NPY_ENABLE_SSE41'] = '1'
    os.environ['NPY_ENABLE_SSE42'] = '1'
    
    # Set numpy threading options
    os.environ['NPY_NUM_THREADS'] = str(max(1, cpu_count() - 1))
    
    # Only show numpy config details if verbose mode is enabled
    if verbose:
        try:
            import numpy as np
            np.__config__.show()
        except:
            pass

def enable_optimizations(verbose=False):
    """Enable all available optimizations at startup"""
    # SIMD vectorization
    enable_simd_optimizations(verbose)
    
    # Apple Silicon specific optimizations
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        os.environ['ACCELERATE'] = '1'  # Use Accelerate framework
        os.environ['OMP_NUM_THREADS'] = str(max(1, cpu_count() - 1))
        os.environ['MKL_NUM_THREADS'] = str(max(1, cpu_count() - 1))
        
        # Try to enable Metal for PyTorch if available
        try:
            import torch
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                if verbose:
                    print("✓ Metal Performance Shaders enabled")
        except ImportError:
            pass
    
    return True

# Apply Apple Silicon optimizations if available
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    import os
    os.environ['ACCELERATE'] = '1'  # Use Accelerate framework

# Apply optimizations immediately
enable_optimizations()

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

def normalize_audio(input_file, output_file, target_lufs=-16.0, true_peak_limit=-1.0, lra_max=9.0, num_processes=max(1, cpu_count() - 1)):
    """
    Analyzes and normalizes audio following user's preferred workflow:
    1. Measure current loudness
    2. Apply dynamic range processing (LRA) if needed
    3. Apply LUFS normalization to target
    4. Check true peak and apply brickwall limiting if needed
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

        # Initialize meter and measure initial loudness
        meter = pyln.Meter(rate)
        initial_loudness = meter.integrated_loudness(data)
        print(f"Input integrated loudness: {initial_loudness:.2f} LUFS")

        # STEP 1: Apply LRA adjustment if needed
        if lra_max > 0:
            from scipy import signal
            
            # Measure current LRA (simplified approximation)
            # This is a basic approximation - a real implementation would be more sophisticated
            window_size = int(3 * rate)  # 3-second window
            hop_size = int(0.1 * rate)   # 100ms hop
            st_loudness = []
            
            # Process in windows for large files, or all at once for small files
            if len(data) > window_size * 10:
                # For larger files, process in windows
                for i in range(0, len(data) - window_size, hop_size):
                    window_data = data[i:i + window_size]
                    window_loudness = meter.integrated_loudness(window_data)
                    if window_loudness > -70:  # Ignore silence
                        st_loudness.append(window_loudness)
            else:
                # For smaller files, just use the whole file
                st_loudness.append(initial_loudness)
                
            if st_loudness:
                st_loudness.sort()
                if len(st_loudness) >= 10:
                    # Calculate approximate LRA from distribution
                    p10_idx = max(0, int(len(st_loudness) * 0.1))
                    p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
                    current_lra = st_loudness[p95_idx] - st_loudness[p10_idx]
                    print(f"Estimated loudness range: {current_lra:.2f} LU")
                    
                    if current_lra > lra_max:
                        print(f"Applying compression to reduce LRA from {current_lra:.2f} to {lra_max:.2f} LU")
                        # Apply multi-stage compression
                        data = apply_multi_stage_compression_parallel(data, rate, current_lra, lra_max, num_processes)
                        
                        # Re-measure loudness after compression
                        compressed_loudness = meter.integrated_loudness(data)
                        print(f"Loudness after LRA adjustment: {compressed_loudness:.2f} LUFS")
        
        # STEP 2: Calculate and apply gain for LUFS normalization
        loudness = meter.integrated_loudness(data)
        loudness_diff = target_lufs - loudness
        print(f"Applying gain adjustment of {loudness_diff:.2f} dB to reach target LUFS")
        normalized_data = data * (10**(loudness_diff/20))
        
        # STEP 3: Check and apply true peak limiting only if needed
        true_peak = measure_true_peak_efficient(normalized_data, rate)
        print(f"True peak after normalization: {true_peak:.2f} dBTP")
        
        if true_peak > true_peak_limit:
            print(f"Applying brickwall limiter to bring {true_peak:.2f} dBTP under {true_peak_limit:.2f} dBTP threshold")
            # Use the brickwall limiter that only affects peaks (not overall gain)
            normalized_data = apply_brickwall_limiter(normalized_data, rate, true_peak_limit)
            
            # Verify final true peak
            final_true_peak = measure_true_peak_efficient(normalized_data, rate)
            print(f"Final true peak: {final_true_peak:.2f} dBTP")
        
        # Final loudness measurement
        final_loudness = meter.integrated_loudness(normalized_data)
        print(f"Final integrated loudness: {final_loudness:.2f} LUFS")
        
        # Ensure data stays within [-1, 1] range
        normalized_data = np.clip(normalized_data, -1.0, 1.0)

        # Write output audio
        sf.write(output_file, normalized_data, rate, subtype=subtype)
        print(f"Audio normalized and saved to {output_file}")
        return f"Successfully normalized {input_file} to {output_file}"

    except Exception as e:
        import traceback
        print(f"Error processing {input_file}: {str(e)}")
        traceback.print_exc()
        return f"Error processing {input_file}: {str(e)}"

def measure_true_peak_efficient(data, rate):
    """
    More efficient true peak measurement.
    """
    # For very short files, upsample entirely
    if len(data) < 500000:
        oversampled_data = optimized_resample(data, rate, rate * 4)  # Use optimized version
        true_peak = np.max(np.abs(oversampled_data))
    else:
        # For longer files, process in chunks
        chunk_size = 500000
        max_peak = 0
        
        for i in range(0, len(data), chunk_size):
            chunk = data[i:min(i + chunk_size, len(data))]
            # Add a bit of overlap to handle edge cases
            if i > 0:
                chunk = np.concatenate([data[max(0, i-100):i], chunk])
            
            oversampled_chunk = optimized_resample(chunk, rate, rate * 4)  # Use optimized version
            chunk_peak = np.max(np.abs(oversampled_chunk))
            max_peak = max(max_peak, chunk_peak)
        
        true_peak = max_peak
    
    true_peak_db = 20 * np.log10(true_peak) if true_peak > 0 else -120.0
    return true_peak_db

def apply_efficient_limiter(data, rate, true_peak_limit):
    """
    Simple but effective limiter implementation.
    """
    from tqdm import tqdm
    import time
    
    print("Starting limiter processing...")
    
    # First measure the true peak again to confirm
    true_peak = measure_true_peak_efficient(data, rate)
    print(f"Confirming true peak: {true_peak:.2f} dBTP")
    
    if true_peak <= true_peak_limit:
        print("No limiting needed, true peak already under threshold")
        return data
    
    # Calculate required gain reduction
    gain_reduction_db = true_peak_limit - true_peak
    gain_factor = 10 ** (gain_reduction_db / 20.0)
    print(f"Applying gain reduction of {gain_reduction_db:.2f} dB")
    
    # Apply gain reduction to entire signal (simple but effective approach)
    with tqdm(total=1, desc="Applying gain reduction", unit="file") as pbar:
        result = data * gain_factor
        time.sleep(0.1)  # Small delay to show progress bar
        pbar.update(1)
    
    # Verify result
    final_peak = measure_true_peak_efficient(result, rate)
    print(f"Final true peak after limiting: {final_peak:.2f} dBTP")
    
    # If the result is still above the limit (due to rounding errors), clip it
    if final_peak > true_peak_limit:
        print(f"Applying hard clipping to ensure true peak limit")
        result = np.clip(result, -0.98, 0.98)  # Slightly below 1.0 for safety
    
    return result

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

def apply_limiter_streaming(input_file, output_file, true_peak_limit, release_time, chunk_seconds=5.0):
    """Apply true peak limiting in a streaming fashion with proper oversampling while preserving stereo balance"""
    from scipy import signal
    
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Add safety margin to ensure we actually hit the target
            working_threshold = true_peak_limit - 0.1
            threshold_linear = 10 ** (working_threshold / 20.0)
            release_samples = int(release_time * rate)
            
            # Create release curve
            release_curve = np.exp(-np.arange(release_samples) / (release_samples / 5))
            release_curve = release_curve / np.sum(release_curve)
            
            # Process in overlapping chunks for smooth transitions
            chunk_size = int(rate * chunk_seconds)
            overlap_size = max(release_samples * 2, int(rate * 0.2))
            
            with tqdm(total=infile.frames, desc="Applying limiting", unit="samples") as pbar:
                pos = 0
                while pos < infile.frames:
                    # Position file pointer
                    infile.seek(pos)
                    
                    # Read chunk with overlap
                    read_size = min(chunk_size + overlap_size, infile.frames - pos)
                    chunk = infile.read(read_size)
                    if len(chunk) == 0:
                        break
                    
                    # Upsample for true peak detection and limiting
                    oversampling_factor = 4
                    oversampled_chunk = optimized_resample(chunk, rate, rate * oversampling_factor)
                    
                    # Calculate gain reduction based on oversampled signal
                    # IMPORTANT CHANGE: Calculate a single gain reduction curve based on the maximum 
                    # absolute value across all channels at each sample point
                    if len(oversampled_chunk.shape) > 1:  # Multi-channel audio
                        # Find the maximum absolute value across all channels
                        max_abs_data = np.max(np.abs(oversampled_chunk), axis=1)
                        gain_reduction = np.ones_like(max_abs_data)
                        mask = max_abs_data > threshold_linear
                        if np.any(mask):
                            gain_reduction[mask] = threshold_linear / max_abs_data[mask]
                        
                        # Apply smoothing filter to gain reduction
                        gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
                        gain_reduction = np.minimum(gain_reduction, gain_reduction_smooth)
                        
                        # Expand gain reduction to apply to all channels
                        gain_reduction_expanded = np.tile(gain_reduction[:, np.newaxis], (1, channels))
                        
                        # Apply the same gain reduction to all channels
                        limited_oversampled = oversampled_chunk * gain_reduction_expanded
                    else:  # Mono audio
                        abs_data = np.abs(oversampled_chunk)
                        gain_reduction = np.ones_like(abs_data)
                        mask = abs_data > threshold_linear
                        if np.any(mask):
                            gain_reduction[mask] = threshold_linear / abs_data[mask]
                        
                        # Apply smoothing with release filter
                        gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
                        gain_reduction = np.minimum(gain_reduction, gain_reduction_smooth)
                        
                        # Apply gain reduction
                        limited_oversampled = oversampled_chunk * gain_reduction
                    
                    # Downsample back to original rate
                    limited_chunk = optimized_resample(limited_oversampled, rate * oversampling_factor, rate)
                    
                    # Ensure correct shape before writing
                    if len(limited_chunk.shape) > 1:
                        if limited_chunk.shape[1] != channels:
                            # Keep the original number of channels
                            limited_chunk = limited_chunk[:, :channels] if limited_chunk.shape[1] > channels else np.pad(
                                limited_chunk, 
                                ((0, 0), (0, channels - limited_chunk.shape[1])), 
                                mode='constant'
                            )
                    elif len(limited_chunk.shape) == 1 and channels > 1:
                        # Convert mono to multi-channel
                        limited_chunk = np.column_stack([limited_chunk] * channels)
                    
                    # Write only the non-overlapping part except for last chunk
                    write_size = min(chunk_size, len(limited_chunk))
                    try:
                        outfile.write(limited_chunk[:write_size])
                    except ValueError as e:
                        print(f"Write error: {e}")
                        print(f"Limited chunk shape: {limited_chunk.shape}, Expected channels: {channels}")
                        # Emergency fix - reshape to correct dimensions
                        if len(limited_chunk.shape) > 1:
                            outfile.write(limited_chunk[:write_size, :channels])
                        else:
                            outfile.write(np.column_stack([limited_chunk[:write_size]] * channels))
                    
                    # Advance position
                    pos += write_size
                    pbar.update(write_size)

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
    perf_group.add_argument("-c", "--chunk_size", type=float, default=5.0,
                          help="Size of processing chunks in seconds (default: 5.0)")
    perf_group.add_argument("--no-cache", action="store_true", 
                          help="Disable caching of loudness analysis results")
    perf_group.add_argument("-v", "--verbose", action="store_true",
                          help="Show detailed optimization and processing information")
   

    # Handle the remaining arguments as lists of input/output files
    args, remaining = parser.parse_known_args()
    
    use_cache = not args.no_cache
    
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
            
        # Process each file using streaming architecture
        for input_file, output_file in zip(input_files, output_files):
            process_audio_streaming(input_file, output_file, 
                                  args.target_lufs, args.true_peak,
                                  args.lra_max, args.num_processes, 
                                  args.chunk_size, use_cache)
    else:
        # Single file mode with streaming
        process_audio_streaming(args.input_file, args.output_file, 
                              args.target_lufs, args.true_peak,
                              args.lra_max, args.num_processes,
                              args.chunk_size, use_cache)

if __name__ == "__main__":
    # Parse args just to get verbose flag
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-v", "--verbose", action="store_true")
    args, _ = parser.parse_known_args()
    verbose = args.verbose
    
    # Apply optimizations but respect verbose flag
    enable_optimizations(verbose)
    freeze_support()
    install_dependencies()
    enable_simd_optimizations(verbose)  # Add SIMD optimizations
    check_optimizations()        # Show optimization status
    main()
