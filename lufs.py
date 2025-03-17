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
    
    # Only show extended config when verbose mode is on
    if verbose:
        try:
            import numpy as np
            print("NumPy configuration details:")
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

def apply_brickwall_limiter(data, rate, true_peak_limit=-1.0, release_time=0.050, num_processes=max(1, cpu_count() - 1)):
    """
    Apply a brickwall limiter with CPU-intensive optimization and reduced memory usage.
    """
    from scipy import signal
    from tqdm import tqdm
    import gc  # For garbage collection
    
    print("Starting brickwall limiting process...")
    
    # Convert threshold to linear once
    threshold_linear = 10 ** (true_peak_limit / 20.0)
    
    # Create release curve once
    release_samples = int(release_time * rate)
    release_curve = np.exp(-np.arange(release_samples) / (release_samples / 5))
    release_curve = release_curve / np.sum(release_curve)
    
    # Determine if we're working with mono or stereo
    is_multichannel = len(data.shape) > 1
    num_channels = data.shape[1] if is_multichannel else 1
    
    # Choose a better chunk size - smaller for better parallelism
    chunk_size = 100000  # Reduced from 500000 
    
    # Process each channel separately for more CPU parallelism
    if is_multichannel:
        result = np.zeros_like(data)
        channel_data = []
        
        # Split channels for parallel processing
        for c in range(num_channels):
            channel_data.append(data[:, c])
        
        # Process each channel in parallel
        with Pool(processes=num_processes) as pool:
            result_channels = list(tqdm(
                pool.starmap(process_audio_channel, 
                            [(channel, rate, threshold_linear, release_curve, chunk_size) for channel in channel_data]),
                total=num_channels,
                desc="Processing channels"
            ))
        
        # Recombine channels
        for c in range(num_channels):
            result[:, c] = result_channels[c]
            
        # Help garbage collection 
        del result_channels
        gc.collect()
            
    else:  # Mono processing - use chunks in parallel
        num_chunks = max(1, len(data) // chunk_size)
        chunks = []
        
        # Create chunks with overlap
        overlap = release_samples * 2
        for i in range(0, len(data), chunk_size):
            end = min(i + chunk_size, len(data))
            # Add some context before and after the chunk for proper limiting
            start_with_context = max(0, i - overlap)
            end_with_context = min(end + overlap, len(data))
            
            chunk_info = {
                'start': i,
                'end': end,
                'data': data[start_with_context:end_with_context],
                'context_start': start_with_context,
                'offset': i - start_with_context
            }
            chunks.append(chunk_info)
            
        # Process chunks in parallel
        with Pool(processes=num_processes) as pool:
            processed_chunks = list(tqdm(
                pool.starmap(process_audio_chunk, 
                            [(chunk, rate, threshold_linear, release_curve) for chunk in chunks]),
                total=len(chunks),
                desc="Processing chunks"
            ))
            
        # Reassemble the audio
        result = np.zeros_like(data)
        for i, processed in enumerate(processed_chunks):
            chunk_info = chunks[i]
            start = chunk_info['start']
            end = chunk_info['end']
            offset = chunk_info['offset']
            
            # Place the processed audio in the correct position
            result[start:end] = processed[offset:offset + (end - start)]
            
        # Help garbage collection
        del chunks
        del processed_chunks
        gc.collect()
    
    return result

def process_audio_channel(channel_data, rate, threshold_linear, release_curve, chunk_size):
    """Process a single audio channel in chunks to reduce memory usage"""
    result = np.zeros_like(channel_data)
    
    for i in range(0, len(channel_data), chunk_size):
        end = min(i + chunk_size, len(channel_data))
        
        # Process this chunk
        chunk = channel_data[i:end]
        
        # Calculate gain reduction for this chunk
        abs_data = np.abs(chunk)
        gain_reduction = np.ones_like(abs_data)
        mask = abs_data > threshold_linear
        if np.any(mask):
            gain_reduction[mask] = threshold_linear / abs_data[mask]
            
            # Apply smoothing with CPU-intensive operation
            gain_reduction = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
        
        # Apply gain reduction (uses CPU for math)
        result[i:end] = chunk * gain_reduction
        
    return result

def process_audio_chunk(chunk_info, rate, threshold_linear, release_curve):
    """Process a chunk of audio with its context and return the limited result"""
    chunk_data = chunk_info['data']
    
    # Calculate gain reduction
    abs_data = np.abs(chunk_data)
    gain_reduction = np.ones_like(abs_data)
    mask = abs_data > threshold_linear
    
    if np.any(mask):
        # Apply CPU-intensive operations
        gain_reduction[mask] = threshold_linear / abs_data[mask]
        
        # Smooth the gain reduction curve
        gain_reduction = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
        
    # Apply the gain reduction
    limited_data = chunk_data * gain_reduction
    
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

def process_audio_streaming(input_file, output_file, target_lufs=-16.0, true_peak_limit=-1.0, 
                           lra_max=9.0, num_processes=max(1, cpu_count() - 1), 
                           chunk_size=10.0, use_cache=True):
    """
    Process audio in a streaming fashion with minimal memory usage and better CPU utilization.
    """
    import tempfile
    from scipy import signal
    import os
    import json
    import hashlib
    import time
    
    print("\n========== LUFS AUDIO NORMALIZER ==========")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Processing settings:")
    print(f"  • Target LUFS: {target_lufs:.1f} LUFS")
    print(f"  • True peak limit: {true_peak_limit:.1f} dBTP")
    print(f"  • Max LRA: {lra_max:.1f} LU")
    print(f"  • Processes: {num_processes}")
    print(f"  • Chunk size: {chunk_size:.1f} seconds")
    print("==========================================\n")
    
    # First check if input file exists and is supported
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return
    
    # Check if file format is supported, convert if needed
    try:
        with sf.SoundFile(input_file, 'r') as f:
            rate = f.samplerate
            channels = f.channels
            subtype = f.subtype
            frames = f.frames
            duration = frames / rate
    except Exception as e:
        print(f"Error opening audio file: {e}")
        print("Attempting to convert using ffmpeg...")
        
        # Try to convert using ffmpeg if installed
        try:
            import subprocess
            temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # Convert to WAV using ffmpeg
            cmd = ["ffmpeg", "-i", input_file, "-y", temp_wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            print(f"Converted input file to WAV format: {temp_wav_path}")
            input_file = temp_wav_path
            
            # Now try to open the converted file
            with sf.SoundFile(input_file, 'r') as f:
                rate = f.samplerate
                channels = f.channels
                subtype = f.subtype
                frames = f.frames
                duration = frames / rate
                
        except Exception as convert_error:
            print(f"Error converting file: {convert_error}")
            print("This file format is not supported. Please provide WAV, FLAC, or OGG files.")
            return
    
    print(f"Audio properties: {rate}Hz, {channels} channels, {duration:.1f}s, format: {subtype}")
    
    # Initialize meter with same rate
    meter = pyln.Meter(rate)
    
    # Calculate cache key based on file path and modification time
    file_stat = os.stat(input_file)
    file_hash = hashlib.md5(f"{input_file}:{file_stat.st_mtime}:{file_stat.st_size}".encode()).hexdigest()
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".lufs_cache")
    cache_file = os.path.join(cache_dir, f"{file_hash}.json")
    
    # Try to load from cache if enabled
    lra_info = None
    if use_cache:
        try:
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
                
            if os.path.exists(cache_file):
                with open(cache_file, 'r') as f:
                    cached_data = json.load(f)
                    # Verify the cached data matches our file
                    if cached_data.get("file") == input_file and \
                       cached_data.get("mtime") == file_stat.st_mtime and \
                       cached_data.get("size") == file_stat.st_size:
                        lra_info = cached_data.get("lra_info")
                        print(f"Using cached loudness analysis from {cache_file}")
        except Exception as cache_error:
            print(f"Warning: Cache read error: {cache_error}")
    
    # Perform analysis if not in cache
    if lra_info is None:
        print("Performing loudness analysis...")
        lra_info = analyze_lra_streaming_optimized(input_file, rate, meter, lra_max)
        
        # Save to cache if enabled
        if use_cache:
            try:
                cache_data = {
                    "file": input_file,
                    "mtime": file_stat.st_mtime,
                    "size": file_stat.st_size,
                    "date": time.time(),
                    "lra_info": lra_info
                }
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f)
                print(f"Saved loudness analysis to cache: {cache_file}")
            except Exception as cache_error:
                print(f"Warning: Cache write error: {cache_error}")
    
    print(f"Loudness analysis: {lra_info['loudness']:.2f} LUFS, LRA: {lra_info['lra']:.2f} LU")
    
    # Calculate target gain
    gain = target_lufs - lra_info['loudness']
    print(f"Target loudness: {target_lufs:.2f} LUFS (gain: {gain:.2f} dB)")
    
    # Create temp files with unique suffixes for multi-stage processing
    temp1_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp1_path = temp1_file.name
    temp1_file.close()
    
    temp2_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp2_path = temp2_file.name
    temp2_file.close()
    
    current_file = input_file
    
    try:
        # Stage 1: Apply dynamic range compression if needed
        if lra_max > 0 and lra_info['lra'] > lra_max:
            print(f"Stage 1: Applying dynamic range compression {lra_info['lra']:.2f} -> {lra_max:.2f} LU")
            apply_compression_streaming(
                current_file, 
                temp1_path,
                threshold=lra_info['threshold'],
                ratio=lra_info['ratio'],
                meter=meter,
                chunk_seconds=chunk_size,
                lra_max=lra_max  # Pass the parameter explicitly
            )
            print("Stage 1: Compression completed")
            current_file = temp1_path
        else:
            print("Stage 1: Dynamic range compression not needed")
        
        # Stage 2: Apply LUFS gain
        print(f"Stage 2: Applying LUFS gain of {gain:.2f} dB")
        apply_gain_streaming(
            current_file,
            temp2_path,
            gain=gain,
            chunk_seconds=chunk_size
        )
        print("Stage 2: Gain application completed")
        current_file = temp2_path
        
        # Stage 3: Measure true peak after gain application
        print("Measuring true peak after gain...")
        true_peak = measure_true_peak_streaming_parallel(current_file, chunk_size, num_processes)  # Use parallel version
        print(f"True peak after gain: {true_peak:.2f} dBTP")
        
        # Apply limiting if needed
        if true_peak > true_peak_limit:
            print(f"Stage 3: Applying true peak limiting {true_peak:.2f} -> {true_peak_limit:.2f} dBTP")
            apply_limiter_streaming(
                current_file,
                output_file,
                true_peak_limit=true_peak_limit,
                release_time=0.050,
                chunk_seconds=chunk_size
            )
            print("Stage 3: Limiting completed")
        else:
            print("Stage 3: True peak limiting not needed, copying file...")
            copy_audio_streaming(current_file, output_file, chunk_seconds=chunk_size)
        
        # Verify final output
        final_loudness = measure_loudness_streaming(output_file, meter, chunk_size)
        final_tp = measure_true_peak_streaming_parallel(output_file, chunk_size, num_processes)  # Use parallel version
        print(f"Final measurements: {final_loudness:.2f} LUFS, {final_tp:.2f} dBTP")

        return "Processing completed successfully"

    finally:
        # Clean up temporary files
        for path in [temp1_path, temp2_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)  # This line needs to be indented
            except Exception as e:
                print(f"Warning: Could not delete temporary file {path}: {e}")
                # Create sliding windows in this chunk
                for i in range(0, len(chunk) - window_size + 1, hop_size):
                    if i + window_size <= len(chunk):
                        window_data = chunk[i:i + window_size]
                        window_loudness = meter.integrated_loudness(window_data)
                        if window_loudness > -70:  # Ignore silence
                            st_loudness.append(window_loudness)
                
                pbar.update(len(chunk))
    
    # Calculate overall loudness
    overall_loudness = meter.integrated_loudness(sf.read(input_file)[0])
    
    # Calculate LRA
    result = {'loudness': overall_loudness, 'lra': 0, 'threshold': 0, 'ratio': 1.0}
    
    if len(st_loudness) >= 10:
        st_loudness.sort()
        p10_idx = max(0, int(len(st_loudness) * 0.1))
        p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
        lra = st_loudness[p95_idx] - st_loudness[p10_idx]
        result['lra'] = lra
        
        if lra > lra_max:
            ratio = lra / lra_max
            threshold = st_loudness[p10_idx] + ((lra / 2) * (1 - 1/ratio))
            result['ratio'] = ratio
            result['threshold'] = threshold
    
    return result

def apply_multi_stage_compression(data, rate, current_lra, target_lra):
    """
    Apply multi-stage compression to reduce LRA in a more natural way
    but with stronger settings to ensure actual reduction.
    """
    print("\n--- Starting Multi-Stage LRA Reduction ---")
    print(f"Current LRA: {current_lra:.2f} LU, Target: {target_lra:.2f} LU")
    
    # Only compress if we need to reduce LRA
    if current_lra <= target_lra:
        print(f"No compression needed - current LRA ({current_lra:.2f}) already below target ({target_lra:.2f})")
        return data
    
    # Calculate how many stages we need (0.8 LU per stage - more aggressive)
    lra_to_reduce = current_lra - target_lra
    num_stages = min(8, max(1, int(np.ceil(lra_to_reduce / 0.8))))
    
    # Metadata to track changes
    meter = pyln.Meter(rate)
    original_loudness = meter.integrated_loudness(data)
    processed_data = data.copy()
    
    print(f"Using {num_stages} compression stages to reduce LRA by {lra_to_reduce:.2f} LU")
    print(f"Starting loudness: {original_loudness:.2f} LUFS")
    
    # Create a list of stage configurations with varying parameters
    stages = []
    
    # Stage 1: High threshold, moderate ratio - targets peaks
    stages.append({
        'name': 'High peaks',
        'percentile': 90,
        'ratio': 2.5,
        'attack': 0.001,  # Very fast
        'release': 0.100  # Moderate
    })
    
    # Stage 2: Medium-high threshold, stronger ratio
    if num_stages >= 2:
        stages.append({
            'name': 'Medium-high peaks',
            'percentile': 80,
            'ratio': 3.0,
            'attack': 0.005,
            'release': 0.150
        })
    
    # Stage 3: Medium threshold, stronger ratio
    if num_stages >= 3:
        stages.append({
            'name': 'Medium peaks',
            'percentile': 65,
            'ratio': 3.5,
            'attack': 0.010,
            'release': 0.200
        })
    
    # Stage 4: Medium-low threshold
    if num_stages >= 4:
        stages.append({
            'name': 'Medium-low range',
            'percentile': 50,
            'ratio': 4.0,
            'attack': 0.015,
            'release': 0.300
        })
    
    # Add additional stages with progressively lower thresholds
    for i in range(4, num_stages):
        stages.append({
            'name': f'Low range {i-3}',
            'percentile': max(30, 50 - (i-3)*10),
            'ratio': min(8.0, 4.0 + (i-3)*1.0),
            'attack': min(0.030, 0.015 + (i-3)*0.005),
            'release': min(0.500, 0.300 + (i-3)*0.100)
        })
    
    # Apply each stage
    for i, stage in enumerate(stages):
        print(f"\nStage {i+1}/{len(stages)}: {stage['name']}")
        print(f"  Percentile: {stage['percentile']}%, Ratio: {stage['ratio']:.1f}:1")
        print(f"  Attack: {stage['attack']*1000:.1f}ms, Release: {stage['release']*1000:.1f}ms")
        
        # Calculate loudness values to determine threshold
        loudness_values = []
        window_size = int(1 * rate)  # 1-second window
        hop_size = int(0.2 * rate)   # 200ms hop
        
        # Get loudness values from throughout the file
        for j in range(0, len(processed_data) - window_size, hop_size):
            if j > len(processed_data) - window_size:
                break
            window_data = processed_data[j:j + window_size]
            window_loudness = meter.integrated_loudness(window_data)
            if window_loudness > -70:  # Ignore silence
                loudness_values.append(window_loudness)
        
        if not loudness_values:
            print("  No valid loudness measurements, skipping stage")
            continue
            
        loudness_values.sort()
        threshold_idx = min(len(loudness_values)-1, 
                           max(0, int(len(loudness_values) * (stage['percentile']/100))))
        threshold_loudness = loudness_values[threshold_idx]
        
        # Convert to linear domain - using both amplitude and dB to ensure accuracy
        threshold_linear = 10 ** (threshold_loudness/20)
        print(f"  Threshold: {threshold_loudness:.2f} LUFS ({threshold_linear:.6f} linear)")
        
        # Apply this stage of compression
        processed_data = apply_compressor_with_time_constants(
            processed_data, 
            rate, 
            threshold_linear,
            stage['ratio'],
            stage['attack'],
            stage['release']
        )
        
        # Re-measure and correct loudness
        stage_loudness = meter.integrated_loudness(processed_data)
        gain_adjust = original_loudness - stage_loudness
        print(f"  Post-compression loudness: {stage_loudness:.2f} LUFS")
        print(f"  Applying correction gain: {gain_adjust:.2f} dB")
        
        # Maintain original loudness
        processed_data = processed_data * (10**(gain_adjust/20))
    
    # Measure final LRA
    try:
        # Approximation of final LRA
        window_size = int(3 * rate)
        hop_size = int(0.1 * rate)
        st_loudness = []
        
        for i in range(0, len(processed_data) - window_size, hop_size):
            window_data = processed_data[i:i + window_size]
            window_loudness = meter.integrated_loudness(window_data)
            if window_loudness > -70:
                st_loudness.append(window_loudness)
                
        if len(st_loudness) >= 10:
            st_loudness.sort()
            p10_idx = max(0, int(len(st_loudness) * 0.1))
            p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
            final_lra = st_loudness[p95_idx] - st_loudness[p10_idx]
            print(f"\nFinal estimated LRA: {final_lra:.2f} LU")
            print(f"Original LRA: {current_lra:.2f} LU")
            print(f"Reduction: {current_lra - final_lra:.2f} LU")
    except:
        print("\nCouldn't estimate final LRA")
    
    final_loudness = meter.integrated_loudness(processed_data)
    print(f"Final loudness: {final_loudness:.2f} LUFS (target was {original_loudness:.2f} LUFS)")
    
    return processed_data

def apply_compressor_with_time_constants(data, rate, threshold, ratio, attack_time, release_time):
    """
    Apply compression with proper attack and release time constants.
    Improved version with more aggressive compression.
    """
    from scipy import signal
    
    # Calculate time constants in samples
    attack_samples = max(1, int(attack_time * rate))
    release_samples = max(1, int(release_time * rate))
    
    # Create attack/release coefficients for first-order filter
    attack_coef = np.exp(-1.0 / attack_samples)
    release_coef = np.exp(-1.0 / release_samples)
    
    # Calculate static gain reduction
    abs_data = np.abs(data)
    mask = abs_data > threshold
    
    # Skip processing if no samples exceed threshold
    if not np.any(mask):
        return data
    
    # Create gain reduction curve
    gain_reduction = np.ones_like(data, dtype=np.float32)
    
    # Process each channel separately
    if len(data.shape) > 1:  # Multi-channel
        for c in range(data.shape[1]):
            # Static gain calculation
            channel_abs = abs_data[:, c]
            channel_mask = channel_abs > threshold
            channel_gain = np.ones_like(channel_abs)
            
            # Apply compression formula with stronger ratio
            above_thresh = channel_abs[channel_mask] - threshold
            compressed_above = above_thresh / ratio
            channel_gain[channel_mask] = (threshold + compressed_above) / channel_abs[channel_mask]
            
            # Apply smoothing with first-order filter
            smoothed_gain = np.ones_like(channel_gain)
            for i in range(1, len(channel_gain)):
                if channel_gain[i] < smoothed_gain[i-1]:  # Attack phase
                    smoothed_gain[i] = attack_coef * smoothed_gain[i-1] + (1 - attack_coef) * channel_gain[i]
                else:  # Release phase
                    smoothed_gain[i] = release_coef * smoothed_gain[i-1] + (1 - release_coef) * channel_gain[i]
            
            gain_reduction[:, c] = smoothed_gain
    else:  # Mono
        # Static gain calculation
        gain_static = np.ones_like(abs_data)
        
        # Apply compression formula with stronger ratio
        above_thresh = abs_data[mask] - threshold
        compressed_above = above_thresh / ratio
        gain_static[mask] = (threshold + compressed_above) / abs_data[mask]
        
        # Apply smoothing with first-order filter
        smoothed_gain = np.ones_like(gain_static)
        for i in range(1, len(gain_static)):
            if gain_static[i] < smoothed_gain[i-1]:  # Attack phase
                smoothed_gain[i] = attack_coef * smoothed_gain[i-1] + (1 - attack_coef) * gain_static[i]
            else:  # Release phase
                smoothed_gain[i] = release_coef * smoothed_gain[i-1] + (1 - release_coef) * gain_static[i]
        
        gain_reduction = smoothed_gain
    
    # Apply smoothed gain reduction
    result = data * gain_reduction
    
    return result

def apply_compression_streaming(input_file, output_file, threshold, ratio, meter, chunk_seconds=5.0, lra_max=9.0):
    """Apply multi-stage compression in streaming fashion"""
    # First load the data for analysis (we need to see the whole file for proper LRA reduction)
    print("Loading audio for LRA analysis and processing...")
    data, rate = sf.read(input_file)
    
    # Measure the current LRA precisely
    window_size = int(3 * rate)
    hop_size = int(0.1 * rate)
    st_loudness = []
    
    print("Measuring exact LRA for compression planning...")
    for i in range(0, len(data) - window_size, hop_size):
        window_data = data[i:i + window_size]
        window_loudness = meter.integrated_loudness(window_data)
        if window_loudness > -70:  # Ignore silence
            st_loudness.append(window_loudness)
            
    if len(st_loudness) >= 10:
        st_loudness.sort()
        p10_idx = max(0, int(len(st_loudness) * 0.1))
        p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
        current_lra = st_loudness[p95_idx] - st_loudness[p10_idx]
        print(f"Precise LRA measurement: {current_lra:.2f} LU")
        
        # Use the parameter passed to this function, not environment variable
        target_lra = lra_max
        print(f"Applying multi-stage compression to reduce LRA from {current_lra:.2f} to {target_lra:.2f} LU")
        
        # Use the multi-stage compression!
        processed_data = apply_multi_stage_compression(data, rate, current_lra, target_lra)
        
        # Write the processed data
        print("Writing compressed audio...")
        sf.write(output_file, processed_data, rate, subtype=sf.SoundFile(input_file, 'r').subtype)
        
        # Verify final LRA
        final_st_loudness = []
        for i in range(0, len(processed_data) - window_size, hop_size):
            window_data = processed_data[i:i + window_size]
            window_loudness = meter.integrated_loudness(window_data)
            if window_loudness > -70:
                final_st_loudness.append(window_loudness)
                
        if len(final_st_loudness) >= 10:
            final_st_loudness.sort()
            p10_idx = max(0, int(len(final_st_loudness) * 0.1))
            p95_idx = min(len(final_st_loudness) - 1, int(len(final_st_loudness) * 0.95))
            final_lra = final_st_loudness[p95_idx] - final_st_loudness[p10_idx]
            print(f"Final LRA: {final_lra:.2f} LU (reduction: {current_lra - final_lra:.2f} LU)")
    
    return "Multi-stage compression completed"

def apply_gain_streaming(input_file, output_file, gain, chunk_seconds=5.0):
    """Apply gain in streaming fashion"""
    gain_factor = 10 ** (gain / 20.0)
    
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Process in chunks
            chunk_size = int(rate * chunk_seconds)
            
            with tqdm(total=infile.frames, desc="Applying gain", unit="samples") as pbar:
                while infile.tell() < infile.frames:
                    # Read chunk
                    chunk = infile.read(chunk_size)
                    if len(chunk) == 0:
                        break
                        
                    # Apply gain
                    processed_chunk = chunk * gain_factor
                    
                    # Write processed chunk
                    outfile.write(processed_chunk)
                    pbar.update(len(chunk))
    
    return "Gain application completed"

def apply_limiter_streaming(input_file, output_file, true_peak_limit, release_time, chunk_seconds=5.0):
    """Apply true peak limiting in a streaming fashion with proper oversampling"""
    from scipy import signal
    
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Add safety margin to ensure we actually hit the target
            # Using a 0.1dB safety margin compensates for lossy processing
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
                    
                    # CRUCIAL CHANGE: Upsample for true peak detection and limiting
                    oversampling_factor = 4
                    oversampled_chunk = optimized_resample(chunk, rate, rate * oversampling_factor)  # Use optimized version
                    
                    # Calculate gain reduction based on oversampled signal
                    abs_data = np.abs(oversampled_chunk)
                    gain_reduction = np.ones_like(abs_data)
                    mask = abs_data > threshold_linear
                    if np.any(mask):
                        gain_reduction[mask] = threshold_linear / abs_data[mask]
                    
                    # Apply smoothing filter to gain reduction
                    if len(oversampled_chunk.shape) > 1:  # Multi-channel audio
                        for c in range(channels):
                            # Forward-backward smoothing for more transparent limiting
                            gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[:, c][::-1])[::-1]
                            gain_reduction[:, c] = np.minimum(gain_reduction[:, c], gain_reduction_smooth)
                    else:  # Mono audio
                        gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
                        gain_reduction = np.minimum(gain_reduction, gain_reduction_smooth)
                    
                    # Apply gain reduction to oversampled signal
                    limited_oversampled = oversampled_chunk * gain_reduction
                    
                    # Downsample back to original rate
                    limited_chunk = optimized_resample(limited_oversampled, rate * oversampling_factor, rate)  # Use optimized version
                    
                    # Write only the non-overlapping part except for last chunk
                    write_size = min(chunk_size, len(limited_chunk))
                    outfile.write(limited_chunk[:write_size])
                    
                    # Advance position
                    pos += write_size
                    pbar.update(write_size)
                    
            # Final verification of true peak
            print("Verifying final true peak level...")
            final_tp = measure_true_peak_streaming_parallel(output_file, chunk_seconds, num_processes=cpu_count() - 1)  # Use parallel version
            print(f"Final true peak after limiting: {final_tp:.2f} dBTP")
            
            # If still above threshold (unlikely), apply a final quick limiting pass
            if final_tp > true_peak_limit:
                print(f"Warning: True peak {final_tp:.2f} dBTP still above target {true_peak_limit:.2f} dBTP.")
                print("Applying final gain adjustment...")
                
                # Create another temp file
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_path = temp_file.name
                temp_file.close()
                
                try:
                    # Move the current output to temp
                    os.rename(output_file, temp_path)
                    
                    # Apply a final gain reduction
                    safety_gain = true_peak_limit - final_tp - 0.05  # Extra safety margin
                    apply_gain_streaming(temp_path, output_file, safety_gain, chunk_seconds)
                    
                    # Clean up
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Error during final adjustment: {e}")
    
    return "Limiting completed"

def copy_audio_streaming(input_file, output_file, chunk_seconds=5.0):
    """Simple streaming copy function"""
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Process in chunks
            chunk_size = int(rate * chunk_seconds)
            
            with tqdm(total=infile.frames, desc="Copying audio", unit="samples") as pbar:
                while infile.tell() < infile.frames:
                    # Read chunk
                    chunk = infile.read(chunk_size)
                    if len(chunk) == 0:
                        break
                        
                    # Write chunk directly
                    outfile.write(chunk)
                    pbar.update(len(chunk))
    
    return "Copy completed"

def measure_loudness_streaming(audio_file, meter, chunk_seconds=5.0):
    """Measure integrated loudness in a streaming fashion"""
    # This is more complex since we need to gather statistics across the entire file
    program = meter.integrated_loudness(sf.read(audio_file)[0])
    return program

def measure_true_peak_streaming(audio_file, chunk_seconds=5.0):
    """Measure true peak in a streaming fashion"""
    # This can be done in chunks to reduce memory usage
    max_peak = -120.0
    
    with sf.SoundFile(audio_file, 'r') as f:
        rate = f.samplerate
        chunk_size = int(rate * chunk_seconds)
        
        with tqdm(total=f.frames, desc="Measuring true peak", unit="samples") as pbar:
            while f.tell() < f.frames:
                # Read chunk
                chunk = f.read(chunk_size)
                if len(chunk) == 0:
                    break
                
                # Measure true peak in this chunk
                # Use 4x oversampling for accurate peak measurement
                oversampled_chunk = optimized_resample(chunk, rate, rate * 4)  # Use optimized version
                chunk_peak = np.max(np.abs(oversampled_chunk))
                if chunk_peak > 0:
                    chunk_peak_db = 20 * np.log10(chunk_peak)
                    max_peak = max(max_peak, chunk_peak_db)
                
                pbar.update(len(chunk))
    
    return max_peak

def optimized_resample(data, orig_sr, target_sr):
    """Use PyTorch with Metal for faster resampling on Apple Silicon"""
    import platform
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import torch
            if torch.backends.mps.is_available():
                # Convert to torch tensor and move to MPS device
                device = torch.device("mps")
                
                # Handle multi-channel audio properly
                if len(data.shape) > 1:
                    # Transpose to [C, T] format for torchaudio
                    tensor_data = torch.tensor(data.T, device=device, dtype=torch.float32)
                    channels = data.shape[1]
                else:
                    # Add channel dimension for mono
                    tensor_data = torch.tensor(data, device=device, dtype=torch.float32).unsqueeze(0)
                    channels = 1
                
                # Use torchaudio's resampler which supports MPS acceleration
                import torchaudio.functional as F
                resampled = F.resample(tensor_data, orig_sr, target_sr)
                
                # Convert back to numpy array with correct shape
                if channels > 1:
                    # Transpose back to [T, C] format
                    return resampled.cpu().numpy().T
                else:
                    return resampled.squeeze(0).cpu().numpy()
                    
        except ImportError:
            print("PyTorch/torchaudio not available, using resampy instead")
        except Exception as e:
            print(f"Metal acceleration error: {e}, falling back to resampy")
            
    # Fall back to resampy if torch isn't available or for non-Apple Silicon
    return resampy.resample(data, orig_sr, target_sr)

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

def check_optimizations(verbose=False):
    """Check if running optimized libraries for current architecture"""
    import numpy as np
    
    if verbose:
        print("\nLibrary optimization check:")
        
        # Check if running on Apple Silicon
        import platform
        is_apple_silicon = platform.system() == 'Darwin' and platform.machine() == 'arm64'
        print(f"Running on Apple Silicon: {is_apple_silicon}")
        
        # Check NumPy config
        print(f"NumPy version: {np.__version__}")
        try:
            # Try to get BLAS info
            blas_info = np.__config__.get_info('blas_opt')
            if 'accelerate' in str(blas_info).lower():
                print("NumPy: Using Apple Accelerate framework ✓")
            elif 'mkl' in str(blas_info).lower():
                print("NumPy: Using Intel MKL")
            elif 'openblas' in str(blas_info).lower():
                print("NumPy: Using OpenBLAS")
            else:
                print("NumPy: Using standard BLAS implementation")
        except:
            print("Couldn't determine NumPy BLAS implementation")
            
        # Check if OpenMP is available for parallel processing
        try:
            from scipy import __config__
            if 'openmp' in str(__config__.get_info('ALL')).lower():
                print("SciPy: OpenMP enabled for parallel processing ✓")
            else:
                print("SciPy: OpenMP not detected")
        except:
            print("Couldn't determine SciPy parallelization")

if __name__ == "__main__":
    freeze_support()
    install_dependencies()
    # Parse args just to get verbose flag
    import sys
    verbose = "--verbose" in sys.argv or "-v" in sys.argv
    enable_simd_optimizations(verbose)  # Only show details if verbose
    check_optimizations(verbose)        # Show optimization status
    main()

def analyze_lra_streaming_optimized(input_file, rate, meter, lra_max):
    """
    Optimized LRA analysis with better parallelization and clear visibility.
    """
    from multiprocessing import Pool, cpu_count
    import numpy as np
    import time
    from tqdm.auto import tqdm
    
    # First get file duration for planning and visibility
    with sf.SoundFile(input_file, 'r') as f:
        duration = f.frames / f.samplerate
        channels = f.channels
        total_frames = f.frames
    
    print(f"Starting LRA analysis for {duration:.1f}s audio ({channels} channels)...")
    
    # Create a progress bar for overall process with clear stages
    stages = [
        ("Pre-scanning", 10),
        ("Window processing", 80), 
        ("Final calculation", 10)
    ]
    
    stage_names = [s[0] for s in stages]
    stage_weights = [s[1] for s in stages]
    
    # Create main progress bar
    with tqdm(total=100, desc="LRA Analysis", unit="%") as main_pbar:
        # STAGE 1: Fast pre-scan for silence detection
        # Use much larger hop size (1s) and simple RMS to detect obvious silence
        print(f"\n[{stage_names[0]}] Quick silence detection scan...")
        silence_map = np.ones(int(np.ceil(duration)), dtype=bool)  # 1-second resolution silence map
        
        # Optimize chunk size for pre-scan
        chunk_seconds = 5.0
        chunk_frames = int(chunk_seconds * rate)
        silent_threshold_db = -50  # Frames below this RMS are considered silent
        silent_threshold_linear = 10**(silent_threshold_db/20)
        
        # Pre-scan in larger chunks using RMS (much faster than LUFS calculation)
        with sf.SoundFile(input_file, 'r') as f:
            with tqdm(total=total_frames, desc="Scanning for silence", unit="frames") as scan_pbar:
                for i in range(0, int(duration)):
                    if i * rate < total_frames:
                        f.seek(i * rate)
                        # Read a small portion for RMS calculation
                        chunk = f.read(min(rate, total_frames - f.tell()))
                        
                        # RMS calculation is much faster than LUFS
                        if len(chunk.shape) > 1:  # Multi-channel
                            # Mix down to mono
                            mono = np.mean(chunk, axis=1)
                            rms = np.sqrt(np.mean(mono**2))
                        else:
                            rms = np.sqrt(np.mean(chunk**2))
                            
                        # Mark as silent if below threshold
                        if rms < silent_threshold_linear:
                            silence_map[i] = False
                        
                        scan_pbar.update(len(chunk))
                
        main_pbar.update(stage_weights[0])
        
        # Report silence percentage
        silent_seconds = np.sum(~silence_map)
        active_seconds = np.sum(silence_map)
        print(f"Silence detection: {silent_seconds} seconds silence, {active_seconds} seconds active audio")
        
        # STAGE 2: More detailed analysis with efficient batching
        print(f"\n[{stage_names[1]}] Processing audio segments...")
        
        # Use more efficient parameters for analysis chunks
        window_size = int(3 * rate)       # Standard 3s window for LUFS
        hop_size = int(1.0 * rate)        # 1s hop for much faster processing 
                                          # (Fewer overlapping windows, but still accurate enough for LRA)
        
        # Only analyze segments with active audio 
        st_loudness = []
        num_processes = max(1, min(cpu_count() - 1, 4))  # Limit to avoid memory issues
        
        # Create a processing pool that we'll reuse for efficiency
        with Pool(processes=num_processes) as pool:
            # Count windows for progress tracking
            active_windows = sum([1 for i in range(len(silence_map)) if silence_map[i]])
            
            # Process in larger chunks for better performance
            batch_size = 4   # Process 4 seconds at once
            batches = []
            
            # Create batches of windows that are likely to have active audio
            for i in range(0, len(silence_map), batch_size):
                batch_end = min(i + batch_size, len(silence_map))
                if any(silence_map[i:batch_end]):  # If any frame in batch is active
                    batches.append((i, batch_end))
            
            # Process batches with progress bar
            with tqdm(total=len(batches), desc="Processing audio batches", unit="batch") as batch_pbar:
                for batch_start, batch_end in batches:
                    # Read this batch of audio
                    batch_start_frame = batch_start * rate
                    batch_end_frame = min(batch_end * rate, total_frames)
                    batch_frames = batch_end_frame - batch_start_frame
                    
                    with sf.SoundFile(input_file, 'r') as f:
                        f.seek(batch_start_frame)
                        batch_audio = f.read(batch_frames)
                    
                    # Create windows within this batch
                    windows = []
                    for j in range(0, len(batch_audio) - window_size + 1, hop_size):
                        if j + window_size <= len(batch_audio):
                            windows.append(batch_audio[j:j + window_size])
                    
                    # Process all windows in this batch in parallel
                    if windows:
                        # Process in larger chunks for better parallel efficiency
                        window_results = pool.map(measure_loudness_wrapper, [(w, meter) for w in windows])
                        valid_results = [r for r in window_results if r > -70]
                        st_loudness.extend(valid_results)
                    
                    batch_pbar.update(1)
                    # Update the main progress bar proportionally
                    main_pbar.update(stage_weights[1] / len(batches))
        
        # STAGE 3: Calculate final results
        print(f"\n[{stage_names[2]}] Calculating final statistics...")
        
        # Fallback to full file loudness if windows didn't work
        with sf.SoundFile(input_file, 'r') as f:
            # Read file in chunks for overall loudness
            block_size = int(rate * 10)  # 10 second blocks
            blocks = []
            
            with tqdm(total=total_frames, desc="Measuring integrated loudness", unit="frames") as loud_pbar:
                while f.tell() < total_frames:
                    block = f.read(min(block_size, total_frames - f.tell()))
                    blocks.append(block)
                    loud_pbar.update(len(block))
            
            # Combine all blocks for integrated loudness measurement
            full_audio = np.concatenate(blocks) if len(blocks) > 1 else blocks[0]
            overall_loudness = meter.integrated_loudness(full_audio)
        
        # Calculate LRA
        result = {'loudness': overall_loudness, 'lra': 0, 'threshold': 0, 'ratio': 1.0}
        
        if len(st_loudness) >= 10:
            st_loudness.sort()
            p10_idx = max(0, int(len(st_loudness) * 0.1))
            p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
            lra = st_loudness[p95_idx] - st_loudness[p10_idx]
            result['lra'] = lra
            
            print(f"Successfully calculated LRA from {len(st_loudness)} measurement windows")
            
            if lra > lra_max:
                ratio = lra / lra_max
                threshold = st_loudness[p10_idx] + ((lra / 2) * (1 - 1/ratio))
                result['ratio'] = ratio
                result['threshold'] = threshold
        else:
            print(f"Warning: Only {len(st_loudness)} valid loudness measurements found (min 10 required)")
            print("Using fallback method to estimate LRA")
            # Make a conservative estimate
            result['lra'] = 5.0  # Conservative default
        
        # Complete the main progress bar
        main_pbar.update(stage_weights[2])
        
        return result
    """
    Measure LRA with progress bar and parallel processing
    """
    window_size = int(3 * rate)  # 3s window
    hop_size = int(0.1 * rate)   # 100ms hop
    st_loudness = []
    
    # Estimate number of windows for progress bar
    num_windows = (len(data) - window_size) // hop_size
    
    print("Measuring exact LRA for compression planning...")
    with tqdm(total=num_windows, desc="Analyzing windows", unit="windows") as pbar:
        # Process in batches for better progress reporting
        batch_size = 100  # Process 100 windows at once
        
        for i in range(0, len(data) - window_size + 1, hop_size * batch_size):
            batch_end = min(i + (hop_size * batch_size), len(data) - window_size + 1)
            batch_windows = range(i, batch_end, hop_size)
            
            for j in batch_windows:
                window_data = data[j:j + window_size]
                window_loudness = meter.integrated_loudness(window_data)
                if window_loudness > -70:  # Ignore silence
                    st_loudness.append(window_loudness)
            
            # Update progress
            pbar.update(len(batch_windows))
    
    # Calculate LRA if we have enough measurements
    if len(st_loudness) >= 10:
        st_loudness.sort()
        p10_idx = max(0, int(len(st_loudness) * 0.1))
        p95_idx = min(len(st_loudness) - 1, int(len(st_loudness) * 0.95))
        lra = st_loudness[p95_idx] - st_loudness[p10_idx]
        return lra
    else:
        print("Not enough valid loudness measurements")
        return 0.0

def measure_true_peak_streaming_parallel(audio_file, chunk_seconds=5.0, num_processes=None):
    """Parallel true peak measurement using multiple processes"""
    import math
    from multiprocessing import Pool
    
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # First determine file properties
    with sf.SoundFile(audio_file, 'r') as f:
        rate = f.samplerate
        total_frames = f.frames
    
    # Create chunks for parallel processing
    chunk_size = int(rate * chunk_seconds)
    num_chunks = math.ceil(total_frames / chunk_size)
    
    # Create a list of chunk positions
    chunk_positions = [(audio_file, i * chunk_size, min((i+1) * chunk_size, total_frames), rate) 
                      for i in range(num_chunks)]
    
    # Process chunks in parallel
    with Pool(processes=min(num_processes, num_chunks)) as pool:
        results = list(tqdm(
            pool.imap(measure_chunk_true_peak, chunk_positions),
            total=len(chunk_positions),
            desc="Measuring true peak",
            unit="chunk"
        ))
    
    # Return the maximum true peak
    return max(results)
    
def measure_chunk_true_peak(args):
    """Process a single chunk for true peak measurement"""
    filename, start, end, rate = args
    
    # Read the chunk
    with sf.SoundFile(filename, 'r') as f:
        f.seek(start)
        chunk = f.read(end - start)
    
    # Process with 4x oversampling
    oversampled_chunk = optimized_resample(chunk, rate, rate * 4)  # Use optimized version
    chunk_peak = np.max(np.abs(oversampled_chunk))
    
    if chunk_peak > 0:
        return 20 * np.log10(chunk_peak)
    else:
        return -120.0

def apply_multi_stage_compression_parallel(data, rate, current_lra, target_lra, num_processes=None):
    """
    Parallel multi-stage compression to reduce LRA in a more natural way
    """
    print("\n--- Starting Parallel Multi-Stage LRA Reduction ---")
    print(f"Current LRA: {current_lra:.2f} LU, Target: {target_lra:.2f} LU")
    
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
    
    # Only compress if we need to reduce LRA
    if current_lra <= target_lra:
        print(f"No compression needed")
        return data
    
    # Calculate how many stages we need
    lra_to_reduce = current_lra - target_lra
    num_stages = min(8, max(1, int(np.ceil(lra_to_reduce / 0.8))))
    
    # Metadata to track changes
    meter = pyln.Meter(rate)
    original_loudness = meter.integrated_loudness(data)
    
    print(f"Using {num_stages} compression stages to reduce LRA by {lra_to_reduce:.2f} LU")
    print(f"Starting loudness: {original_loudness:.2f} LUFS")
    
    # Create stages configuration
    stages = []
    
    # Define stage configurations as before
    # ... (your existing stage configuration code) ...
    
    # Create a process pool for parallel stage analysis
    pool = Pool(processes=min(num_processes, num_stages))
    
    # First analyze the file in parallel to determine thresholds
    # (each stage needs to calculate its own threshold based on signal statistics)
    stage_configs = []
    for i, stage in enumerate(stages):
        # Create a configuration for this stage
        config = {
            'stage_num': i+1,
            'stage_name': stage['name'],
            'percentile': stage['percentile'],
            'ratio': stage['ratio'],
            'attack': stage['attack'],
            'release': stage['release'],
            'data': data,  # This will be large but necessary for parallel processing
            'rate': rate
        }
        stage_configs.append(config)
    
    # Process stages in parallel for threshold analysis
    analyzed_stages = list(tqdm(
        pool.imap(analyze_compression_stage, stage_configs),
        total=len(stage_configs),
        desc="Analyzing compression stages"
    ))
    
    # Now apply stages sequentially (compression stages are inherently sequential)
    processed_data = data.copy()
    for stage_result in analyzed_stages:
        print(f"\nApplying stage {stage_result['stage_num']}/{len(stages)}: {stage_result['stage_name']}")
        print(f"  Threshold: {stage_result['threshold_loudness']:.2f} LUFS")
        print(f"  Ratio: {stage_result['ratio']:.1f}:1")
        
        processed_data = apply_compressor_with_time_constants(
            processed_data, 
            rate, 
            stage_result['threshold_linear'],
            stage_result['ratio'],
            stage_result['attack'],
            stage_result['release']
        )
        
        # Re-measure and correct loudness
        stage_loudness = meter.integrated_loudness(processed_data)
        gain_adjust = original_loudness - stage_loudness
        print(f"  Post-compression loudness: {stage_loudness:.2f} LUFS")
        print(f"  Applying correction gain: {gain_adjust:.2f} dB")
        
        # Maintain original loudness
        processed_data = processed_data * (10**(gain_adjust/20))
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Measure final LRA
    # WRITE THE MISSING CODE!
    
    return processed_data

def analyze_compression_stage(config):
    """Analyze a compression stage to determine threshold (for parallel execution)"""
    # Extract configuration
    data = config['data']
    rate = config['rate']
    percentile = config['percentile']
    
    # Calculate loudness values to determine threshold
    meter = pyln.Meter(rate)
    loudness_values = []
    window_size = int(1 * rate)
    hop_size = int(0.2 * rate)
    
    # Use strided analysis to improve performance
    for j in range(0, len(data) - window_size, hop_size):
        window_data = data[j:j + window_size]
        window_loudness = meter.integrated_loudness(window_data)
        if window_loudness > -70:
            loudness_values.append(window_loudness)
    
    # Calculate threshold based on percentile
    if loudness_values:
        loudness_values.sort()
        threshold_idx = min(len(loudness_values)-1, 
                           max(0, int(len(loudness_values) * (percentile/100))))
        threshold_loudness = loudness_values[threshold_idx]
        
        # Convert to linear domain
        threshold_linear = 10 ** (threshold_loudness/20)
        
        # Return the results along with the original config
        result = config.copy()
        result['threshold_loudness'] = threshold_loudness
        result['threshold_linear'] = threshold_linear
        
        # Remove large data array from result to save memory
        result.pop('data')
        
        return result
    else:
        # Return original config with default values if no valid loudness
        result = config.copy()
        result['threshold_loudness'] = -20.0
        result['threshold_linear'] = 0.1
        result.pop('data')
        return result

def apply_metal_optimized_dsp(audio_file, output_file, process_fn):
    """
    Process audio using Metal acceleration on Apple Silicon.
    This is a general framework for accelerated DSP operations.
    
    Args:
        audio_file: Input audio path
        output_file: Output audio path
        process_fn: Function that processes numpy arrays using Metal
    """
    import platform
    
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        try:
            import torch
            if torch.backends.mps.is_available():
                print("Using Metal Performance Shaders for audio processing")
                
                # Read audio data
                data, rate = sf.read(audio_file)
                
                # Get the subtype from the input file
                with sf.SoundFile(audio_file, 'r') as f:
                    subtype = f.subtype
                
                # Convert to torch tensor and move to MPS device
                device = torch.device("mps")
                
                if len(data.shape) > 1:  # Stereo/multichannel
                    tensor_data = torch.tensor(data, device=device, dtype=torch.float32)
                else:  # Mono
                    tensor_data = torch.tensor(data, device=device, dtype=torch.float32).unsqueeze(1)
                
                # Apply the processing function
                processed_tensor = process_fn(tensor_data)
                
                # Move back to CPU and convert to numpy
                processed_data = processed_tensor.cpu().numpy()
                
                # Write output
                sf.write(output_file, processed_data, rate, subtype=subtype)
                return True
        except Exception as e:
            print(f"Metal acceleration failed: {e}")
    
    # Fallback to regular processing
    return False

def metal_optimized_gain(input_tensor, gain_db):
    """Example of a Metal-optimized gain function"""
    gain_linear = 10 ** (gain_db / 20.0)
    return input_tensor * gain_linear

def setup_apple_silicon_optimizations():
    """Configure environment for optimal performance on Apple Silicon"""
    import platform
    
    if platform.system() == 'Darwin' and platform.machine() == 'arm64':
        import os
        # Use Accelerate framework for numpy/scipy
        os.environ['ACCELERATE'] = '1'
        
        # Set threading options
        os.environ['OMP_NUM_THREADS'] = str(max(1, cpu_count() - 1))
        os.environ['MKL_NUM_THREADS'] = str(max(1, cpu_count() - 1))
        
        # Enable Metal Performance Shaders if PyTorch is available
        try:
            import torch
            if torch.backends.mps.is_available():
                os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
                print("Metal Performance Shaders enabled for PyTorch acceleration")
        except ImportError:
            pass
            
        print("Optimizations enabled for Apple Silicon")
    
    # Always return True so this can be used in an if statement
    return True

def process_audio_streaming_parallel(input_file, output_file, target_lufs=-16.0, 
                                    true_peak_limit=-1.0, lra_max=9.0, 
                                    num_processes=None, chunk_size=10.0, use_cache=True):
    """
    Enhanced parallel streaming audio processor with better CPU and Metal utilization.
    """
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)
        
    # Initialize Metal acceleration if available
    metal_available = setup_apple_silicon_optimizations()
    
    # Create a process pool that will be reused across different processing stages
    with Pool(processes=num_processes) as pool:
        # Perform analysis (already optimized in your code)
        lra_info = analyze_lra_streaming_optimized(input_file, rate, meter, lra_max)
        
        # Launch a parallel pipeline of processing stages
        # Each stage can be executed in parallel if they don't depend on each other
        pipeline_tasks = []
        
        # Compression stage (if needed)
        if lra_max > 0 and lra_info['lra'] > lra_max:
            compression_task = pool.apply_async(
                apply_compression_streaming,
                (input_file, temp1_path, lra_info['threshold'], lra_info['ratio'], 
                 meter, chunk_size, lra_max)
            )
            pipeline_tasks.append(('compression', compression_task))
        
        # Block until compression is done (gain stage depends on this)
        current_file = input_file
        for task_name, task in pipeline_tasks:
            task.wait()
            if task_name == 'compression':
                current_file = temp1_path
        
        # Calculate gain for LUFS normalization
        gain = target_lufs - lra_info['loudness']
        
        # Apply gain (can run in parallel with previous stage)
        gain_task = pool.apply_async(
            apply_gain_streaming,
            (current_file, temp2_path, gain, chunk_size)
        )
        
        # Wait for gain to complete
        gain_task.wait()
        current_file = temp2_path
        
        # Measure true peak (can be parallelized internally)
        true_peak = measure_true_peak_streaming_parallel(current_file, chunk_size, num_processes)
        
        # Apply limiting if needed (final stage)
        if true_peak > true_peak_limit:
            apply_limiter_streaming(current_file, output_file, true_peak_limit, 0.050, chunk_size)
        else:
            copy_audio_streaming(current_file, output_file, chunk_size)
        
        # Final measurements can run in parallel
        loudness_task = pool.apply_async(measure_loudness_streaming, (output_file, meter, chunk_size))
        peak_task = pool.apply_async(measure_true_peak_streaming_parallel, 
                                   (output_file, chunk_size, num_processes))
        
        # Get final measurements
        final_loudness = loudness_task.get()
        final_tp = peak_task.get()

def enable_simd_optimizations():
    """Enable CPU SIMD vectorization optimizations"""
    import os
    
    # Use AVX2/SSE on Intel or NEON on ARM
    os.environ['NPY_ENABLE_AVX2'] = '1'
    os.environ['NPY_ENABLE_SSE41'] = '1'
    os.environ['NPY_ENABLE_SSE42'] = '1'
    
    # Set numpy threading options
    os.environ['NPY_NUM_THREADS'] = str(max(1, cpu_count() - 1))
    
    # Try to import numpy with optimizations enabled
    try:
        import numpy as np
        np.__config__.show()
    except:
        pass

if __name__ == "__main__":
    freeze_support()
    install_dependencies()
    check_optimizations()        # Show optimization status
    main()