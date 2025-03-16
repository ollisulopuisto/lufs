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
if platform.system() == 'Darwin' and platform.machine() == 'arm64':
    import os
    os.environ['ACCELERATE'] = '1'  # Use Accelerate framework

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
                        data = apply_multi_stage_compression(data, rate, current_lra, lra_max)
                        
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
        oversampled_data = resampy.resample(data, rate, rate * 4)
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
            
            oversampled_chunk = resampy.resample(chunk, rate, rate * 4)
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
    
    Args:
        input_file: Input audio file path
        output_file: Output audio file path
        target_lufs: Target LUFS loudness
        true_peak_limit: Maximum true peak level in dBTP
        lra_max: Maximum loudness range in LU
        num_processes: Number of processes to use for parallel processing
        chunk_size: Size of each processing chunk in seconds
        use_cache: Whether to cache loudness analysis results
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
        lra_info = analyze_lra_streaming(input_file, rate, meter, lra_max)
        
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
                chunk_seconds=chunk_size
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
        true_peak = measure_true_peak_streaming(current_file, chunk_size)
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
        final_tp = measure_true_peak_streaming(output_file, chunk_size)
        print(f"Final measurements: {final_loudness:.2f} LUFS, {final_tp:.2f} dBTP")
        
        return "Processing completed successfully"
        
    finally:
        # Clean up temporary files
        for path in [temp1_path, temp2_path]:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file {path}: {e}")

def process_stage(stage):
    """Process a single stage in the audio pipeline"""
    return stage['function'](stage['input'], stage['output'], **stage['args'])

def analyze_lra_streaming(input_file, rate, meter, lra_max):
    """
    Analyze LRA in a memory-efficient way by processing chunks
    """
    # Window and hop size for short-term loudness calculation
    window_size = int(3 * rate)  # 3 seconds
    hop_size = int(0.1 * rate)   # 100ms
    
    # Create loudness history buffer
    st_loudness = []
    
    # Process file in chunks
    chunk_samples = int(5.0 * rate)  # 5 seconds chunks for reading
    
    with sf.SoundFile(input_file, 'r') as f:
        # Read chunks and compute loudness
        with tqdm(total=f.frames, desc="Analyzing loudness", unit="samples") as pbar:
            while f.tell() < f.frames:
                chunk = f.read(chunk_samples)
                if len(chunk) == 0:
                    break
                    
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

def apply_compression_streaming(input_file, output_file, threshold, ratio, meter, chunk_seconds=5.0):
    """Apply compression in streaming fashion"""
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Calculate threshold in linear domain
        threshold_linear = 10 ** (threshold / 20.0)
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Process in chunks
            chunk_size = int(rate * chunk_seconds)
            
            # Set up a buffer for overlap processing
            overlap_size = int(rate * 0.1)  # 100ms overlap
            overlap_buffer = None
            
            with tqdm(total=infile.frames, desc="Applying compression", unit="samples") as pbar:
                while infile.tell() < infile.frames:
                    # Read chunk
                    chunk = infile.read(chunk_size)
                    if len(chunk) == 0:
                        break
                        
                    # Combine with previous overlap
                    if overlap_buffer is not None:
                        chunk = np.concatenate((overlap_buffer, chunk))
                    
                    # Save data for next overlap
                    if len(chunk) > overlap_size:
                        overlap_buffer = chunk[-overlap_size:]
                    else:
                        overlap_buffer = None
                    
                    # Apply compression
                    abs_data = np.abs(chunk)
                    gain_reduction = np.ones_like(chunk)
                    mask = abs_data > threshold_linear
                    
                    if np.any(mask):
                        gain_reduction[mask] = (threshold_linear + 
                                              ((abs_data[mask] - threshold_linear) / ratio)) / abs_data[mask]
                        
                    processed_chunk = chunk * gain_reduction
                    
                    # Write processed chunk (excluding overlap for all but the last chunk)
                    write_size = len(processed_chunk) if infile.tell() >= infile.frames else len(processed_chunk) - overlap_size
                    outfile.write(processed_chunk[:write_size])
                    
                    pbar.update(len(chunk))
                    
            # Write final overlap if there is any
            if overlap_buffer is not None:
                outfile.write(overlap_buffer)
    
    return "Compression completed"

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
    """Apply true peak limiting in a streaming fashion"""
    from scipy import signal
    
    with sf.SoundFile(input_file, 'r') as infile:
        rate = infile.samplerate
        channels = infile.channels
        
        # Create output file with same properties
        with sf.SoundFile(output_file, 'w', samplerate=rate, 
                          channels=channels, subtype=infile.subtype) as outfile:
            
            # Calculate parameters
            threshold_linear = 10 ** (true_peak_limit / 20.0)
            release_samples = int(release_time * rate)
            
            # Create release curve
            release_curve = np.exp(-np.arange(release_samples) / (release_samples / 5))
            release_curve = release_curve / np.sum(release_curve)
            
            # Process in overlapping chunks for smooth transitions
            chunk_size = int(rate * chunk_seconds)
            overlap_size = max(release_samples * 2, int(rate * 0.2))  # Max of 2x release or 200ms
            
            # State variables for filter
            gain_state = None
            
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
                    
                    # Calculate gain reduction curve
                    abs_data = np.abs(chunk)
                    gain_reduction = np.ones_like(chunk)
                    mask = abs_data > threshold_linear
                    if np.any(mask):
                        gain_reduction[mask] = threshold_linear / abs_data[mask]
                    
                    # Apply smoothing filter to gain reduction
                    if len(chunk.shape) > 1:  # Multi-channel audio
                        for c in range(channels):
                            gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[:, c][::-1])[::-1]
                            gain_reduction[:, c] = np.minimum(gain_reduction[:, c], gain_reduction_smooth)
                    else:  # Mono audio
                        gain_reduction_smooth = signal.lfilter(release_curve, [1.0], gain_reduction[::-1])[::-1]
                        gain_reduction = np.minimum(gain_reduction, gain_reduction_smooth)
                    
                    # Apply gain reduction
                    limited_chunk = chunk * gain_reduction
                    
                    # Write only the non-overlapping part except for last chunk
                    write_size = min(chunk_size, len(limited_chunk))
                    outfile.write(limited_chunk[:write_size])
                    
                    # Advance position
                    pos += write_size
                    pbar.update(write_size)
    
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
                oversampled_chunk = resampy.resample(chunk, rate, rate * 4)
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
                tensor_data = torch.tensor(data, device=device)
                # Process with torch's resampling (implementation required)
                # ... 
                return result.cpu().numpy()
        except:
            pass
            
    # Fall back to resampy if torch isn't available
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

def check_optimizations():
    """Check if running optimized libraries for current architecture"""
    import numpy as np
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
    main()

def apply_multi_stage_compression(data, rate, current_lra, target_lra):
    """
    Apply multi-stage compression to reduce LRA in a more natural way
    """
    # Calculate how many stages we need (1 LU per stage)
    lra_to_reduce = current_lra - target_lra
    num_stages = min(8, max(1, int(np.ceil(lra_to_reduce))))
    
    print(f"Using {num_stages} compression stages to reduce LRA by {lra_to_reduce:.1f} LU")
    
    # Metadata to track changes
    meter = pyln.Meter(rate)
    original_loudness = meter.integrated_loudness(data)
    
    # Set up compressor parameters for each stage
    stage_reduction = lra_to_reduce / num_stages
    processed_data = data.copy()
    
    for stage in range(num_stages):
        print(f"Stage {stage+1}/{num_stages}: Reducing LRA by {stage_reduction:.1f} LU")
        
        # Create a different threshold and ratio for each stage
        if stage < num_stages/2:
            # Early stages focus on peaks (higher threshold)
            percentile = 60 + (20 * stage / (num_stages/2))
        else:
            # Later stages focus on body (lower threshold)
            percentile = 40 - (20 * (stage - num_stages/2) / (num_stages/2))
        
        # Calculate compression parameters
        loudness_values = []
        window_size = int(1 * rate)  # 1-second window
        hop_size = int(0.2 * rate)   # 200ms hop
        
        # Get some loudness values to determine threshold
        for i in range(0, min(len(processed_data) - window_size, window_size * 50), hop_size):
            window_data = processed_data[i:i + window_size]
            window_loudness = meter.integrated_loudness(window_data)
            if window_loudness > -70:  # Ignore silence
                loudness_values.append(window_loudness)
        
        if not loudness_values:
            continue
            
        loudness_values.sort()
        threshold_idx = int(len(loudness_values) * (percentile/100))
        threshold_loudness = loudness_values[threshold_idx]
        
        # Convert to linear domain
        threshold_linear = 10 ** (threshold_loudness/20)
        
        # Use a gentler ratio for more transparent compression
        ratio = 1.2 + (0.3 * stage)  # Gradually increase ratio
        
        # Apply attack and release (smoother than current implementation)
        attack_time = 0.005 + (0.015 * stage / num_stages)  # 5-20ms
        release_time = 0.050 + (0.150 * stage / num_stages)  # 50-200ms
        
        # Apply this stage of compression
        processed_data = apply_compressor_with_time_constants(
            processed_data, 
            rate, 
            threshold_linear,
            ratio,
            attack_time,
            release_time
        )
        
        # Re-measure stage progress
        stage_loudness = meter.integrated_loudness(processed_data)
        
        # Maintain target loudness
        gain_adjust = original_loudness - stage_loudness
        processed_data = processed_data * (10**(gain_adjust/20))
        
    return processed_data

def apply_compressor_with_time_constants(data, rate, threshold, ratio, attack_time, release_time):
    """
    Apply compression with proper attack and release time constants
    """
    from scipy import signal
    
    # Calculate time constants in samples
    attack_samples = int(attack_time * rate)
    release_samples = int(release_time * rate)
    
    # Create attack/release filters
    attack_curve = 1 - np.exp(np.arange(attack_samples) / (-attack_samples/2))
    release_curve = np.exp(np.arange(release_samples) / (-release_samples/5))
    
    # Calculate static gain reduction
    abs_data = np.abs(data)
    gain_reduction = np.ones_like(abs_data)
    mask = abs_data > threshold
    
    if np.any(mask):
        # Apply static gain reduction formula
        gain_reduction[mask] = (threshold + ((abs_data[mask] - threshold) / ratio)) / abs_data[mask]
        
        # Apply time constants
        if len(data.shape) > 1:  # Multi-channel
            for c in range(data.shape[1]):
                smoothed = np.ones_like(gain_reduction[:, c])
                # Apply time constants with a state-machine approach
                for i in range(1, len(gain_reduction[:, c])):
                    if gain_reduction[i, c] < smoothed[i-1]:  # More reduction (attack)
                        smoothed[i] = attack_curve[0] * gain_reduction[i, c] + (1 - attack_curve[0]) * smoothed[i-1]
                    else:  # Less reduction (release)
                        smoothed[i] = release_curve[0] * gain_reduction[i, c] + (1 - release_curve[0]) * smoothed[i-1]
                gain_reduction[:, c] = smoothed
        else:  # Mono
            smoothed = np.ones_like(gain_reduction)
            for i in range(1, len(gain_reduction)):
                if gain_reduction[i] < smoothed[i-1]:  # More reduction (attack)
                    smoothed[i] = attack_curve[0] * gain_reduction[i] + (1 - attack_curve[0]) * smoothed[i-1]
                else:  # Less reduction (release)
                    smoothed[i] = release_curve[0] * gain_reduction[i] + (1 - release_curve[0]) * smoothed[i-1]
            gain_reduction = smoothed
    
    # Apply smoothed gain reduction
    return data * gain_reduction
