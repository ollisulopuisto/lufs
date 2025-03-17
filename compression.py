import soundfile as sf
import numpy as np
from tqdm import tqdm
from scipy import signal
from multiprocessing import Pool, cpu_count
import pyloudnorm as pyln # Assuming pyloudnorm is needed here
from loudness import measure_exact_lra # Assuming loudness.py is in the same directory

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
    # ... (your existing LRA measurement code) ...

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
