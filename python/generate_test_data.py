import numpy as np
import os
import sys
from pathlib import Path
import cv2 # Needed for resize in case of shape mismatch

# Ensure the src directory is in the Python path
# Assuming the script is run from the 'evmpy' directory
script_dir = Path(__file__).parent.resolve()
src_dir = script_dir / 'src'
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

try:
    from processing import loadVideo, rgb2yiq, yiq2rgb, pyrDown, pyrUp, reconstructGaussianImage # Added yiq2rgb
    from constants import gaussian_kernel
    from laplacian_pyramid import filterLaplacianPyramids # Import the filter function
    from gaussian_pyramid import getGaussianPyramids, filterGaussianPyramids, generateGaussianPyramid # Import Gaussian functions
    from scipy.signal import butter # Import butter
except ImportError as e:
    print(f"Error importing modules. Make sure you run this script from the 'evmpy' directory or have 'src' in PYTHONPATH.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    print(f"Import Error: {e}")
    sys.exit(1)

# --- Configuration ---
VIDEO_PATH = "data/face.mp4" # Path relative to the script's directory (evmpy)
OUTPUT_DIR = "../evmcpp/tests/data" # Relative path to C++ test data dir
NUM_FRAMES_TO_PROCESS = 5 # Need a few frames for temporal filter to work
TEST_PYRAMID_LEVELS = 4 # Matches laplacian test
# Parameters matching laplacian_pyramid.py usage for filtering
TEST_FPS = 30.0
TEST_FREQ_RANGE = [0.4, 3.0] # Example range (Hz)
TEST_ALPHA = 10.0 # Example magnification factor
TEST_LAMBDA_CUTOFF = 16.0 # Example spatial cutoff
TEST_ATTENUATION = 1.0 # Example chrominance attenuation

# --- Helper Function ---
def save_array_to_txt(array, filename, output_dir):
    """Saves a numpy array to a text file, reshaping 3D/1D arrays."""
    filepath = Path(output_dir) / filename
    try:
        if array is None:
             print(f"Warning: Skipping save for None array for file {filename}")
             return
        if array.ndim == 3:
            # Reshape (H, W, C) to (H, W*C) for savetxt
            reshaped_array = array.reshape(array.shape[0], -1)
            np.savetxt(filepath, reshaped_array, fmt='%.18e', delimiter=',')
            print(f"Saved reshaped array (shape {array.shape} -> {reshaped_array.shape}) to {filepath}")
        elif array.ndim == 2:
            np.savetxt(filepath, array, fmt='%.18e', delimiter=',')
            print(f"Saved array (shape {array.shape}) to {filepath}")
        elif array.ndim == 1: # Handle 1D arrays as well
            # Save 1D array as a single row
            np.savetxt(filepath, array.reshape(1, -1), fmt='%.18e', delimiter=',')
            print(f"Saved array (shape {array.shape}) to {filepath}")
        else:
             print(f"Warning: Skipping save for array with unsupported ndim={array.ndim} for file {filename}")

    except Exception as e:
        print(f"Error saving array to {filepath}: {e}")

# --- Main Logic ---
def main():
    print("Starting reference data generation...")

    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path.resolve()}")

    # Construct absolute path to video
    absolute_video_path = script_dir / VIDEO_PATH
    print(f"Loading video: {absolute_video_path}")
    try:
        original_images, fps = loadVideo(str(absolute_video_path)) # Pass absolute path as string
        # Override FPS with test value if needed, though they should match
        if fps != TEST_FPS:
            print(f"Warning: Video FPS ({fps}) differs from TEST_FPS ({TEST_FPS}). Using TEST_FPS.")
            fps = TEST_FPS
    except Exception as e:
        print(f"Error loading video '{absolute_video_path}': {e}")
        sys.exit(1)

    if original_images is None or len(original_images) == 0:
        print(f"No frames loaded from video: {VIDEO_PATH}")
        sys.exit(1)

    print(f"Loaded {len(original_images)} frames at {fps} FPS.")
    num_frames = min(NUM_FRAMES_TO_PROCESS, len(original_images))
    print(f"Processing first {num_frames} frames for pyramid generation...")

    # Use the defined Gaussian kernel
    kernel = gaussian_kernel
    laplacian_pyramids_batch = [] # To store Laplacian pyramids for all frames
    # gaussian_pyramids_batch = [] # To store Gaussian pyramids for all frames (Removed, using reconstructed batch now)
    gaussian_reconstructed_batch = [] # To store the spatially reconstructed Gaussian frames for temporal filtering

    yiq_frame0_original = None # To store the original YIQ of frame 0
    for i in range(num_frames):
        print(f"\n--- Processing Frame {i} ---")
        rgb_frame = original_images[i]
        save_array_to_txt(rgb_frame, f"frame_{i}_rgb.txt", OUTPUT_DIR)

        # 1. rgb2yiq
        try:
            yiq_frame = rgb2yiq(rgb_frame)
            save_array_to_txt(yiq_frame, f"frame_{i}_yiq.txt", OUTPUT_DIR)
            # --- Save Step 2 Output for Frame 0 ---
            if i == 0:
                save_array_to_txt(yiq_frame, "frame_0_step2_yiq.txt", OUTPUT_DIR)
                yiq_frame0_original = yiq_frame.copy() # Store original YIQ for frame 0
        except Exception as e:
            print(f"Error during rgb2yiq for frame {i}: {e}")
            continue # Skip to next frame if error

        # 2. Generate Laplacian Pyramid and save levels
        print(f"Generating Laplacian pyramid (levels={TEST_PYRAMID_LEVELS})...")
        laplacian_pyramid_levels = []
        prev_image = yiq_frame.copy()
        valid_pyramid = True

        for lvl in range(TEST_PYRAMID_LEVELS):
            try:
                downsampled_image = pyrDown(image=prev_image, kernel=kernel)
                if downsampled_image is None or downsampled_image.size == 0:
                     print(f"Warning: pyrDown produced empty image at level {lvl}. Stopping pyramid generation for frame {i}.")
                     valid_pyramid = False
                     break

                target_shape = prev_image.shape[:2]
                upsampled_image = pyrUp(image=downsampled_image, kernel=kernel, dst_shape=target_shape)
                if upsampled_image is None or upsampled_image.size == 0:
                     print(f"Warning: pyrUp produced empty image at level {lvl}. Stopping pyramid generation for frame {i}.")
                     valid_pyramid = False
                     break

                # Ensure shapes match for subtraction, resize if necessary (OpenCV might handle slightly differently)
                if upsampled_image.shape != prev_image.shape:
                     print(f"Warning: Resizing pyrUp output at level {lvl} from {upsampled_image.shape} to {prev_image.shape}")
                     upsampled_image = cv2.resize(upsampled_image, (prev_image.shape[1], prev_image.shape[0]))

                laplacian_level = prev_image - upsampled_image
                laplacian_pyramid_levels.append(laplacian_level)
                save_array_to_txt(laplacian_level, f"frame_{i}_laplacian_level_{lvl}.txt", OUTPUT_DIR)

                prev_image = downsampled_image # For next iteration

            except Exception as e:
                print(f"Error during Laplacian pyramid generation at level {lvl} for frame {i}: {e}")
                valid_pyramid = False
                break # Stop generating levels for this frame on error

        if valid_pyramid:
            laplacian_pyramids_batch.append(laplacian_pyramid_levels) # Store the generated pyramid only if valid
        else:
             laplacian_pyramids_batch.append(None) # Add placeholder if pyramid generation failed

        # --- Gaussian Pathway ---
        print(f"Generating Gaussian pyramid representation (levels={TEST_PYRAMID_LEVELS})...") # Updated print statement
        try:
            # 1. Generate Reconstructed Gaussian Image (Spatially Reconstructed Pyramid)
            # Pass frame index, output dir, and save function
            reconstructed_gaussian_frame = generateGaussianPyramid(
                image=yiq_frame,
                kernel=kernel,
                level=TEST_PYRAMID_LEVELS,
                frame_index=i, # Pass frame index
                output_dir=OUTPUT_DIR, # Pass output directory
                save_func=save_array_to_txt # Pass save function
            )
            valid_gaussian_output = True # Assume true initially

            # 2. Save Reconstructed Gaussian Image (Original Goal) and Intermediate Frame 0 Data
            if reconstructed_gaussian_frame is not None and reconstructed_gaussian_frame.size > 0:
                 # Save the standard output (spatially reconstructed pyramid)
                 save_array_to_txt(reconstructed_gaussian_frame, f"frame_{i}_gaussian_reconstructed.txt", OUTPUT_DIR)
                 # --- Save Step 3 Output for Frame 0 (Spatially Filtered YIQ - Gaussian Reconstruction) ---
                 if i == 0:
                      save_array_to_txt(reconstructed_gaussian_frame, "frame_0_step3_spatial_filtered_yiq.txt", OUTPUT_DIR)

                 # --- Removed incorrect save logic for frame 0 ---
                 # The correct save will happen after temporal filtering below.
            else:
                 print(f"Warning: generateGaussianPyramid returned None or empty array for frame {i}.")
                 valid_gaussian_output = False # Mark as invalid if generation failed

            # Note: The third requested file, frame_0_reconstructed_yiq.txt (YIQ after adding filtered signal),
            # cannot be generated with the current script structure, as the temporal filtering and
            # final addition step (like in reconstructGaussianImage) are not performed here for the Gaussian path.

            # Collect the valid reconstructed frame for batch temporal filtering
            if valid_gaussian_output and reconstructed_gaussian_frame is not None:
                 gaussian_reconstructed_batch.append(reconstructed_gaussian_frame)
            else:
                 gaussian_reconstructed_batch.append(None) # Keep alignment if a frame failed

        except Exception as e:
            print(f"Error during Gaussian pyramid generation/reconstruction for frame {i}: {e}")
            # gaussian_pyramids_batch.append(None) # Not collecting batch anymore
    # End of the main frame processing loop (for i in range(num_frames))

    # --- Generate Butterworth Coefficients --- # Moved inside main
    print("\n--- Generating Butterworth Coefficients ---")
    print(f"Params: Fs={TEST_FPS}, Freq Range={TEST_FREQ_RANGE}, Order=1")
    try:
        # Low-pass component
        b_low, a_low = butter(1, TEST_FREQ_RANGE[0], btype='low', output='ba', fs=TEST_FPS)
        save_array_to_txt(b_low, "butter_low_b.txt", OUTPUT_DIR)
        save_array_to_txt(a_low, "butter_low_a.txt", OUTPUT_DIR)
        print(f"Low-pass coeffs: b={b_low}, a={a_low}")

        # High-pass component (using lowpass design as per python code)
        b_high, a_high = butter(1, TEST_FREQ_RANGE[1], btype='low', output='ba', fs=TEST_FPS)
        save_array_to_txt(b_high, "butter_high_b.txt", OUTPUT_DIR)
        save_array_to_txt(a_high, "butter_high_a.txt", OUTPUT_DIR)
        print(f"High-pass coeffs: b={b_high}, a={a_high}")

    except Exception as e:
        print(f"Error generating Butterworth coefficients: {e}")


    # --- Apply Temporal Filter and Save Results --- # Moved inside main
    # Filter out None entries from pyramids_batch before converting to NumPy array
    valid_laplacian_pyramids_batch = [p for p in laplacian_pyramids_batch if p is not None]

    if len(valid_laplacian_pyramids_batch) >= 2: # Need at least 2 valid frames for filter
        print("\n--- Applying Laplacian Temporal Filter ---")
        try:
            # Assuming all valid pyramids have the same number of levels
            laplacian_pyramids_np = np.array(valid_laplacian_pyramids_batch, dtype=object)
            print(f"Shape of Laplacian pyramids batch before filtering: {laplacian_pyramids_np.shape}")

            # Determine the actual number of levels from the first valid pyramid
            actual_laplacian_levels = len(valid_laplacian_pyramids_batch[0]) if valid_laplacian_pyramids_batch else 0

            if actual_laplacian_levels > 0:
                filtered_laplacian_pyramids_np = filterLaplacianPyramids(
                    pyramids=laplacian_pyramids_np,
                    level=actual_laplacian_levels, # Use actual number of levels
                    fps=TEST_FPS,
                    freq_range=TEST_FREQ_RANGE,
                    alpha=TEST_ALPHA,
                    lambda_cutoff=TEST_LAMBDA_CUTOFF,
                    attenuation=TEST_ATTENUATION
                )
                print(f"Shape of Laplacian pyramids batch after filtering: {filtered_laplacian_pyramids_np.shape}")

                # Save the filtered levels for each frame that was filtered
                num_filtered_laplacian_frames = filtered_laplacian_pyramids_np.shape[0]
                num_filtered_laplacian_levels = filtered_laplacian_pyramids_np.shape[1]

                # Find original indices of the frames that were filtered
                original_laplacian_indices = [idx for idx, p in enumerate(laplacian_pyramids_batch) if p is not None]

                for frame_idx_filtered, original_frame_idx in enumerate(original_laplacian_indices):
                     if frame_idx_filtered < num_filtered_laplacian_frames: # Ensure index is valid
                         for lvl in range(num_filtered_laplacian_levels):
                             save_array_to_txt(filtered_laplacian_pyramids_np[frame_idx_filtered, lvl], f"frame_{original_frame_idx}_filtered_level_{lvl}.txt", OUTPUT_DIR)
                     else:
                          print(f"Warning: Index mismatch when saving filtered Laplacian frame {original_frame_idx}")

            else:
                 print("Skipping Laplacian filtering as no valid pyramid levels were generated.")

        except Exception as e:
            print(f"Error during Laplacian temporal filtering or saving filtered results: {e}")
    else:
        print(f"\nSkipping Laplacian temporal filtering (need at least 2 valid frames, found {len(valid_laplacian_pyramids_batch)}).")

    # --- Gaussian Temporal Filter and Saving --- # Moved inside main
    valid_gaussian_reconstructed_batch = [f for f in gaussian_reconstructed_batch if f is not None]

    if len(valid_gaussian_reconstructed_batch) >= 2: # Need at least 2 valid frames for filter
        print("\n--- Applying Gaussian Temporal Filter ---")
        try:
            gaussian_reconstructed_np = np.array(valid_gaussian_reconstructed_batch) # Convert list of frames to numpy array
            print(f"Shape of Gaussian reconstructed batch before filtering: {gaussian_reconstructed_np.shape}")

            # Assuming filterGaussianPyramids takes the batch and filtering parameters
            # Note: It might not need level or lambda_cutoff if working on reconstructed images
            filtered_gaussian_frames_np = filterGaussianPyramids(
                pyramids=gaussian_reconstructed_np, # Pass the batch of reconstructed frames
                fps=TEST_FPS,
                freq_range=TEST_FREQ_RANGE,
                alpha=TEST_ALPHA,
                attenuation=TEST_ATTENUATION
                # level and lambda_cutoff might not be needed here, adjust if filter function requires them
            )
            print(f"Shape of Gaussian frames batch after filtering: {filtered_gaussian_frames_np.shape}")

            # Save the temporally filtered YIQ data for the first frame
            if filtered_gaussian_frames_np.shape[0] > 0:
                 # Find the index of the first valid frame in the original batch
                 first_valid_original_index = -1
                 for idx, frame in enumerate(gaussian_reconstructed_batch):
                     if frame is not None:
                         first_valid_original_index = idx
                         break

                 if first_valid_original_index == 0: # Only save if the *first* frame was successfully processed and filtered
                     # --- Save Step 4 Output for Frame 0 (Temporally Filtered YIQ) ---
                     filtered_yiq_frame0_temporal = filtered_gaussian_frames_np[0] # Assign to variable for clarity
                     save_array_to_txt(filtered_yiq_frame0_temporal, "frame_0_step4_temporal_filtered_yiq.txt", OUTPUT_DIR)
                     print(f"Saved Step 4 (Temporally Filtered YIQ) for frame 0.")
                     # --- Save Step 5 Output for Frame 0 (Amplified Filtered YIQ) ---
                     # Note: In this Gaussian path, amplification is part of the temporal filter output
                     save_array_to_txt(filtered_yiq_frame0_temporal, "frame_0_step5_amplified_filtered_yiq.txt", OUTPUT_DIR)
                     print(f"Saved Step 5 (Amplified Filtered YIQ) for frame 0.")
                     # --- Calculate and Save Step 6b Output for Frame 0 (Combined YIQ) ---
                     if yiq_frame0_original is not None:
                         # Ensure shapes match before adding (should match if processing was consistent)
                         if yiq_frame0_original.shape == filtered_yiq_frame0_temporal.shape:
                             combined_yiq_frame0 = yiq_frame0_original + filtered_yiq_frame0_temporal
                             save_array_to_txt(combined_yiq_frame0, "frame_0_step6b_combined_yiq.txt", OUTPUT_DIR)
                             print(f"Saved Step 6b (Combined YIQ = Original YIQ + Amplified Filtered YIQ) for frame 0.")
                         else:
                             print(f"Error: Shape mismatch between original YIQ ({yiq_frame0_original.shape}) and filtered YIQ ({filtered_yiq_frame0_temporal.shape}). Cannot calculate Step 6b.")
                     else:
                         print("Error: Original YIQ for frame 0 not found. Cannot calculate Step 6b.")
                 elif first_valid_original_index > 0:
                     print(f"Warning: First frame (index 0) failed processing, cannot save frame_0_step4_temporal_filtered_yiq.txt. First valid frame was {first_valid_original_index}.")
                 else: # first_valid_original_index == -1
                     print(f"Warning: No valid frames found in Gaussian batch, cannot save frame_0_step4_temporal_filtered_yiq.txt.")

            else:
                 print("Warning: Gaussian temporal filtering produced an empty result.")

        except Exception as e:
            print(f"Error during Gaussian temporal filtering or saving filtered result: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
    else:
        print(f"\nSkipping Gaussian temporal filtering (need at least 2 valid frames, found {len(valid_gaussian_reconstructed_batch)}).")

    # --- Reconstruct and Save Final Steps for Frame 0 (Gaussian Path) ---
    if 'filtered_gaussian_frames_np' in locals() and \
       filtered_gaussian_frames_np is not None and \
       filtered_gaussian_frames_np.shape[0] > 0 and \
       'first_valid_original_index' in locals() and \
       first_valid_original_index == 0:

        print("\n--- Performing final RGB conversion steps for Frame 0 (Gaussian Path) ---")
        try:
            # Use the variable assigned earlier
            filtered_yiq_frame0 = filtered_yiq_frame0_temporal

            # Step 6c: YIQ -> RGB Float
            reconstructed_rgb_float = yiq2rgb(filtered_yiq_frame0)
            save_array_to_txt(reconstructed_rgb_float, "frame_0_step6c_reconstructed_rgb_float.txt", OUTPUT_DIR)
            print("Saved Step 6c (YIQ -> RGB Float) for frame 0.")

            # Step 6d: Clip Float RGB
            clipped_rgb_float = np.clip(reconstructed_rgb_float, 0, 255)
            save_array_to_txt(clipped_rgb_float, "frame_0_step6d_clipped_rgb_float.txt", OUTPUT_DIR)
            print("Saved Step 6d (Clipped RGB Float) for frame 0.")

            # Step 6e: Final uint8 RGB
            final_rgb_uint8 = clipped_rgb_float.astype(np.uint8)
            save_array_to_txt(final_rgb_uint8, "frame_0_step6e_final_rgb_uint8.txt", OUTPUT_DIR)
            print("Saved Step 6e (Final RGB uint8) for frame 0.")

        except Exception as e:
            print(f"Error during final reconstruction steps for frame 0: {e}")
            import traceback
            traceback.print_exc()
    else:
         print("\nSkipping final reconstruction steps for Frame 0 as prerequisites not met (filtered data unavailable or frame 0 failed).")


    print("\nReference data generation complete.")
# End of main() function

if __name__ == "__main__":
    main()