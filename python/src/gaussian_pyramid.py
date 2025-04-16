import numpy as np
import tqdm

# Removed unused pyrUp import, kept pyrDown
from processing import idealTemporalBandpassFilter, pyrDown, rgb2yiq, pyrUp # Keep pyrUp needed later

# Modified function signature
def generateGaussianPyramid(image, kernel, level, frame_index, output_dir, save_func):
    """Generates the Gaussian pyramid representation used in EVM (downsample then upsample)."""
    image_shape = [image.shape[:2]]
    downsampled_image = image.copy()

    for curr_level in range(level):
        downsampled_result = pyrDown(image=downsampled_image, kernel=kernel)
        # --- Added Save Logic ---
        if frame_index == 0 and curr_level == 0 and save_func is not None and output_dir is not None:
             save_func(downsampled_result, f"frame_{frame_index}_gaussian_pyrdown_level0.txt", output_dir)
        # --- End Added Save Logic ---
        if downsampled_result is None or downsampled_result.size == 0:
             print(f"Warning: pyrDown returned None or empty array at level {curr_level} for frame {frame_index}. Stopping pyramid generation.")
             return None # Return None if downsampling fails

        downsampled_image = downsampled_result # Update for next iteration or final pyramid base
        image_shape.append(downsampled_image.shape[:2])

    # This is the smallest level of the pyramid before upsampling starts
    gaussian_pyramid_base = downsampled_image

    # Upsample back to original size (or close to it)
    reconstructed_pyramid = gaussian_pyramid_base
    for up_level in range(level):
         # Target shape is the shape from one level down in the pyramid generation
         target_shape = image_shape[level - up_level - 1]
         reconstructed_pyramid = pyrUp(
                             image=reconstructed_pyramid,
                             kernel=kernel,
                             dst_shape=target_shape
                         )
         if reconstructed_pyramid is None or reconstructed_pyramid.size == 0:
              print(f"Warning: pyrUp returned None or empty array at up_level {up_level} for frame {frame_index}. Returning partial reconstruction.")
              # Decide how to handle this - return None or the partial result?
              # Returning None might be safer if subsequent steps expect a full pyramid.
              return None


    # Ensure final shape matches original input image shape if possible
    # This might be needed if pyrUp doesn't perfectly restore dimensions
    if reconstructed_pyramid.shape[:2] != image_shape[0]:
        print(f"Warning: Final reconstructed Gaussian pyramid shape {reconstructed_pyramid.shape[:2]} differs from original YIQ frame shape {image_shape[0]} for frame {frame_index}. Resizing.")
        import cv2 # Local import for resize
        reconstructed_pyramid = cv2.resize(reconstructed_pyramid, (image_shape[0][1], image_shape[0][0]))


    return reconstructed_pyramid


# Modified function signature - though this function isn't used by generate_test_data.py currently
def getGaussianPyramids(images, kernel, level, output_dir, save_func):
    gaussian_pyramids = np.zeros_like(images, dtype=np.float32)

    for i in tqdm.tqdm(range(images.shape[0]),
                       ascii=True,
                       desc='Gaussian Pyramids Generation'):

        yiq_image = rgb2yiq(images[i])
        # Pass frame index, output_dir, and save_func
        pyramid_result = generateGaussianPyramid(
                                    image=yiq_image,
                                    kernel=kernel,
                                    level=level,
                                    frame_index=i,
                                    output_dir=output_dir,
                                    save_func=save_func
                        )
        if pyramid_result is not None:
             # Handle potential shape mismatch if resize occurred
             if pyramid_result.shape != gaussian_pyramids[i].shape:
                  print(f"Warning: Shape mismatch storing Gaussian pyramid for frame {i}. Expected {gaussian_pyramids[i].shape}, got {pyramid_result.shape}. Attempting resize.")
                  import cv2 # Local import
                  try:
                       gaussian_pyramids[i] = cv2.resize(pyramid_result, (gaussian_pyramids[i].shape[1], gaussian_pyramids[i].shape[0]))
                  except Exception as e:
                       print(f"Error resizing pyramid for frame {i}: {e}. Skipping frame.")
                       # Or handle differently, e.g., store None, skip frame
             else:
                  gaussian_pyramids[i] = pyramid_result
        # else: Handle case where pyramid generation failed (e.g., gaussian_pyramids[i] = None or skip)

    return gaussian_pyramids


def filterGaussianPyramids(pyramids,
                           fps,
                           freq_range,
                           alpha,
                           attenuation):

    # --- Added Save Logic for Filtered Signal (if this function were used) ---
    # Assuming 'pyramids' is the input batch (T, H, W, C)
    # The filtering happens here:
    filtered_signal_batch = idealTemporalBandpassFilter(
                            images=pyramids,
                            fps=fps,
                            freq_range=freq_range
                        ).astype(np.float32)

    # If we needed to save frame 0's raw filtered signal:
    # if save_func is not None and output_dir is not None and filtered_signal_batch.shape[0] > 0:
    #     save_func(filtered_signal_batch[0], "frame_0_gaussian_filtered_raw_yiq.txt", output_dir)
    # --- End Added Save Logic ---


    filtered_pyramids = filtered_signal_batch # Use the result from bandpass

    filtered_pyramids *= alpha
    filtered_pyramids[:, :, :, 1:] *= attenuation

    return filtered_pyramids
