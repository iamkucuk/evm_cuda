#!/usr/bin/env python3
"""
Frame-by-frame PSNR analysis between CPU and CUDA Butterworth videos
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    
    # Convert to float to avoid overflow
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Calculate MSE
    mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return 100.0  # Use 100 dB instead of inf for perfect match
    
    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

def load_video_frames(video_path):
    """Load all frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError("Cannot open video: {}".format(video_path))
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("Loading {} frames from {} at {} FPS...".format(frame_count, os.path.basename(video_path), fps))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print("Loaded {} frames".format(len(frames)))
    
    return frames, fps

def analyze_frame_psnr(cpu_video_path, cuda_video_path, output_dir="./"):
    """Analyze frame-by-frame PSNR and create plots"""
    
    # Load videos
    print("=== Loading Videos ===")
    cpu_frames, cpu_fps = load_video_frames(cpu_video_path)
    cuda_frames, cuda_fps = load_video_frames(cuda_video_path)
    
    # Verify frame counts match
    if len(cpu_frames) != len(cuda_frames):
        print("Warning: Frame count mismatch - CPU: {}, CUDA: {}".format(len(cpu_frames), len(cuda_frames)))
        min_frames = min(len(cpu_frames), len(cuda_frames))
        cpu_frames = cpu_frames[:min_frames]
        cuda_frames = cuda_frames[:min_frames]
        print("Using first {} frames for comparison".format(min_frames))
    
    # Calculate PSNR for each frame
    print("\n=== Calculating Frame-by-Frame PSNR ===")
    frame_psnrs = []
    num_frames = len(cpu_frames)
    
    for i in range(num_frames):
        # Ensure frames have same size and type
        cpu_frame = cpu_frames[i]
        cuda_frame = cuda_frames[i]
        
        if cpu_frame.shape != cuda_frame.shape:
            print("Frame {}: Size mismatch, resizing CUDA frame to match CPU".format(i))
            cuda_frame = cv2.resize(cuda_frame, (cpu_frame.shape[1], cpu_frame.shape[0]))
        
        # Calculate PSNR
        psnr = calculate_psnr(cpu_frame, cuda_frame)
        frame_psnrs.append(psnr)
        
        if i % 50 == 0:
            print("  Frame {}/{}: PSNR = {:.2f} dB".format(i, num_frames, psnr))
    
    # Convert to numpy array for analysis
    frame_psnrs = np.array(frame_psnrs)
    
    # Find statistics
    avg_psnr = np.mean(frame_psnrs)
    min_psnr = np.min(frame_psnrs)
    max_psnr = np.max(frame_psnrs)
    std_psnr = np.std(frame_psnrs)
    min_frame_idx = np.argmin(frame_psnrs)
    
    print("\n=== PSNR Statistics ===")
    print("Average PSNR: {:.2f} dB".format(avg_psnr))
    print("Minimum PSNR: {:.2f} dB (Frame {})".format(min_psnr, min_frame_idx))
    print("Maximum PSNR: {:.2f} dB".format(max_psnr))
    print("Standard Deviation: {:.2f} dB".format(std_psnr))
    
    # Create time axis (in seconds)
    time_axis = np.arange(num_frames) / cpu_fps
    
    # Create plots
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Frame-by-Frame PSNR Analysis: CPU vs CUDA Butterworth', fontsize=16)
    
    # Plot 1: PSNR vs Frame Number
    axes[0, 0].plot(frame_psnrs, 'b-', linewidth=1, alpha=0.7)
    axes[0, 0].axhline(y=avg_psnr, color='r', linestyle='--', label=f'Average: {avg_psnr:.2f} dB')
    axes[0, 0].axhline(y=30, color='g', linestyle='--', label='Target: 30 dB')
    axes[0, 0].scatter([min_frame_idx], [min_psnr], color='red', s=100, zorder=5, 
                      label=f'Lowest: {min_psnr:.2f} dB (Frame {min_frame_idx})')
    axes[0, 0].set_xlabel('Frame Number')
    axes[0, 0].set_ylabel('PSNR (dB)')
    axes[0, 0].set_title('PSNR vs Frame Number')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: PSNR vs Time
    axes[0, 1].plot(time_axis, frame_psnrs, 'b-', linewidth=1, alpha=0.7)
    axes[0, 1].axhline(y=avg_psnr, color='r', linestyle='--', label=f'Average: {avg_psnr:.2f} dB')
    axes[0, 1].axhline(y=30, color='g', linestyle='--', label='Target: 30 dB')
    axes[0, 1].scatter([time_axis[min_frame_idx]], [min_psnr], color='red', s=100, zorder=5,
                      label=f'Lowest: {min_psnr:.2f} dB (t={time_axis[min_frame_idx]:.2f}s)')
    axes[0, 1].set_xlabel('Time (seconds)')
    axes[0, 1].set_ylabel('PSNR (dB)')
    axes[0, 1].set_title('PSNR vs Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: PSNR Histogram
    axes[1, 0].hist(frame_psnrs, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].axvline(x=avg_psnr, color='r', linestyle='--', label=f'Average: {avg_psnr:.2f} dB')
    axes[1, 0].axvline(x=30, color='g', linestyle='--', label='Target: 30 dB')
    axes[1, 0].set_xlabel('PSNR (dB)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('PSNR Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Moving Average PSNR (window size = 10 frames)
    window_size = 10
    moving_avg = np.convolve(frame_psnrs, np.ones(window_size)/window_size, mode='valid')
    moving_time = time_axis[window_size-1:]
    
    axes[1, 1].plot(time_axis, frame_psnrs, 'b-', alpha=0.3, label='Individual Frames')
    axes[1, 1].plot(moving_time, moving_avg, 'r-', linewidth=2, label=f'Moving Average ({window_size} frames)')
    axes[1, 1].axhline(y=avg_psnr, color='orange', linestyle='--', label=f'Overall Average: {avg_psnr:.2f} dB')
    axes[1, 1].axhline(y=30, color='g', linestyle='--', label='Target: 30 dB')
    axes[1, 1].set_xlabel('Time (seconds)')
    axes[1, 1].set_ylabel('PSNR (dB)')
    axes[1, 1].set_title(f'PSNR Smoothed (Moving Average)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'frame_psnr_analysis.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {plot_path}")
    
    # Save detailed frame analysis for lowest PSNR frames
    print(f"\n=== Detailed Analysis of Lowest PSNR Frames ===")
    
    # Find frames with PSNR below certain thresholds
    low_psnr_frames = []
    thresholds = [25, 30, 35]
    
    for threshold in thresholds:
        below_threshold = np.where(frame_psnrs < threshold)[0]
        if len(below_threshold) > 0:
            low_psnr_frames.extend(below_threshold)
            print(f"Frames below {threshold} dB: {len(below_threshold)} frames")
    
    # Get top 10 lowest PSNR frames
    lowest_indices = np.argsort(frame_psnrs)[:10]
    
    print(f"\nTop 10 Lowest PSNR Frames:")
    for i, frame_idx in enumerate(lowest_indices):
        print(f"  {i+1:2d}. Frame {frame_idx:3d}: {frame_psnrs[frame_idx]:.2f} dB (t={time_axis[frame_idx]:.2f}s)")
    
    # Save frame analysis to text file
    analysis_path = os.path.join(output_dir, 'frame_psnr_analysis.txt')
    with open(analysis_path, 'w') as f:
        f.write("Frame-by-Frame PSNR Analysis: CPU vs CUDA Butterworth\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Frames: {num_frames}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Minimum PSNR: {min_psnr:.2f} dB (Frame {min_frame_idx})\n")
        f.write(f"Maximum PSNR: {max_psnr:.2f} dB\n")
        f.write(f"Standard Deviation: {std_psnr:.2f} dB\n\n")
        
        f.write("Top 10 Lowest PSNR Frames:\n")
        for i, frame_idx in enumerate(lowest_indices):
            f.write(f"  {i+1:2d}. Frame {frame_idx:3d}: {frame_psnrs[frame_idx]:.2f} dB (t={time_axis[frame_idx]:.2f}s)\n")
        
        f.write(f"\nAll Frame PSNR Values:\n")
        for i, psnr in enumerate(frame_psnrs):
            f.write(f"Frame {i:3d}: {psnr:.2f} dB\n")
    
    print(f"Detailed analysis saved to: {analysis_path}")
    
    # Show plot
    plt.show()
    
    return frame_psnrs, min_frame_idx, min_psnr

def main():
    parser = argparse.ArgumentParser(description="Frame-by-frame PSNR analysis between two videos.")
    parser.add_argument("cpu_video_path", help="Path to the CPU-generated video file.")
    parser.add_argument("cuda_video_path", help="Path to the CUDA-generated video file.")
    args = parser.parse_args()

    cpu_video = args.cpu_video_path
    cuda_video = args.cuda_video_path
    
    # The analyze_frame_psnr function and load_video_frames within it already handle
    # checking if videos exist and can be opened.
    
    print("=== Frame-by-Frame PSNR Analysis ===")
    print(f"CPU Video: {cpu_video}")
    print(f"CUDA Video: {cuda_video}")
    
    try:
        frame_psnrs, min_frame_idx, min_psnr = analyze_frame_psnr(cpu_video, cuda_video)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Lowest PSNR occurs at Frame {min_frame_idx} with {min_psnr:.2f} dB")
        print(f"Overall average PSNR: {np.mean(frame_psnrs):.2f} dB")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())