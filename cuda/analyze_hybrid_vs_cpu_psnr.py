#!/usr/bin/env python3
"""
Frame-by-frame PSNR analysis between CPU full and CUDA Gaussian + CPU rest hybrid videos
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Loading {frame_count} frames from {os.path.basename(video_path)} at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames")
    
    return frames, fps

def main():
    # Define video paths
    cpu_video = "../cpp/build/cpu_full_face_output.mp4"
    hybrid_video = "cuda_gaussian_cpu_hybrid_output.avi"
    
    # Check if videos exist
    if not os.path.exists(cpu_video):
        print(f"Error: CPU video not found at {cpu_video}")
        return 1
    
    if not os.path.exists(hybrid_video):
        print(f"Error: Hybrid video not found at {hybrid_video}")
        return 1
    
    print("=== Frame-by-Frame PSNR Analysis ===")
    print(f"CPU Full Video: {cpu_video}")
    print(f"CUDA Gaussian + CPU Rest Video: {hybrid_video}")
    
    try:
        # Load videos
        print("\n=== Loading Videos ===")
        cpu_frames, cpu_fps = load_video_frames(cpu_video)
        hybrid_frames, hybrid_fps = load_video_frames(hybrid_video)
        
        # Verify frame counts match
        if len(cpu_frames) != len(hybrid_frames):
            print(f"Warning: Frame count mismatch - CPU: {len(cpu_frames)}, Hybrid: {len(hybrid_frames)}")
            min_frames = min(len(cpu_frames), len(hybrid_frames))
            cpu_frames = cpu_frames[:min_frames]
            hybrid_frames = hybrid_frames[:min_frames]
            print(f"Using first {min_frames} frames for comparison")
        
        # Calculate PSNR for each frame
        print("\n=== Calculating Frame-by-Frame PSNR ===")
        frame_psnrs = []
        num_frames = len(cpu_frames)
        
        for i in range(num_frames):
            # Ensure frames have same size and type
            cpu_frame = cpu_frames[i]
            hybrid_frame = hybrid_frames[i]
            
            if cpu_frame.shape != hybrid_frame.shape:
                print(f"Frame {i}: Size mismatch, resizing Hybrid frame to match CPU")
                hybrid_frame = cv2.resize(hybrid_frame, (cpu_frame.shape[1], cpu_frame.shape[0]))
            
            # Calculate PSNR
            psnr = calculate_psnr(cpu_frame, hybrid_frame)
            frame_psnrs.append(psnr)
            
            if i % 50 == 0:
                print(f"  Frame {i}/{num_frames}: PSNR = {psnr:.2f} dB")
        
        # Convert to numpy array for analysis
        frame_psnrs = np.array(frame_psnrs)
        
        # Find statistics
        avg_psnr = np.mean(frame_psnrs)
        min_psnr = np.min(frame_psnrs)
        max_psnr = np.max(frame_psnrs)
        std_psnr = np.std(frame_psnrs)
        min_frame_idx = np.argmin(frame_psnrs)
        
        print(f"\n=== PSNR Statistics ===")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Minimum PSNR: {min_psnr:.2f} dB (Frame {min_frame_idx})")
        print(f"Maximum PSNR: {max_psnr:.2f} dB")
        print(f"Standard Deviation: {std_psnr:.2f} dB")
        
        # Get top 10 lowest PSNR frames
        lowest_indices = np.argsort(frame_psnrs)[:10]
        
        print(f"\nTop 10 Lowest PSNR Frames:")
        for i, frame_idx in enumerate(lowest_indices):
            time_stamp = frame_idx / cpu_fps
            print(f"  {i+1:2d}. Frame {frame_idx:3d}: {frame_psnrs[frame_idx]:.2f} dB (t={time_stamp:.2f}s)")
        
        print(f"\n=== Analysis Complete ===")
        print(f"Lowest PSNR occurs at Frame {min_frame_idx} with {min_psnr:.2f} dB")
        print(f"Overall average PSNR: {avg_psnr:.2f} dB")
        
        return 0
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())