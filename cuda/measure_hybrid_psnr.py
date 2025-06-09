#!/usr/bin/env python3
"""
Measure PSNR between hybrid CUDA/CPU output and CPU reference
"""

import cv2
import numpy as np
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
        return 100.0  # Use 100 dB for perfect match
    
    # Calculate PSNR
    max_pixel_value = 255.0
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    from skimage.metrics import structural_similarity as ssim
    
    # Convert to grayscale if color
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1
        img2_gray = img2
    
    return ssim(img1_gray, img2_gray)

def load_video_frames(video_path):
    """Load all frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Loading {frame_count} frames from {video_path} at {fps} FPS...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames")
    
    return frames, fps

def main():
    if len(sys.argv) != 3:
        print("Usage: python measure_hybrid_psnr.py <hybrid_video> <cpu_reference_video>")
        sys.exit(1)
    
    hybrid_video = sys.argv[1]
    cpu_video = sys.argv[2]
    
    print("=== Hybrid CUDA/CPU vs CPU Reference PSNR Analysis ===")
    print(f"Hybrid video: {hybrid_video}")
    print(f"CPU reference: {cpu_video}")
    
    # Load videos
    hybrid_frames, hybrid_fps = load_video_frames(hybrid_video)
    cpu_frames, cpu_fps = load_video_frames(cpu_video)
    
    # Verify frame counts match
    if len(hybrid_frames) != len(cpu_frames):
        print(f"Warning: Frame count mismatch - Hybrid: {len(hybrid_frames)}, CPU: {len(cpu_frames)}")
        min_frames = min(len(hybrid_frames), len(cpu_frames))
        hybrid_frames = hybrid_frames[:min_frames]
        cpu_frames = cpu_frames[:min_frames]
        print(f"Using first {min_frames} frames for comparison")
    
    # Calculate PSNR for each frame
    print("\n=== Calculating Frame-by-Frame PSNR ===")
    frame_psnrs = []
    frame_ssims = []
    num_frames = len(hybrid_frames)
    
    for i in range(num_frames):
        # Ensure frames have same size and type
        hybrid_frame = hybrid_frames[i]
        cpu_frame = cpu_frames[i]
        
        if hybrid_frame.shape != cpu_frame.shape:
            print(f"Frame {i}: Size mismatch, resizing hybrid frame to match CPU")
            hybrid_frame = cv2.resize(hybrid_frame, (cpu_frame.shape[1], cpu_frame.shape[0]))
        
        # Calculate PSNR
        psnr = calculate_psnr(cpu_frame, hybrid_frame)
        frame_psnrs.append(psnr)
        
        # Calculate SSIM (optional - comment out if skimage not available)
        try:
            ssim_val = calculate_ssim(cpu_frame, hybrid_frame)
            frame_ssims.append(ssim_val)
        except ImportError:
            frame_ssims.append(0.0)  # Skip SSIM if skimage not available
        
        if i % 50 == 0:
            print(f"  Frame {i}/{num_frames}: PSNR = {psnr:.2f} dB")
    
    # Convert to numpy arrays for analysis
    frame_psnrs = np.array(frame_psnrs)
    frame_ssims = np.array(frame_ssims)
    
    # Find statistics
    avg_psnr = np.mean(frame_psnrs)
    min_psnr = np.min(frame_psnrs)
    max_psnr = np.max(frame_psnrs)
    std_psnr = np.std(frame_psnrs)
    
    if len(frame_ssims) > 0 and frame_ssims[0] != 0.0:
        avg_ssim = np.mean(frame_ssims)
        min_ssim = np.min(frame_ssims)
        max_ssim = np.max(frame_ssims)
    else:
        avg_ssim = min_ssim = max_ssim = 0.0
    
    # Print results
    print("\n=== COMPARISON RESULTS ===")
    print(f"Frames analyzed: {num_frames}")
    print(f"Average PSNR: {avg_psnr:.2f} dB (±{std_psnr:.2f} dB)")
    print(f"PSNR range: {min_psnr:.2f} - {max_psnr:.2f} dB")
    
    if avg_ssim > 0.0:
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"SSIM range: {min_ssim:.4f} - {max_ssim:.4f}")
    
    print("\n=== QUALITY ASSESSMENT ===")
    if avg_psnr >= 30:
        print("✅ EXCELLENT: PSNR ≥ 30 dB - High quality match")
    elif avg_psnr >= 25:
        print("✅ GOOD: PSNR ≥ 25 dB - Acceptable quality")
    elif avg_psnr >= 20:
        print("⚠️  FAIR: 20-25 dB - Moderate differences")
    else:
        print("❌ POOR: PSNR < 20 dB - Significant differences")
    
    if avg_ssim > 0.0:
        if avg_ssim >= 0.9:
            print("✅ Excellent structural similarity (SSIM ≥ 0.9)")
        elif avg_ssim >= 0.8:
            print("✅ Good structural similarity (SSIM ≥ 0.8)")
        elif avg_ssim >= 0.5:
            print("⚠️  Fair structural similarity (SSIM ≥ 0.5)")
        else:
            print("❌ Poor structural similarity (SSIM < 0.5)")

if __name__ == "__main__":
    main()