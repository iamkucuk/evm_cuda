#!/usr/bin/env python3
"""
Frame-by-frame PSNR analysis for CUDA Gaussian hybrid implementation
"""

import cv2
import numpy as np
import sys

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 analyze_gaussian_psnr.py <cuda_video> <cpu_reference>")
        sys.exit(1)
    
    video1_path = sys.argv[1]  # CUDA hybrid video
    video2_path = sys.argv[2]  # CPU reference video
    
    # Load videos
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print('Error: Cannot open one or both videos')
        sys.exit(1)
    
    frame_count = 0
    psnr_values = []
    
    print('Frame-by-frame PSNR Analysis:')
    print(f'CUDA Hybrid: {video1_path}')
    print(f'CPU Reference: {video2_path}')
    print('=' * 60)
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        if frame1.shape != frame2.shape:
            print(f'Warning: Frame size mismatch at frame {frame_count}')
            break
        
        psnr = calculate_psnr(frame1, frame2)
        psnr_values.append(psnr)
        
        if frame_count % 50 == 0 or frame_count < 10:
            print(f'Frame {frame_count:3d}: PSNR = {psnr:6.2f} dB')
        
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if psnr_values:
        avg_psnr = np.mean(psnr_values)
        std_psnr = np.std(psnr_values)
        min_psnr = np.min(psnr_values)
        max_psnr = np.max(psnr_values)
        
        print('=' * 60)
        print(f'SUMMARY ({frame_count} frames):')
        print(f'Average PSNR: {avg_psnr:.2f} ± {std_psnr:.2f} dB')
        print(f'PSNR range:   {min_psnr:.2f} - {max_psnr:.2f} dB')
        
        # Quality assessment
        if avg_psnr >= 30:
            print('✅ EXCELLENT: Average PSNR >= 30 dB - Perfect match with CPU')
        elif avg_psnr >= 25:
            print('✅ GOOD: Average PSNR >= 25 dB - Good match with CPU') 
        elif avg_psnr >= 20:
            print('⚠️  ACCEPTABLE: Average PSNR >= 20 dB - Acceptable quality')
        else:
            print('❌ POOR: Average PSNR < 20 dB - Poor quality match')
        
        # Component analysis
        print('\nCOMPONENT ANALYSIS:')
        print('This hybrid test validates CUDA Gaussian spatial filtering')
        print('combined with CPU temporal filtering and reconstruction.')
        print(f'PSNR result indicates CUDA spatial filtering accuracy.')

if __name__ == '__main__':
    main()