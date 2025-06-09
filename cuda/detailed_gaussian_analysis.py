#!/usr/bin/env python3
"""
Comprehensive analysis of CUDA Gaussian implementation with detailed figures
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images"""
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_ssim_manual(img1, img2):
    """Simple SSIM calculation"""
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.std(img1)
    sigma2 = np.std(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    return ssim

def analyze_videos_comprehensive(video1_path, video2_path):
    """Comprehensive analysis with detailed statistics"""
    
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print('Error: Cannot open one or both videos')
        return
    
    frame_count = 0
    psnr_values = []
    ssim_values = []
    mse_values = []
    max_diff_values = []
    
    print('🎯 COMPREHENSIVE CUDA GAUSSIAN PYRAMID ANALYSIS')
    print('=' * 70)
    print(f'CUDA Hybrid Output: {video1_path}')
    print(f'CPU Reference:      {video2_path}')
    print('=' * 70)
    
    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            break
        
        if frame1.shape != frame2.shape:
            print(f'Warning: Frame size mismatch at frame {frame_count}')
            break
        
        # Calculate metrics
        psnr = calculate_psnr(frame1, frame2)
        ssim = calculate_ssim_manual(frame1.astype(np.float64), frame2.astype(np.float64))
        mse = np.mean((frame1.astype(np.float64) - frame2.astype(np.float64)) ** 2)
        max_diff = np.max(np.abs(frame1.astype(np.float64) - frame2.astype(np.float64)))
        
        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)
        max_diff_values.append(max_diff)
        
        if frame_count % 25 == 0 or frame_count < 15:
            print(f'Frame {frame_count:3d}: PSNR={psnr:6.2f}dB, SSIM={ssim:6.4f}, MSE={mse:8.2f}, MaxDiff={max_diff:6.1f}')
        
        frame_count += 1
    
    cap1.release()
    cap2.release()
    
    if not psnr_values:
        print('No frames processed')
        return
    
    # Calculate comprehensive statistics
    psnr_array = np.array(psnr_values)
    ssim_array = np.array(ssim_values)
    mse_array = np.array(mse_values)
    max_diff_array = np.array(max_diff_values)
    
    print('=' * 70)
    print(f'📊 STATISTICAL SUMMARY ({frame_count} frames):')
    print('=' * 70)
    
    print(f'PSNR Statistics:')
    print(f'  Mean:    {np.mean(psnr_array):8.2f} ± {np.std(psnr_array):6.2f} dB')
    print(f'  Median:  {np.median(psnr_array):8.2f} dB')
    print(f'  Range:   {np.min(psnr_array):8.2f} - {np.max(psnr_array):6.2f} dB')
    print(f'  Q1-Q3:   {np.percentile(psnr_array, 25):8.2f} - {np.percentile(psnr_array, 75):6.2f} dB')
    
    print(f'\nSSIM Statistics:')
    print(f'  Mean:    {np.mean(ssim_array):8.4f} ± {np.std(ssim_array):8.4f}')
    print(f'  Median:  {np.median(ssim_array):8.4f}')
    print(f'  Range:   {np.min(ssim_array):8.4f} - {np.max(ssim_array):8.4f}')
    
    print(f'\nMSE Statistics:')
    print(f'  Mean:    {np.mean(mse_array):8.2f} ± {np.std(mse_array):8.2f}')
    print(f'  Median:  {np.median(mse_array):8.2f}')
    print(f'  Range:   {np.min(mse_array):8.2f} - {np.max(mse_array):8.2f}')
    
    print(f'\nMax Pixel Difference Statistics:')
    print(f'  Mean:    {np.mean(max_diff_array):8.1f} ± {np.std(max_diff_array):8.1f}')
    print(f'  Median:  {np.median(max_diff_array):8.1f}')
    print(f'  Range:   {np.min(max_diff_array):8.1f} - {np.max(max_diff_array):8.1f}')
    
    # Quality assessment
    print('\n' + '=' * 70)
    print('🏆 QUALITY ASSESSMENT:')
    print('=' * 70)
    
    avg_psnr = np.mean(psnr_array)
    avg_ssim = np.mean(ssim_array)
    
    if avg_psnr >= 50:
        psnr_grade = "🌟 EXCEPTIONAL"
    elif avg_psnr >= 40:
        psnr_grade = "✅ EXCELLENT"
    elif avg_psnr >= 30:
        psnr_grade = "✅ VERY GOOD"
    elif avg_psnr >= 25:
        psnr_grade = "✅ GOOD"
    else:
        psnr_grade = "⚠️  NEEDS IMPROVEMENT"
    
    if avg_ssim >= 0.999:
        ssim_grade = "🌟 EXCEPTIONAL"
    elif avg_ssim >= 0.99:
        ssim_grade = "✅ EXCELLENT"
    elif avg_ssim >= 0.95:
        ssim_grade = "✅ VERY GOOD"
    elif avg_ssim >= 0.8:
        ssim_grade = "✅ GOOD"
    else:
        ssim_grade = "⚠️  NEEDS IMPROVEMENT"
    
    print(f'PSNR Quality: {psnr_grade} ({avg_psnr:.2f} dB)')
    print(f'SSIM Quality: {ssim_grade} ({avg_ssim:.4f})')
    
    # Frames with different quality levels
    excellent_frames = np.sum(psnr_array >= 70)
    very_good_frames = np.sum((psnr_array >= 50) & (psnr_array < 70))
    good_frames = np.sum((psnr_array >= 30) & (psnr_array < 50))
    poor_frames = np.sum(psnr_array < 30)
    
    print(f'\nFrame Quality Distribution:')
    print(f'  Exceptional (≥70dB): {excellent_frames:3d} frames ({excellent_frames/frame_count*100:5.1f}%)')
    print(f'  Very Good (50-70dB): {very_good_frames:3d} frames ({very_good_frames/frame_count*100:5.1f}%)')
    print(f'  Good (30-50dB):      {good_frames:3d} frames ({good_frames/frame_count*100:5.1f}%)')
    print(f'  Poor (<30dB):        {poor_frames:3d} frames ({poor_frames/frame_count*100:5.1f}%)')
    
    print('\n' + '=' * 70)
    print('🔬 COMPONENT ANALYSIS:')
    print('=' * 70)
    print('This analysis validates CUDA Gaussian spatial filtering accuracy.')
    print('High PSNR indicates excellent numerical precision in:')
    print('  • Gaussian kernel convolution')
    print('  • Pyramid down/up sampling')
    print('  • Border handling (REFLECT_101)')
    print('  • YIQ color space processing')
    print('  • GPU↔CPU memory transfers')
    
    if avg_psnr >= 70:
        print('\n🎉 CONCLUSION: CUDA implementation achieves near-perfect accuracy!')
        print('   The spatial filtering component is ready for production use.')
    elif avg_psnr >= 50:
        print('\n✅ CONCLUSION: CUDA implementation achieves excellent accuracy!')
        print('   Minor differences likely due to floating-point precision.')
    elif avg_psnr >= 30:
        print('\n✅ CONCLUSION: CUDA implementation achieves good accuracy!')
        print('   Suitable for most practical applications.')
    else:
        print('\n⚠️  CONCLUSION: CUDA implementation needs optimization.')
        print('   Consider investigating numerical differences.')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python3 detailed_gaussian_analysis.py <cuda_video> <cpu_reference>")
        sys.exit(1)
    
    analyze_videos_comprehensive(sys.argv[1], sys.argv[2])