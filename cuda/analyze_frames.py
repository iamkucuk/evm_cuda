import cv2
import numpy as np

# Compare frame 1 from both outputs
working_cap = cv2.VideoCapture("cuda_butterworth_laplacian_output.avi")
hybrid_cap = cv2.VideoCapture("cuda_color_cuda_butter_cpu_rest_fixed_output.avi")

# Read first frame from each
ret1, working_frame = working_cap.read()
ret2, hybrid_frame = hybrid_cap.read()

if ret1 and ret2:
    # Convert to float for PSNR calculation
    working_frame = working_frame.astype(np.float32)
    hybrid_frame = hybrid_frame.astype(np.float32)
    
    # Calculate MSE and PSNR
    mse = np.mean((working_frame - hybrid_frame) ** 2)
    if mse == 0:
        psnr = 100
    else:
        psnr = 20 * np.log10(255.0 / np.sqrt(mse))
    
    print(f"Frame 1 PSNR between working and hybrid: {psnr:.2f} dB")
    print(f"Working frame shape: {working_frame.shape}")
    print(f"Hybrid frame shape: {hybrid_frame.shape}")
    print(f"Working frame stats: min={working_frame.min():.2f}, max={working_frame.max():.2f}, mean={working_frame.mean():.2f}")
    print(f"Hybrid frame stats: min={hybrid_frame.min():.2f}, max={hybrid_frame.max():.2f}, mean={hybrid_frame.mean():.2f}")
else:
    print("Could not read frames from one or both videos")

working_cap.release()
hybrid_cap.release()