#include <cuda_runtime.h>

__global__ void transpose_frame_to_pixel_kernel(
    const float* d_frame_major, float* d_pixel_major,
    int width, int height, int channels, int num_frames) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    int pixel_idx = y * width + x;
    int channel_offset = c;
    
    for (int frame = 0; frame < num_frames; frame++) {
        // Frame-major: [frame][y][x][c]
        int frame_idx = frame * height * width * channels + y * width * channels + x * channels + c;
        
        // Pixel-major: [pixel][channel][frame]
        int pixel_major_idx = pixel_idx * channels * num_frames + channel_offset * num_frames + frame;
        
        d_pixel_major[pixel_major_idx] = d_frame_major[frame_idx];
    }
}

__global__ void transpose_pixel_to_frame_kernel(
    const float* d_pixel_major, float* d_frame_major,
    int width, int height, int channels, int num_frames) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || c >= channels) return;
    
    int pixel_idx = y * width + x;
    int channel_offset = c;
    
    for (int frame = 0; frame < num_frames; frame++) {
        // Pixel-major: [pixel][channel][frame]
        int pixel_major_idx = pixel_idx * channels * num_frames + channel_offset * num_frames + frame;
        
        // Frame-major: [frame][y][x][c]
        int frame_idx = frame * height * width * channels + y * width * channels + x * channels + c;
        
        d_frame_major[frame_idx] = d_pixel_major[pixel_major_idx];
    }
}

extern "C" cudaError_t launch_transpose_frame_to_pixel(
    const float* d_frame_major, float* d_pixel_major,
    int width, int height, int channels, int num_frames,
    dim3 gridSize, dim3 blockSize) {
    
    transpose_frame_to_pixel_kernel<<<gridSize, blockSize>>>(
        d_frame_major, d_pixel_major, width, height, channels, num_frames);
    
    return cudaGetLastError();
}

extern "C" cudaError_t launch_transpose_pixel_to_frame(
    const float* d_pixel_major, float* d_frame_major,
    int width, int height, int channels, int num_frames,
    dim3 gridSize, dim3 blockSize) {
    
    transpose_pixel_to_frame_kernel<<<gridSize, blockSize>>>(
        d_pixel_major, d_frame_major, width, height, channels, num_frames);
    
    return cudaGetLastError();
}