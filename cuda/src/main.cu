/**
 * CUDA Eulerian Video Magnification - Unified Command Line Interface
 * Dual Algorithm Support: Gaussian + Laplacian modes
 * All code organized in proper src/ and include/ structure
 */

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <cmath>

// Include modular CUDA components from include/ directory
#include "cuda_gaussian_pyramid.cuh"    // Gaussian pyramid operations
#include "cuda_laplacian_pyramid.cuh"   // Laplacian pyramid operations
#include "cuda_temporal_filter.cuh"     // FFT + IIR temporal filtering
#include "cuda_processing.cuh"          // EVM reconstruction
#include "cuda_color_conversion.cuh"    // RGB ↔ YIQ conversion
#include "cuda_scaling.cuh"             // GPU data scaling operations

#include "cuda_transpose.cuh"

void check_cuda_error(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        std::cerr << "CUDA Error (" << message << "): " << cudaGetErrorString(error) << std::endl;
        exit(1);
    }
}

enum class EVMMode {
    GAUSSIAN,
    LAPLACIAN
};

struct EVMConfig {
    std::string input_video;
    std::string output_video;
    EVMMode mode = EVMMode::GAUSSIAN;
    float alpha = 50.0f;
    int level = 4;
    float fl = 0.8333f;  // low omega 
    float fh = 1.0f;     // high omega 
    float fps = 30.0f;
    float chrom_attenuation = 1.0f;
    bool timing = false;
    bool quiet = false;
    bool gpu_sync = false;  // GPU synchronization timing mode
    int warmup_runs = 0;
    int benchmark_runs = 1;
};

void gpu_sync_if_enabled(const EVMConfig& config, const char* operation) {
    if (config.gpu_sync) {
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cerr << "GPU sync error after " << operation << ": " << cudaGetErrorString(sync_error) << std::endl;
        }
    }
}

void print_usage(const char* program_name) {
    std::cout << "CUDA Eulerian Video Magnification - Command Line Interface\\n"
              << "Dual Algorithm Support: Gaussian and Laplacian modes\\n\\n"
              << "Usage: " << program_name << " --input=<input_video> --output=<output_video> [options]\\n\\n"
              << "Required Arguments:\\n"
              << "  --input=<path>           Input video file path\\n"
              << "  --output=<path>          Output video file path\\n\\n"
              << "Optional Arguments:\\n"
              << "  --mode=<algorithm>       EVM algorithm: gaussian or laplacian (default: gaussian)\\n"
              << "  --alpha=<float>          Magnification factor (default: 50.0)\\n"
              << "  --level=<int>            Number of pyramid levels (default: 4)\\n"
              << "  --fl=<float>             Low frequency cutoff in Hz (default: 0.8333)\\n"
              << "  --fh=<float>             High frequency cutoff in Hz (default: 1.0)\\n"
              << "  --fps=<float>            Video frame rate (default: 30.0)\\n"
              << "  --chrom_attenuation=<f>  Chrominance attenuation (default: 1.0)\\n"
              << "  --timing                 Show detailed timing information\\n"
              << "  --gpu-sync               Enable GPU synchronization for accurate kernel timing (slower)\\n"
              << "  --warmup=<int>           Number of warmup runs (default: 0)\\n"
              << "  --benchmark=<int>        Number of benchmark runs (default: 1)\\n"
              << "  --quiet                  Suppress progress output\\n"
              << "  --help                   Show this help message\\n\\n"
              << "Algorithm Modes:\\n"
              << "  gaussian                 FFT-based temporal filtering (high quality)\\n"
              << "  laplacian                IIR-based temporal filtering (high speed)\\n\\n"
              << "Timing Modes:\\n"
              << "  Default (async)          Measures kernel launch time (faster, preserves async nature)\\n"
              << "  --gpu-sync              Measures actual GPU execution time (slower, accurate)\\n\\n"
              << "Examples:\\n"
              << "  " << program_name << " --input=face.mp4 --output=magnified.avi\\n"
              << "  " << program_name << " --input=video.mp4 --output=result.avi --mode=laplacian --alpha=100\\n"
              << std::endl;
}

bool parse_arguments(int argc, char* argv[], EVMConfig& config) {
    if (argc < 3) {
        return false;
    }
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg.substr(0, 8) == "--input=") {
            config.input_video = arg.substr(8);
        } else if (arg.substr(0, 9) == "--output=") {
            config.output_video = arg.substr(9);
        } else if (arg.substr(0, 8) == "--alpha=") {
            config.alpha = std::stof(arg.substr(8));
        } else if (arg.substr(0, 8) == "--level=") {
            config.level = std::stoi(arg.substr(8));
        } else if (arg.substr(0, 5) == "--fl=") {
            config.fl = std::stof(arg.substr(5));
        } else if (arg.substr(0, 5) == "--fh=") {
            config.fh = std::stof(arg.substr(5));
        } else if (arg.substr(0, 6) == "--fps=") {
            config.fps = std::stof(arg.substr(6));
        } else if (arg.substr(0, 20) == "--chrom_attenuation=") {
            config.chrom_attenuation = std::stof(arg.substr(20));
        } else if (arg.substr(0, 7) == "--mode=") {
            std::string mode_str = arg.substr(7);
            if (mode_str == "gaussian") {
                config.mode = EVMMode::GAUSSIAN;
            } else if (mode_str == "laplacian") {
                config.mode = EVMMode::LAPLACIAN;
            } else {
                std::cerr << "Error: Invalid mode '" << mode_str << "'. Use 'gaussian' or 'laplacian'" << std::endl;
                return false;
            }
        } else if (arg == "--timing") {
            config.timing = true;
        } else if (arg == "--gpu-sync") {
            config.gpu_sync = true;
        } else if (arg == "--quiet") {
            config.quiet = true;
        } else if (arg.substr(0, 9) == "--warmup=") {
            config.warmup_runs = std::stoi(arg.substr(9));
        } else if (arg.substr(0, 12) == "--benchmark=") {
            config.benchmark_runs = std::stoi(arg.substr(12));
        } else {
            std::cerr << "Error: Unknown argument: " << arg << std::endl;
            return false;
        }
    }
    
    // Validate required arguments
    if (config.input_video.empty() || config.output_video.empty()) {
        std::cerr << "Error: Both --input and --output are required" << std::endl;
        return false;
    }
    
    // Validate parameters
    if (config.alpha <= 0) {
        std::cerr << "Error: Alpha must be positive" << std::endl;
        return false;
    }
    if (config.level < 1 || config.level > 10) {
        std::cerr << "Error: Level must be between 1 and 10" << std::endl;
        return false;
    }
    if (config.fl <= 0 || config.fh <= 0 || config.fl >= config.fh) {
        std::cerr << "Error: Invalid frequency range. Requires 0 < fl < fh" << std::endl;
        return false;
    }
    if (config.fps <= 0) {
        std::cerr << "Error: FPS must be positive" << std::endl;
        return false;
    }
    if (config.warmup_runs < 0) {
        std::cerr << "Error: Warmup runs must be non-negative" << std::endl;
        return false;
    }
    if (config.benchmark_runs < 1) {
        std::cerr << "Error: Benchmark runs must be at least 1" << std::endl;
        return false;
    }
    
    return true;
}

// Forward declarations
int run_gaussian_pipeline(const EVMConfig& config, const std::vector<cv::Mat>& frames, 
                         int width, int height, int num_frames, 
                         std::chrono::high_resolution_clock::time_point start_time);
int run_laplacian_pipeline(const EVMConfig& config, const std::vector<cv::Mat>& frames,
                          int width, int height, int num_frames,
                          std::chrono::high_resolution_clock::time_point start_time);

int main(int argc, char* argv[]) {
    EVMConfig config;
    
    if (!parse_arguments(argc, argv, config)) {
        print_usage(argv[0]);
        return 1;
    }
    
    if (!config.quiet) {
        std::cout << "CUDA Eulerian Video Magnification - Full GPU Pipeline" << std::endl;
        std::string mode_name = (config.mode == EVMMode::GAUSSIAN) ? "Gaussian" : "Laplacian";
        std::string quality_info = (config.mode == EVMMode::GAUSSIAN) ? "high quality" : "high speed";
        std::cout << "Algorithm: " << mode_name << " EVM (" << quality_info << " mode)" << std::endl;
        std::cout << "Parameters: alpha=" << config.alpha << ", level=" << config.level 
                  << ", fl=" << config.fl << ", fh=" << config.fh 
                  << ", fps=" << config.fps << ", chrom_attenuation=" << config.chrom_attenuation << std::endl;
    }
    
    // Open input video
    cv::VideoCapture cap(config.input_video);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file " << config.input_video << std::endl;
        return 1;
    }
    
    // Get video properties
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double input_fps = cap.get(cv::CAP_PROP_FPS);
    
    if (!config.quiet) {
        std::cout << "Video: " << width << "x" << height << ", " << total_frames << " frames @ " << input_fps << " FPS" << std::endl;
    }
    
    // Read all frames
    std::vector<cv::Mat> frames;
    cv::Mat frame;
    while (cap.read(frame) && frames.size() < total_frames) {
        frames.push_back(frame.clone());
    }
    cap.release();
    
    int num_frames = frames.size();
    if (!config.quiet) {
        std::cout << "Loaded " << num_frames << " frames" << std::endl;
    }
    
    // Benchmarking setup
    std::vector<double> execution_times;
    int total_runs = config.warmup_runs + config.benchmark_runs;
    
    if (!config.quiet && (config.warmup_runs > 0 || config.benchmark_runs > 1)) {
        std::cout << "\\n=== Benchmarking Mode ===" << std::endl;
        std::cout << "Warmup runs: " << config.warmup_runs << std::endl;
        std::cout << "Benchmark runs: " << config.benchmark_runs << std::endl;
        std::cout << "Timing mode: " << (config.gpu_sync ? "GPU-synchronized (accurate kernels)" : "Async launch (default)") << std::endl;
    }
    
    // Execute warmup + benchmark runs
    for (int run = 0; run < total_runs; run++) {
        bool is_warmup = (run < config.warmup_runs);
        bool is_benchmark = !is_warmup;
        
        if (!config.quiet && total_runs > 1) {
            if (is_warmup) {
                std::cout << "\\n--- Warmup Run " << (run + 1) << "/" << config.warmup_runs << " ---" << std::endl;
            } else {
                std::cout << "\\n--- Benchmark Run " << (run - config.warmup_runs + 1) << "/" << config.benchmark_runs << " ---" << std::endl;
            }
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Execute pipeline
        int result;
        if (config.mode == EVMMode::GAUSSIAN) {
            result = run_gaussian_pipeline(config, frames, width, height, num_frames, start_time);
        } else {
            result = run_laplacian_pipeline(config, frames, width, height, num_frames, start_time);
        }
        
        gpu_sync_if_enabled(config, "complete pipeline");
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double seconds = duration.count() / 1000.0;
        
        // Store benchmark results (not warmup)
        if (is_benchmark) {
            execution_times.push_back(seconds);
        }
        
        if (!config.quiet && total_runs > 1) {
            std::cout << "Run time: " << seconds << " seconds";
            if (config.gpu_sync) {
                std::cout << " (GPU-synchronized)";
            } else {
                std::cout << " (async mode)";
            }
            std::cout << std::endl;
        }
        
        if (result != 0) {
            return result;
        }
    }
    
    // Report benchmarking statistics
    if (!config.quiet && config.benchmark_runs > 1) {
        double total_time = 0.0;
        double min_time = execution_times[0];
        double max_time = execution_times[0];
        
        for (double time : execution_times) {
            total_time += time;
            min_time = std::min(min_time, time);
            max_time = std::max(max_time, time);
        }
        
        double mean_time = total_time / execution_times.size();
        
        // Calculate standard deviation
        double variance = 0.0;
        for (double time : execution_times) {
            variance += (time - mean_time) * (time - mean_time);
        }
        variance /= execution_times.size();
        double std_dev = std::sqrt(variance);
        double cv = (std_dev / mean_time) * 100.0;
        
        std::cout << "\\n=== Benchmark Results ===" << std::endl;
        std::cout << "Runs: " << config.benchmark_runs << std::endl;
        std::cout << "Mean: " << mean_time << " seconds" << std::endl;
        std::cout << "Std Dev: " << std_dev << " seconds" << std::endl;
        std::cout << "CV: " << cv << "%" << std::endl;
        std::cout << "Min: " << min_time << " seconds" << std::endl;
        std::cout << "Max: " << max_time << " seconds" << std::endl;
    }
    
    return 0;
}

int run_gaussian_pipeline(const EVMConfig& config, const std::vector<cv::Mat>& frames, 
                         int width, int height, int num_frames, 
                         std::chrono::high_resolution_clock::time_point start_time) {
    
    if (!config.quiet) {
        std::cout << "\\n=== GPU-Resident Pipeline (CORRECTED): Uploading All Frames ===" << std::endl;
    }
    
    // =====================================================================
    // MEMORY ALLOCATION: GPU-RESIDENT ARCHITECTURE  
    // =====================================================================
    const int channels = 3;
    const size_t frame_size = width * height * channels * sizeof(float);
    const size_t total_size = width * height * channels * num_frames;
    const size_t total_bytes = total_size * sizeof(float);
    
    // GPU memory for entire pipeline - all data stays on device
    float* d_input_frames_255 = nullptr;      // Input RGB frames [0,255]
    float* d_input_frames_1 = nullptr;        // Input RGB frames [0,1] for reconstruction
    float* d_spatially_filtered_255 = nullptr; // Spatially filtered YIQ [0,255]
    float* d_spatially_filtered_1 = nullptr;  // Spatially filtered YIQ [0,1] for temporal
    float* d_pixel_major = nullptr;           // Transposed for temporal processing
    float* d_temporal_filtered = nullptr;     // Temporally filtered results [0,1]
    float* d_final_frames = nullptr;          // Final reconstructed RGB frames [0,1]
    
    check_cuda_error(cudaMalloc(&d_input_frames_255, total_bytes), "Allocate input frames 255");
    check_cuda_error(cudaMalloc(&d_input_frames_1, total_bytes), "Allocate input frames 1");
    check_cuda_error(cudaMalloc(&d_spatially_filtered_255, total_bytes), "Allocate spatial filtered 255");
    check_cuda_error(cudaMalloc(&d_spatially_filtered_1, total_bytes), "Allocate spatial filtered 1");
    check_cuda_error(cudaMalloc(&d_pixel_major, total_bytes), "Allocate pixel major");
    check_cuda_error(cudaMalloc(&d_temporal_filtered, total_bytes), "Allocate temporal filtered");
    check_cuda_error(cudaMalloc(&d_final_frames, total_bytes), "Allocate final frames");
    
    // =====================================================================
    // UPLOAD: All input frames to GPU (SINGLE TRANSFER)
    // =====================================================================
    auto upload_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_frames; i++) {
        cv::Mat rgb_frame;
        cv::cvtColor(frames[i], rgb_frame, cv::COLOR_BGR2RGB);
        cv::Mat rgb_float;
        rgb_frame.convertTo(rgb_float, CV_32FC3);  // Keep [0,255] range
        
        float* frame_ptr = d_input_frames_255 + (i * width * height * channels);
        check_cuda_error(
            cudaMemcpy(frame_ptr, rgb_float.ptr<float>(), frame_size, cudaMemcpyHostToDevice),
            "Upload input frame"
        );
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Uploaded " << (i + 1) << "/" << num_frames << " frames to GPU" << std::endl;
        }
    }
    auto upload_end = std::chrono::high_resolution_clock::now();
    auto upload_duration = std::chrono::duration_cast<std::chrono::milliseconds>(upload_end - upload_start);
    
    if (config.timing) {
        std::cout << "Upload time: " << upload_duration.count() << " ms" << std::endl;
    }
    
    // =====================================================================
    // STEP 1: GPU-RESIDENT SPATIAL FILTERING (WORKS WITH [0,255])
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 1: GPU-Resident Spatial Filtering ===" << std::endl;
    }
    
    auto spatial_start = std::chrono::high_resolution_clock::now();
    
    // Process all frames on GPU without host transfers - spatial filtering expects [0,255]
    for (int i = 0; i < num_frames; i++) {
        float* input_ptr = d_input_frames_255 + (i * width * height * channels);      // [0,255]
        float* output_ptr = d_spatially_filtered_255 + (i * width * height * channels); // [0,255]
        
        cudaError_t result = cuda_evm::spatially_filter_gaussian_gpu(
            input_ptr, output_ptr, width, height, channels, config.level);
        check_cuda_error(result, "GPU-resident spatial filtering");
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "GPU spatial filtered " << (i + 1) << "/" << num_frames << " frames" << std::endl;
        }
    }
    gpu_sync_if_enabled(config, "spatial filtering");
    auto spatial_end = std::chrono::high_resolution_clock::now();
    auto spatial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spatial_end - spatial_start);
    
    if (config.timing) {
        std::cout << "GPU spatial filtering time: " << spatial_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // STEP 2: GPU-RESIDENT TEMPORAL FILTERING (NEEDS [0,1] INPUT)
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 2: GPU-Resident Temporal Filtering ===" << std::endl;
    }
    
    auto temporal_start = std::chrono::high_resolution_clock::now();
    
    // Scale spatially filtered data from [0,255] to [0,1] for temporal processing
    if (!config.quiet) {
        std::cout << "Scaling spatially filtered data [0,255] → [0,1]..." << std::endl;
    }
    const int total_elements = width * height * channels * num_frames;
    cudaError_t scale_err = gpu_scale_255_to_1(d_spatially_filtered_255, d_spatially_filtered_1, total_elements);
    check_cuda_error(scale_err, "Scale YIQ data to [0,1] range");
    
    // Transpose scaled YIQ data to pixel-major layout for temporal processing
    if (!config.quiet) {
        std::cout << "Transposing to pixel-major layout..." << std::endl;
    }
    dim3 blockSize(16, 16, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y,
                  (channels + blockSize.z - 1) / blockSize.z);
    
    cudaError_t transpose_err = launch_transpose_frame_to_pixel(
        d_spatially_filtered_1, d_pixel_major, width, height, channels, num_frames, gridSize, blockSize);
    check_cuda_error(transpose_err, "Transpose frame to pixel major");
    
    // Apply CUDA temporal filtering (all data stays on GPU, [0,1] range)
    if (!config.quiet) {
        std::cout << "Applying GPU-resident temporal filtering..." << std::endl;
    }
    cudaError_t temporal_err = cuda_evm::temporal_filter_gaussian_batch_gpu(
        d_pixel_major, d_temporal_filtered, width, height, channels, num_frames,
        config.fl, config.fh, config.fps, config.alpha, config.chrom_attenuation);
    check_cuda_error(temporal_err, "GPU-resident temporal filtering");
    
    // Transpose filtered results back to frame-major layout (keep in [0,1] for reconstruction)
    if (!config.quiet) {
        std::cout << "Transposing filtered results to frame-major layout..." << std::endl;
    }
    transpose_err = launch_transpose_pixel_to_frame(
        d_temporal_filtered, d_spatially_filtered_1, width, height, channels, num_frames, gridSize, blockSize);
    check_cuda_error(transpose_err, "Transpose pixel to frame major");
    
    gpu_sync_if_enabled(config, "temporal filtering");
    auto temporal_end = std::chrono::high_resolution_clock::now();
    auto temporal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(temporal_end - temporal_start);
    
    if (config.timing) {
        std::cout << "GPU temporal filtering time: " << temporal_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // STEP 3: GPU-RESIDENT RECONSTRUCTION (NEEDS [0,1] INPUTS)
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 3: GPU-Resident Reconstruction ===" << std::endl;
    }
    
    auto recon_start = std::chrono::high_resolution_clock::now();
    
    // Scale input frames from [0,255] to [0,1] for reconstruction
    if (!config.quiet) {
        std::cout << "Scaling input frames [0,255] → [0,1] for reconstruction..." << std::endl;
    }
    scale_err = gpu_scale_255_to_1(d_input_frames_255, d_input_frames_1, total_elements);
    check_cuda_error(scale_err, "Scale input frames to [0,1] range");
    
    // Process all frames on GPU without host transfers - both inputs now in [0,1] range
    if (!config.quiet) {
        std::cout << "Processing reconstruction on GPU..." << std::endl;
    }
    for (int i = 0; i < num_frames; i++) {
        float* original_ptr = d_input_frames_1 + (i * width * height * channels);       // Original RGB frames [0,1]
        float* filtered_ptr = d_spatially_filtered_1 + (i * width * height * channels); // Temporally filtered YIQ [0,1]
        float* output_ptr = d_final_frames + (i * width * height * channels);           // Final RGB output [0,1]
        
        // Apply CUDA reconstruction (all pointers are GPU memory, [0,1] range)
        cudaError_t result = cuda_evm::reconstruct_gaussian_frame_gpu(
            original_ptr, filtered_ptr, output_ptr, width, height, channels, config.alpha, config.chrom_attenuation);
        check_cuda_error(result, "GPU-resident reconstruction");
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "GPU reconstructed " << (i + 1) << "/" << num_frames << " frames" << std::endl;
        }
    }
    
    gpu_sync_if_enabled(config, "reconstruction");
    auto recon_end = std::chrono::high_resolution_clock::now();
    auto recon_duration = std::chrono::duration_cast<std::chrono::milliseconds>(recon_end - recon_start);
    
    if (config.timing) {
        std::cout << "GPU reconstruction time: " << recon_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // DOWNLOAD: Final results from GPU (SINGLE TRANSFER)
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Downloading Final Results from GPU ===" << std::endl;
    }
    
    auto download_start = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> final_frames;
    final_frames.reserve(num_frames);
    
    for (int i = 0; i < num_frames; i++) {
        cv::Mat output_frame(height, width, CV_32FC3);
        float* frame_ptr = d_final_frames + (i * width * height * channels);
        check_cuda_error(
            cudaMemcpy(output_frame.ptr<float>(), frame_ptr, frame_size, cudaMemcpyDeviceToHost),
            "Download final frame"
        );
        final_frames.push_back(output_frame.clone());
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Downloaded " << (i + 1) << "/" << num_frames << " final frames" << std::endl;
        }
    }
    auto download_end = std::chrono::high_resolution_clock::now();
    auto download_duration = std::chrono::duration_cast<std::chrono::milliseconds>(download_end - download_start);
    
    if (config.timing) {
        std::cout << "Download time: " << download_duration.count() << " ms" << std::endl;
    }
    
    // =====================================================================
    // CLEANUP: Free all GPU memory
    // =====================================================================
    cudaFree(d_input_frames_255);
    cudaFree(d_input_frames_1);
    cudaFree(d_spatially_filtered_255);
    cudaFree(d_spatially_filtered_1);
    cudaFree(d_pixel_major);
    cudaFree(d_temporal_filtered);
    cudaFree(d_final_frames);
    
    // =====================================================================
    // STEP 4: SAVE OUTPUT VIDEO
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 4: Saving Output Video ===" << std::endl;
    }
    
    cv::VideoWriter writer(config.output_video, cv::VideoWriter::fourcc('M','J','P','G'), config.fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open video writer" << std::endl;
        return 1;
    }
    
    for (int i = 0; i < num_frames; i++) {
        cv::Mat frame_bgr;
        cv::cvtColor(final_frames[i], frame_bgr, cv::COLOR_RGB2BGR);
        frame_bgr.convertTo(frame_bgr, CV_8UC3, 255.0);
        writer << frame_bgr;
    }
    writer.release();
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!config.quiet) {
        std::cout << "\\n=== CORRECTED GPU-Resident Pipeline Complete ===" << std::endl;
        std::cout << "Output video: " << config.output_video << std::endl;
        std::cout << "Total processing time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        
        if (config.timing) {
            std::cout << "Breakdown:" << std::endl;
            std::cout << "  Upload: " << upload_duration.count() << " ms" << std::endl;
            std::cout << "  GPU Spatial filtering: " << spatial_duration.count() << " ms" << std::endl;
            std::cout << "  GPU Temporal filtering: " << temporal_duration.count() << " ms" << std::endl;
            std::cout << "  GPU Reconstruction: " << recon_duration.count() << " ms" << std::endl;
            std::cout << "  Download: " << download_duration.count() << " ms" << std::endl;
        }
        
        std::cout << "\\n=== Pipeline Summary ===" << std::endl;
        std::cout << "✅ GPU-Resident Architecture: Complete with Correct Scaling" << std::endl;
        std::cout << "✅ Data Range Management: [0,255] ↔ [0,1] conversions handled" << std::endl;
        std::cout << "✅ Minimal Data Transfers: Upload once + Download once" << std::endl;
        std::cout << "✅ Full GPU Acceleration: Complete" << std::endl;
    }
    
    return 0;
}

int run_laplacian_pipeline(const EVMConfig& config, const std::vector<cv::Mat>& frames,
                          int width, int height, int num_frames,
                          std::chrono::high_resolution_clock::time_point start_time) {
    if (!config.quiet) {
        std::cout << "\\n=== CUDA Laplacian EVM Pipeline ===" << std::endl;
    }
    
    const int channels = 3;
    const size_t frame_size = width * height * channels * sizeof(float);
    const size_t total_size = width * height * channels * num_frames;
    const size_t total_bytes = total_size * sizeof(float);
    
    // =====================================================================
    // STEP 1: UPLOAD ALL FRAMES TO GPU
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 1: Uploading Frames to GPU ===" << std::endl;
    }
    
    auto upload_start = std::chrono::high_resolution_clock::now();
    
    // Allocate GPU memory for input frames
    float* d_input_frames = nullptr;
    check_cuda_error(cudaMalloc(&d_input_frames, total_bytes), "Allocate input frames");
    
    // Upload all frames
    for (int i = 0; i < num_frames; i++) {
        cv::Mat rgb_frame;
        cv::cvtColor(frames[i], rgb_frame, cv::COLOR_BGR2RGB);
        cv::Mat rgb_float;
        rgb_frame.convertTo(rgb_float, CV_32FC3, 1.0/255.0);  // Convert [0,255] to [0,1]
        
        float* frame_ptr = d_input_frames + (i * width * height * channels);
        check_cuda_error(
            cudaMemcpy(frame_ptr, rgb_float.ptr<float>(), frame_size, cudaMemcpyHostToDevice),
            "Upload input frame"
        );
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Uploaded " << (i + 1) << "/" << num_frames << " frames" << std::endl;
        }
    }
    
    auto upload_end = std::chrono::high_resolution_clock::now();
    auto upload_duration = std::chrono::duration_cast<std::chrono::milliseconds>(upload_end - upload_start);
    
    if (config.timing) {
        std::cout << "Upload time: " << upload_duration.count() << " ms" << std::endl;
    }
    
    // =====================================================================
    // STEP 2: SPATIAL PROCESSING - GENERATE LAPLACIAN PYRAMIDS
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 2: Generating Laplacian Pyramids ===" << std::endl;
    }
    
    auto spatial_start = std::chrono::high_resolution_clock::now();
    
    // Convert RGB to YIQ and generate Laplacian pyramids
    float* d_yiq_frames = nullptr;
    check_cuda_error(cudaMalloc(&d_yiq_frames, total_bytes), "Allocate YIQ frames");
    
    // Convert all frames to YIQ
    for (int i = 0; i < num_frames; i++) {
        float* rgb_ptr = d_input_frames + (i * width * height * channels);
        float* yiq_ptr = d_yiq_frames + (i * width * height * channels);
        
        // Use existing planar RGB to YIQ kernel from cuda_color_conversion.cu
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        cuda_evm::rgb_to_yiq_planar_kernel<<<gridSize, blockSize>>>(rgb_ptr, yiq_ptr, width, height, channels);
        cudaError_t result = cudaGetLastError();
        check_cuda_error(result, "RGB to YIQ conversion");
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Converted " << (i + 1) << "/" << num_frames << " frames to YIQ" << std::endl;
        }
    }
    
    // Generate Laplacian pyramids for all frames
    std::vector<cuda_evm::LaplacianPyramidGPU> pyramids;
    cudaError_t pyramid_result = cuda_evm::getLaplacianPyramids_gpu(
        d_yiq_frames, width, height, num_frames, config.level, pyramids);
    check_cuda_error(pyramid_result, "Generate Laplacian pyramids");
    
    gpu_sync_if_enabled(config, "spatial processing");
    auto spatial_end = std::chrono::high_resolution_clock::now();
    auto spatial_duration = std::chrono::duration_cast<std::chrono::milliseconds>(spatial_end - spatial_start);
    
    if (config.timing) {
        std::cout << "Spatial processing time: " << spatial_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // STEP 3: TEMPORAL FILTERING WITH SPATIAL ATTENUATION
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 3: Temporal Filtering with Spatial Attenuation ===" << std::endl;
    }
    
    auto temporal_start = std::chrono::high_resolution_clock::now();
    
    // Apply temporal filtering to Laplacian pyramids
    // Calculate lambda cutoff and delta exactly like CPU reference (laplacian_pyramid.cpp line 186)
    float lambda_cutoff = 16.0f;  // Default from EVM paper
    float delta = lambda_cutoff / (8.0f * (1.0f + config.alpha));
    
    cudaError_t filter_result = cuda_evm::filterLaplacianPyramids_gpu(
        pyramids, num_frames, config.level,
        config.fps, config.fl, config.fh,
        config.alpha, delta, config.chrom_attenuation);
    check_cuda_error(filter_result, "Temporal filtering");
    
    gpu_sync_if_enabled(config, "temporal filtering");
    auto temporal_end = std::chrono::high_resolution_clock::now();
    auto temporal_duration = std::chrono::duration_cast<std::chrono::milliseconds>(temporal_end - temporal_start);
    
    if (config.timing) {
        std::cout << "Temporal filtering time: " << temporal_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // STEP 4: RECONSTRUCTION
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 4: Laplacian Reconstruction ===" << std::endl;
    }
    
    auto recon_start = std::chrono::high_resolution_clock::now();
    
    // Allocate memory for reconstructed YIQ frames
    float* d_reconstructed_yiq = nullptr;
    check_cuda_error(cudaMalloc(&d_reconstructed_yiq, total_bytes), "Allocate reconstructed YIQ");
    
    // Reconstruct each frame from its filtered Laplacian pyramid
    for (int i = 0; i < num_frames; i++) {
        float* original_yiq_ptr = d_yiq_frames + (i * width * height * channels);
        float* output_yiq_ptr = d_reconstructed_yiq + (i * width * height * channels);
        
        cudaError_t recon_result = cuda_evm::reconstructLaplacianImage_gpu(
            original_yiq_ptr, pyramids[i], output_yiq_ptr, width, height);
        check_cuda_error(recon_result, "Laplacian reconstruction");
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Reconstructed " << (i + 1) << "/" << num_frames << " frames" << std::endl;
        }
    }
    
    // Convert reconstructed YIQ back to RGB
    float* d_final_rgb = nullptr;
    check_cuda_error(cudaMalloc(&d_final_rgb, total_bytes), "Allocate final RGB frames");
    
    for (int i = 0; i < num_frames; i++) {
        float* yiq_ptr = d_reconstructed_yiq + (i * width * height * channels);
        float* rgb_ptr = d_final_rgb + (i * width * height * channels);
        
        // Use existing planar YIQ to RGB kernel from cuda_color_conversion.cu
        dim3 blockSize(16, 16);
        dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);
        cuda_evm::yiq_to_rgb_planar_kernel<<<gridSize, blockSize>>>(yiq_ptr, rgb_ptr, width, height, channels);
        cudaError_t convert_result = cudaGetLastError();
        check_cuda_error(convert_result, "YIQ to RGB conversion");
    }
    
    gpu_sync_if_enabled(config, "reconstruction");
    auto recon_end = std::chrono::high_resolution_clock::now();
    auto recon_duration = std::chrono::duration_cast<std::chrono::milliseconds>(recon_end - recon_start);
    
    if (config.timing) {
        std::cout << "Reconstruction time: " << recon_duration.count() << " ms";
        if (config.gpu_sync) {
            std::cout << " (GPU-synchronized)";
        } else {
            std::cout << " (async launch only)";
        }
        std::cout << std::endl;
    }
    
    // =====================================================================
    // STEP 5: DOWNLOAD RESULTS AND SAVE VIDEO
    // =====================================================================
    if (!config.quiet) {
        std::cout << "\\n=== Step 5: Downloading Results and Saving Video ===" << std::endl;
    }
    
    auto download_start = std::chrono::high_resolution_clock::now();
    
    // Download final RGB frames
    std::vector<cv::Mat> final_frames;
    final_frames.reserve(num_frames);
    
    for (int i = 0; i < num_frames; i++) {
        cv::Mat output_frame(height, width, CV_32FC3);
        float* frame_ptr = d_final_rgb + (i * width * height * channels);
        check_cuda_error(
            cudaMemcpy(output_frame.ptr<float>(), frame_ptr, frame_size, cudaMemcpyDeviceToHost),
            "Download final frame"
        );
        final_frames.push_back(output_frame.clone());
        
        if (!config.quiet && (i + 1) % 50 == 0) {
            std::cout << "Downloaded " << (i + 1) << "/" << num_frames << " frames" << std::endl;
        }
    }
    
    auto download_end = std::chrono::high_resolution_clock::now();
    auto download_duration = std::chrono::duration_cast<std::chrono::milliseconds>(download_end - download_start);
    
    if (config.timing) {
        std::cout << "Download time: " << download_duration.count() << " ms" << std::endl;
    }
    
    // Save output video
    cv::VideoWriter writer(config.output_video, cv::VideoWriter::fourcc('M','J','P','G'), config.fps, cv::Size(width, height));
    if (!writer.isOpened()) {
        std::cerr << "Error: Cannot open video writer" << std::endl;
        
        // Cleanup GPU memory
        cudaFree(d_input_frames);
        cudaFree(d_yiq_frames);
        cudaFree(d_reconstructed_yiq);
        cudaFree(d_final_rgb);
        
        return 1;
    }
    
    for (int i = 0; i < num_frames; i++) {
        cv::Mat frame_bgr;
        cv::cvtColor(final_frames[i], frame_bgr, cv::COLOR_RGB2BGR);
        frame_bgr.convertTo(frame_bgr, CV_8UC3, 255.0);
        writer << frame_bgr;
    }
    writer.release();
    
    // =====================================================================
    // CLEANUP GPU MEMORY
    // =====================================================================
    cudaFree(d_input_frames);
    cudaFree(d_yiq_frames);
    cudaFree(d_reconstructed_yiq);
    cudaFree(d_final_rgb);
    // Note: pyramids will be automatically cleaned up by their destructors
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (!config.quiet) {
        std::cout << "\\n=== CUDA Laplacian Pipeline Complete ===" << std::endl;
        std::cout << "Output video: " << config.output_video << std::endl;
        std::cout << "Total processing time: " << duration.count() / 1000.0 << " seconds" << std::endl;
        
        if (config.timing) {
            std::cout << "Breakdown:" << std::endl;
            std::cout << "  Upload: " << upload_duration.count() << " ms" << std::endl;
            std::cout << "  Spatial Processing: " << spatial_duration.count() << " ms" << std::endl;
            std::cout << "  Temporal Filtering: " << temporal_duration.count() << " ms" << std::endl;
            std::cout << "  Reconstruction: " << recon_duration.count() << " ms" << std::endl;
            std::cout << "  Download: " << download_duration.count() << " ms" << std::endl;
        }
        
        std::cout << "\\n=== Pipeline Summary ===" << std::endl;
        std::cout << "✅ CUDA Laplacian Processing: Complete" << std::endl;
        std::cout << "✅ IIR Temporal Filtering: High Speed Mode" << std::endl;
        std::cout << "✅ Spatial Attenuation: Applied" << std::endl;
        std::cout << "✅ Multi-level Pyramid: " << config.level << " levels" << std::endl;
        std::cout << "✅ GPU Acceleration: Complete" << std::endl;
    }
    
    return 0;
}