#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <filesystem> // For creating output directory

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp> // For VideoCapture, VideoWriter
#include <opencv2/imgproc.hpp> // For BGR <-> RGB conversion if needed
#include <opencv2/highgui.hpp> // For imshow (optional debugging)

#include "evmcpp/laplacian_pyramid.hpp"
#include "evmcpp/gaussian_pyramid.hpp" // Include Gaussian pyramid header
#include "evmcpp/processing.hpp"
// butterworth.hpp is included via processing.hpp or pyramid headers

// --- Default Configuration Values ---
// These can be overridden by command-line arguments
struct Config {
    std::string input_video_path = "../../evmpy/data/face.mp4"; // Relative path from build dir
    std::string output_video_path = "face_cpp.avi"; // Output in build dir
    int pyramid_levels = 4;
    double alpha = 50.0; // Default from user feedback
    double lambda_cutoff = 16.0; // Keeping previous default, make configurable
    double fl = 0.8333; // Default from user feedback
    double fh = 1.0;    // Default from user feedback
    double chrom_attenuation = 1.0; // Default from user feedback
    std::string mode = "laplacian"; // Default mode
};

// --- Simple Argument Parser ---
void printUsage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [options]\n\n"
              << "Options:\n"
              << "  --input <path>       Input video file path (default: ../../evmpy/data/face.mp4)\n"
              << "  --output <path>      Output video file path (default: face_cpp.avi)\n"
              << "  --level <int>        Number of pyramid levels (default: 4)\n"
              << "  --alpha <float>      Magnification factor (default: 50.0)\n"
              << "  --lambda_cutoff <f>  Spatial cutoff wavelength (default: 16.0)\n"
              << "  --fl <float>         Low frequency cutoff (Hz) (default: 0.8333)\n"
              << "  --fh <float>         High frequency cutoff (Hz) (default: 1.0)\n"
              << "  --chrom_atten <f>    Chrominance attenuation (default: 1.0)\n"
              << "  --mode <name>        Processing mode: 'laplacian' or 'gaussian' (default: laplacian)\n"
              << "  --help               Show this help message\n"
              << std::endl;
}

bool parseArgs(int argc, char* argv[], Config& config) {
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help") {
            printUsage(argv[0]);
            return false; // Indicate exit
        } else if (arg == "--input" && i + 1 < argc) {
            config.input_video_path = argv[++i];
        } else if (arg == "--output" && i + 1 < argc) {
            config.output_video_path = argv[++i];
        } else if (arg == "--level" && i + 1 < argc) {
            try { config.pyramid_levels = std::stoi(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid integer for --level\n"; return false; }
        } else if (arg == "--alpha" && i + 1 < argc) {
            try { config.alpha = std::stod(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid float for --alpha\n"; return false; }
        } else if (arg == "--lambda_cutoff" && i + 1 < argc) {
            try { config.lambda_cutoff = std::stod(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid float for --lambda_cutoff\n"; return false; }
        } else if (arg == "--fl" && i + 1 < argc) {
            try { config.fl = std::stod(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid float for --fl\n"; return false; }
        } else if (arg == "--fh" && i + 1 < argc) {
            try { config.fh = std::stod(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid float for --fh\n"; return false; }
        } else if (arg == "--chrom_atten" && i + 1 < argc) {
            try { config.chrom_attenuation = std::stod(argv[++i]); }
            catch (...) { std::cerr << "Error: Invalid float for --chrom_atten\n"; return false; }
        } else if (arg == "--mode" && i + 1 < argc) {
            config.mode = argv[++i];
            if (config.mode != "laplacian" && config.mode != "gaussian") {
                std::cerr << "Error: Invalid mode '" << config.mode << "'. Must be 'laplacian' or 'gaussian'.\n";
                return false;
            }
        } else {
            std::cerr << "Error: Unknown or invalid argument: " << arg << std::endl;
            printUsage(argv[0]);
            return false;
        }
    }
    if (config.pyramid_levels <= 0) {
        std::cerr << "Error: Pyramid level must be positive.\n"; return false;
    }
     if (config.fl >= config.fh || config.fl <= 0 || config.fh <= 0) {
        std::cerr << "Error: Invalid frequency range (fl=" << config.fl << ", fh=" << config.fh << "). Requires 0 < fl < fh.\n"; return false;
    }
    return true; // Indicate success
}

// --- Helper: Load Video Frames ---
bool loadVideoFrames(const std::string& path, std::vector<cv::Mat>& frames, double& fps) {
    cv::VideoCapture cap(path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Cannot open video file: " << path << std::endl;
        return false;
    }

    fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) {
        std::cerr << "Warning: Could not get valid FPS from video. Using default 30.0." << std::endl;
        fps = 30.0; // Default fallback
    }

    frames.clear();
    cv::Mat frame;
    while (cap.read(frame)) {
        if (frame.empty()) {
            std::cerr << "Warning: Read empty frame." << std::endl;
            continue;
        }
        // OpenCV reads in BGR, convert to RGB for consistency with Python processing
        cv::Mat rgb_frame;
        cv::cvtColor(frame, rgb_frame, cv::COLOR_BGR2RGB);
        frames.push_back(rgb_frame); // Store RGB frames
    }
    cap.release();

    if (frames.empty()) {
        std::cerr << "Error: No frames loaded from video: " << path << std::endl;
        return false;
    }

    std::cout << "Loaded " << frames.size() << " frames at " << fps << " FPS from " << path << std::endl;
    return true;
}

// --- Helper: Save Video Frames ---
bool saveVideoFrames(const std::string& path, const std::vector<cv::Mat>& frames, double fps, const cv::Size& frame_size) {
     if (frames.empty()) {
        std::cerr << "Error: No frames to save." << std::endl;
        return false;
    }

    // Use MJPG codec, similar to Python example
    int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
    cv::VideoWriter writer(path, fourcc, fps, frame_size);

    if (!writer.isOpened()) {
        std::cerr << "Error: Could not open video writer for path: " << path << std::endl;
        return false;
    }

    std::cout << "Saving " << frames.size() << " frames to " << path << "..." << std::endl;
    for (size_t i = 0; i < frames.size(); ++i) {
        if (frames[i].empty() || frames[i].size() != frame_size) {
             std::cerr << "Warning: Skipping invalid frame " << i << " during save." << std::endl;
             continue;
        }
        // Convert final RGB back to BGR for OpenCV VideoWriter
        cv::Mat bgr_frame;
        cv::cvtColor(frames[i], bgr_frame, cv::COLOR_RGB2BGR);
        writer.write(bgr_frame);
    }
    writer.release();
    std::cout << "Video saved successfully." << std::endl;
    return true;
}


int main(int argc, char* argv[]) {
    std::cout << "EVM C++ Implementation - Full Pipeline" << std::endl;
    std::cout << "Using OpenCV version: " << CV_VERSION << std::endl;

    Config config;
    if (!parseArgs(argc, argv, config)) {
        return (argc > 1 && std::string(argv[1]) == "--help") ? 0 : 1; // Exit code 0 for --help, 1 for error
    }

    std::cout << "--- Configuration ---" << std::endl;
    std::cout << "Input Video: " << config.input_video_path << std::endl;
    std::cout << "Output Video: " << config.output_video_path << std::endl;
    std::cout << "Pyramid Levels: " << config.pyramid_levels << std::endl;
    std::cout << "Alpha: " << config.alpha << std::endl;
    std::cout << "Lambda Cutoff: " << config.lambda_cutoff << std::endl;
    std::cout << "Freq Low (fl): " << config.fl << " Hz" << std::endl;
    std::cout << "Freq High (fh): " << config.fh << " Hz" << std::endl;
    std::cout << "Chrom Attenuation: " << config.chrom_attenuation << std::endl;
    std::cout << "Mode: " << config.mode << std::endl;
    std::cout << "---------------------" << std::endl;

    // Define the standard 5x5 Gaussian kernel (like OpenCV's default for pyrDown/Up)
    cv::Mat kernel1d = cv::getGaussianKernel(5, -1, CV_32F); // ksize=5, sigma=-1 (default calculation)
    cv::Mat kernel = kernel1d * kernel1d.t(); // Create 2D kernel by outer product
    std::cout << "Using 5x5 Gaussian Kernel for pyrDown/pyrUp." << std::endl;


    try {
        if (config.mode == "laplacian") {
            // --- Laplacian Pipeline ---
            std::cout << "Starting Laplacian Pipeline..." << std::endl;

            std::vector<cv::Mat> original_frames;
            double fps = 0;

            // 1. Load Video
            if (!loadVideoFrames(config.input_video_path, original_frames, fps)) {
                return 1;
            }
            if (original_frames.empty()) return 1;

            // Check frequency range against Nyquist frequency (fps/2)
            if (config.fh >= fps / 2.0) {
                 std::cerr << "Error: High frequency cutoff (" << config.fh << " Hz) must be less than Nyquist frequency (" << fps / 2.0 << " Hz)." << std::endl;
                 return 1;
            }
            if (config.fl >= fps / 2.0) {
                 std::cerr << "Error: Low frequency cutoff (" << config.fl << " Hz) must be less than Nyquist frequency (" << fps / 2.0 << " Hz)." << std::endl;
                 return 1;
            }

            std::vector<cv::Mat> output_frames;
            output_frames.reserve(original_frames.size());

            // 2. Build Laplacian Pyramids
            std::cout << " Building Laplacian pyramids..." << std::endl;
            std::vector<std::vector<cv::Mat>> laplacian_pyramids = evmcpp::getLaplacianPyramids(original_frames, config.pyramid_levels, kernel);
            std::cout << " Finished building Laplacian pyramids." << std::endl;

            // 3. Filter Pyramids Temporally
            std::cout << " Filtering Laplacian pyramids temporally..." << std::endl;
            std::vector<std::vector<cv::Mat>> filtered_pyramids = evmcpp::filterLaplacianPyramids(
                laplacian_pyramids,
                config.pyramid_levels,
                fps,
                {config.fl, config.fh},
                config.alpha,
                config.lambda_cutoff,
                config.chrom_attenuation
            );
            std::cout << " Finished filtering Laplacian pyramids." << std::endl;

            // 4. Reconstruct Output Video Frames
            std::cout << " Reconstructing output frames from Laplacian..." << std::endl;
            for (size_t i = 0; i < original_frames.size(); ++i) {
                 if (i >= filtered_pyramids.size()) {
                     std::cerr << "Warning: Mismatch between original frame count and filtered pyramid count at index " << i << ". Using original frame." << std::endl;
                     output_frames.push_back(original_frames[i].clone());
                     continue;
                 }
                 if (original_frames[i].empty()) {
                      std::cerr << "Warning: Skipping reconstruction for empty original frame " << i << "." << std::endl;
                      output_frames.push_back(cv::Mat());
                      continue;
                 }
                 if (filtered_pyramids[i].empty()) {
                      std::cerr << "Warning: Filtered Laplacian pyramid for frame " << i << " is empty. Using original frame." << std::endl;
                      output_frames.push_back(original_frames[i].clone());
                      continue;
                 }
                output_frames.push_back(evmcpp::reconstructLaplacianImage(original_frames[i], filtered_pyramids[i], kernel));
                if (i > 0 && i % 50 == 0) {
                     std::cout << "  Reconstructed frame " << i << "/" << original_frames.size() << std::endl;
                }
            }
            std::cout << " Finished reconstructing Laplacian frames." << std::endl;

            // 5. Save Output Video (Laplacian Path)
            if (!output_frames.empty() && !output_frames[0].empty()) {
                 cv::Size frame_size = output_frames[0].size();
                 if (!saveVideoFrames(config.output_video_path, output_frames, fps, frame_size)) {
                     return 1;
                 }
            } else {
                 std::cerr << "Error: No valid output frames generated to save." << std::endl;
                 return 1;
            }


        } else if (config.mode == "gaussian") {
            // --- Gaussian Pipeline ---
            std::cout << "Starting Gaussian Pipeline..." << std::endl;
            // Call the new batch processing function
            evmcpp::processVideoGaussianBatch(
                config.input_video_path,
                config.output_video_path,
                config.pyramid_levels,
                config.alpha,
                // lambda_c is not used in the Gaussian batch process
                config.fl,
                config.fh,
                config.chrom_attenuation
            );
             std::cout << " Finished Gaussian Pipeline." << std::endl;

        } else {
             // Should not happen due to arg parsing, but good practice
             std::cerr << "Error: Unknown mode '" << config.mode << "' selected." << std::endl;
             return 1;
        }

        // Saving is handled internally by processVideoGaussian or at the end of the Laplacian block

    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Processing complete. Output saved to " << config.output_video_path << std::endl;
    return 0;
}