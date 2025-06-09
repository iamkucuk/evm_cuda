#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <iomanip>

double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    
    // Convert to float for calculation
    diff.convertTo(diff, CV_32F);
    
    // Square the differences
    diff = diff.mul(diff);
    
    // Calculate mean squared error
    cv::Scalar mse_scalar = cv::mean(diff);
    double mse = (mse_scalar[0] + mse_scalar[1] + mse_scalar[2]) / 3.0; // Average across channels
    
    if (mse < 1e-10) {
        return 100.0; // Perfect match
    }
    
    double max_pixel_value = 255.0;
    double psnr = 20.0 * std::log10(max_pixel_value / std::sqrt(mse));
    return psnr;
}

double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;
    
    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);
    
    cv::Mat I1_2 = I1.mul(I1);
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_I2 = I1.mul(I2);
    
    cv::Mat mu1, mu2;
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);
    
    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    
    cv::Mat sigma1_2, sigma2_2, sigma12;
    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    
    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    
    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    
    cv::Mat t1, t2, t3;
    
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    
    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);
    
    cv::Scalar mssim = cv::mean(ssim_map);
    double ssim = (mssim[0] + mssim[1] + mssim[2]) / 3.0;
    
    return ssim;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <video1> <video2>" << std::endl;
        return -1;
    }
    
    std::string video1_path = argv[1];
    std::string video2_path = argv[2];
    
    std::cout << "=== Video Comparison Tool ===" << std::endl;
    std::cout << "Video 1 (CPU): " << video1_path << std::endl;
    std::cout << "Video 2 (CUDA): " << video2_path << std::endl;
    
    // Open video files
    cv::VideoCapture cap1(video1_path);
    cv::VideoCapture cap2(video2_path);
    
    if (!cap1.isOpened()) {
        std::cerr << "Error: Cannot open video 1: " << video1_path << std::endl;
        return -1;
    }
    
    if (!cap2.isOpened()) {
        std::cerr << "Error: Cannot open video 2: " << video2_path << std::endl;
        return -1;
    }
    
    // Get video properties
    int frame_count1 = (int)cap1.get(cv::CAP_PROP_FRAME_COUNT);
    int frame_count2 = (int)cap2.get(cv::CAP_PROP_FRAME_COUNT);
    double fps1 = cap1.get(cv::CAP_PROP_FPS);
    double fps2 = cap2.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video 1: " << frame_count1 << " frames at " << fps1 << " FPS" << std::endl;
    std::cout << "Video 2: " << frame_count2 << " frames at " << fps2 << " FPS" << std::endl;
    
    if (frame_count1 != frame_count2) {
        std::cout << "Warning: Videos have different frame counts!" << std::endl;
    }
    
    int num_frames = std::min(frame_count1, frame_count2);
    std::cout << "Comparing " << num_frames << " frames..." << std::endl;
    
    // Statistics
    std::vector<double> psnr_values;
    std::vector<double> ssim_values;
    double total_psnr = 0.0;
    double total_ssim = 0.0;
    int valid_frames = 0;
    
    cv::Mat frame1, frame2;
    
    for (int i = 0; i < num_frames; ++i) {
        bool ret1 = cap1.read(frame1);
        bool ret2 = cap2.read(frame2);
        
        if (!ret1 || !ret2 || frame1.empty() || frame2.empty()) {
            std::cout << "Warning: Could not read frame " << i << std::endl;
            continue;
        }
        
        // Ensure frames have the same size
        if (frame1.size() != frame2.size()) {
            std::cout << "Warning: Frame " << i << " has different sizes. Resizing..." << std::endl;
            cv::resize(frame2, frame2, frame1.size());
        }
        
        // Ensure frames have the same type
        if (frame1.type() != frame2.type()) {
            frame2.convertTo(frame2, frame1.type());
        }
        
        // Calculate PSNR and SSIM
        double psnr = calculatePSNR(frame1, frame2);
        double ssim = calculateSSIM(frame1, frame2);
        
        psnr_values.push_back(psnr);
        ssim_values.push_back(ssim);
        total_psnr += psnr;
        total_ssim += ssim;
        valid_frames++;
        
        if (i % 50 == 0) {
            std::cout << "Frame " << i << ": PSNR = " << std::fixed << std::setprecision(2) 
                      << psnr << " dB, SSIM = " << std::setprecision(4) << ssim << std::endl;
        }
    }
    
    cap1.release();
    cap2.release();
    
    if (valid_frames == 0) {
        std::cerr << "Error: No valid frames found for comparison!" << std::endl;
        return -1;
    }
    
    // Calculate statistics
    double avg_psnr = total_psnr / valid_frames;
    double avg_ssim = total_ssim / valid_frames;
    
    // Calculate standard deviations
    double psnr_variance = 0.0;
    double ssim_variance = 0.0;
    for (int i = 0; i < valid_frames; ++i) {
        psnr_variance += (psnr_values[i] - avg_psnr) * (psnr_values[i] - avg_psnr);
        ssim_variance += (ssim_values[i] - avg_ssim) * (ssim_values[i] - avg_ssim);
    }
    double psnr_std = std::sqrt(psnr_variance / valid_frames);
    double ssim_std = std::sqrt(ssim_variance / valid_frames);
    
    // Find min and max
    double min_psnr = *std::min_element(psnr_values.begin(), psnr_values.end());
    double max_psnr = *std::max_element(psnr_values.begin(), psnr_values.end());
    double min_ssim = *std::min_element(ssim_values.begin(), ssim_values.end());
    double max_ssim = *std::max_element(ssim_values.begin(), ssim_values.end());
    
    std::cout << "\n=== COMPARISON RESULTS ===" << std::endl;
    std::cout << "Valid frames compared: " << valid_frames << std::endl;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Average PSNR: " << avg_psnr << " dB (±" << psnr_std << " dB)" << std::endl;
    std::cout << "PSNR range: " << min_psnr << " - " << max_psnr << " dB" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average SSIM: " << avg_ssim << " (±" << ssim_std << ")" << std::endl;
    std::cout << "SSIM range: " << min_ssim << " - " << max_ssim << std::endl;
    
    std::cout << "\n=== QUALITY ASSESSMENT ===" << std::endl;
    if (avg_psnr > 30.0) {
        std::cout << "✅ EXCELLENT: PSNR > 30 dB - CUDA Butterworth matches CPU implementation very well!" << std::endl;
    } else if (avg_psnr > 25.0) {
        std::cout << "✅ GOOD: PSNR > 25 dB - CUDA Butterworth matches CPU implementation acceptably" << std::endl;
    } else if (avg_psnr > 20.0) {
        std::cout << "⚠️  FAIR: PSNR > 20 dB - Some differences exist" << std::endl;
    } else {
        std::cout << "❌ POOR: PSNR < 20 dB - Significant differences detected" << std::endl;
    }
    
    if (avg_ssim > 0.9) {
        std::cout << "✅ EXCELLENT structural similarity (SSIM > 0.9)" << std::endl;
    } else if (avg_ssim > 0.8) {
        std::cout << "✅ GOOD structural similarity (SSIM > 0.8)" << std::endl;
    } else {
        std::cout << "⚠️  FAIR structural similarity (SSIM < 0.8)" << std::endl;
    }
    
    return 0;
}