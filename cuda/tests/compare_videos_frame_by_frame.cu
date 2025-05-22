#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

// Function to calculate PSNR between two images
double calculatePSNR(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    diff.convertTo(diff, CV_32F);
    diff = diff.mul(diff);
    
    cv::Scalar mse = cv::mean(diff);
    double mse_avg = (mse[0] + mse[1] + mse[2]) / 3.0;
    
    if (mse_avg < 1e-10) {
        return 100.0; // Perfect match
    }
    
    return 20.0 * log10(255.0 / sqrt(mse_avg));
}

// Function to calculate SSIM between two images
double calculateSSIM(const cv::Mat& img1, const cv::Mat& img2) {
    const double C1 = 6.5025, C2 = 58.5225;
    
    cv::Mat I1, I2;
    img1.convertTo(I1, CV_32F);
    img2.convertTo(I2, CV_32F);
    
    cv::Mat I2_2 = I2.mul(I2);
    cv::Mat I1_2 = I1.mul(I1);
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
    return (mssim[0] + mssim[1] + mssim[2]) / 3.0;
}

struct FrameMetrics {
    int frame_number;
    double psnr;
    double ssim;
    double max_pixel_diff;
    double mean_pixel_diff;
};

bool compareVideos(const std::string& video1_path, const std::string& video2_path, 
                  const std::string& output_csv = "") {
    cv::VideoCapture cap1(video1_path);
    cv::VideoCapture cap2(video2_path);
    
    if (!cap1.isOpened()) {
        std::cerr << "Error: Cannot open video 1: " << video1_path << std::endl;
        return false;
    }
    
    if (!cap2.isOpened()) {
        std::cerr << "Error: Cannot open video 2: " << video2_path << std::endl;
        return false;
    }
    
    // Get video properties
    int frame_count1 = static_cast<int>(cap1.get(cv::CAP_PROP_FRAME_COUNT));
    int frame_count2 = static_cast<int>(cap2.get(cv::CAP_PROP_FRAME_COUNT));
    double fps1 = cap1.get(cv::CAP_PROP_FPS);
    double fps2 = cap2.get(cv::CAP_PROP_FPS);
    
    std::cout << "Video 1: " << frame_count1 << " frames @ " << fps1 << " FPS" << std::endl;
    std::cout << "Video 2: " << frame_count2 << " frames @ " << fps2 << " FPS" << std::endl;
    
    if (frame_count1 != frame_count2) {
        std::cerr << "Warning: Videos have different frame counts!" << std::endl;
    }
    
    std::vector<FrameMetrics> metrics;
    std::ofstream csv_file;
    
    if (!output_csv.empty()) {
        csv_file.open(output_csv);
        csv_file << "Frame,PSNR,SSIM,MaxPixelDiff,MeanPixelDiff" << std::endl;
    }
    
    cv::Mat frame1, frame2;
    int frame_num = 0;
    
    double total_psnr = 0.0, total_ssim = 0.0;
    double min_psnr = 100.0, max_psnr = 0.0;
    double min_ssim = 1.0, max_ssim = 0.0;
    
    std::cout << "\nFrame-by-frame comparison:" << std::endl;
    std::cout << "Frame\tPSNR\t\tSSIM\t\tMax Diff\tMean Diff" << std::endl;
    std::cout << "-----\t----\t\t----\t\t--------\t---------" << std::endl;
    
    while (cap1.read(frame1) && cap2.read(frame2)) {
        // Ensure frames have the same size
        if (frame1.size() != frame2.size()) {
            std::cerr << "Error: Frame " << frame_num << " has different sizes!" << std::endl;
            break;
        }
        
        // Calculate metrics
        double psnr = calculatePSNR(frame1, frame2);
        double ssim = calculateSSIM(frame1, frame2);
        
        // Calculate pixel differences
        cv::Mat diff;
        cv::absdiff(frame1, frame2, diff);
        double min_diff, max_diff;
        cv::minMaxLoc(diff, &min_diff, &max_diff);
        cv::Scalar mean_diff = cv::mean(diff);
        double mean_diff_avg = (mean_diff[0] + mean_diff[1] + mean_diff[2]) / 3.0;
        
        FrameMetrics fm;
        fm.frame_number = frame_num;
        fm.psnr = psnr;
        fm.ssim = ssim;
        fm.max_pixel_diff = max_diff;
        fm.mean_pixel_diff = mean_diff_avg;
        metrics.push_back(fm);
        
        // Update statistics
        total_psnr += psnr;
        total_ssim += ssim;
        if (psnr < min_psnr) min_psnr = psnr;
        if (psnr > max_psnr) max_psnr = psnr;
        if (ssim < min_ssim) min_ssim = ssim;
        if (ssim > max_ssim) max_ssim = ssim;
        
        // Print every 10th frame or first/last frames
        if (frame_num % 10 == 0 || frame_num < 5 || frame_num == frame_count1 - 1) {
            std::cout << std::fixed << std::setprecision(2);
            std::cout << frame_num << "\t" << psnr << "\t\t" << ssim 
                      << "\t\t" << max_diff << "\t\t" << mean_diff_avg << std::endl;
        }
        
        // Write to CSV
        if (csv_file.is_open()) {
            csv_file << frame_num << "," << psnr << "," << ssim 
                     << "," << max_diff << "," << mean_diff_avg << std::endl;
        }
        
        frame_num++;
        
        // Limit output for very long videos
        if (frame_num >= 300) {  // Process max 300 frames for analysis
            std::cout << "... (processing limited to 300 frames)" << std::endl;
            break;
        }
    }
    
    if (csv_file.is_open()) {
        csv_file.close();
    }
    
    // Print summary statistics
    int total_frames = metrics.size();
    double avg_psnr = total_psnr / total_frames;
    double avg_ssim = total_ssim / total_frames;
    
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "COMPARISON SUMMARY" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    std::cout << "Total frames compared: " << total_frames << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Average PSNR: " << avg_psnr << " dB" << std::endl;
    std::cout << "PSNR range: " << min_psnr << " - " << max_psnr << " dB" << std::endl;
    std::cout << "Average SSIM: " << avg_ssim << std::endl;
    std::cout << "SSIM range: " << min_ssim << " - " << max_ssim << std::endl;
    
    // Interpretation
    std::cout << "\nINTERPRETATION:" << std::endl;
    if (avg_psnr > 40) {
        std::cout << "• PSNR > 40 dB: Excellent quality, very similar videos" << std::endl;
    } else if (avg_psnr > 30) {
        std::cout << "• PSNR 30-40 dB: Good quality, noticeable but acceptable differences" << std::endl;
    } else if (avg_psnr > 20) {
        std::cout << "• PSNR 20-30 dB: Fair quality, significant differences" << std::endl;
    } else {
        std::cout << "• PSNR < 20 dB: Poor quality, major differences" << std::endl;
    }
    
    if (avg_ssim > 0.95) {
        std::cout << "• SSIM > 0.95: Excellent structural similarity" << std::endl;
    } else if (avg_ssim > 0.8) {
        std::cout << "• SSIM 0.8-0.95: Good structural similarity" << std::endl;
    } else if (avg_ssim > 0.6) {
        std::cout << "• SSIM 0.6-0.8: Fair structural similarity" << std::endl;
    } else {
        std::cout << "• SSIM < 0.6: Poor structural similarity" << std::endl;
    }
    
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <video1_path> <video2_path> [output_csv]" << std::endl;
        std::cout << "Example: " << argv[0] << " cpu_output.mp4 cuda_output.mp4 comparison.csv" << std::endl;
        return 1;
    }
    
    std::string video1_path = argv[1];
    std::string video2_path = argv[2];
    std::string output_csv = (argc > 3) ? argv[3] : "";
    
    std::cout << "Frame-by-Frame Video Comparison Tool" << std::endl;
    std::cout << "Video 1 (CPU): " << video1_path << std::endl;
    std::cout << "Video 2 (CUDA): " << video2_path << std::endl;
    if (!output_csv.empty()) {
        std::cout << "CSV output: " << output_csv << std::endl;
    }
    std::cout << std::string(60, '=') << std::endl;
    
    bool success = compareVideos(video1_path, video2_path, output_csv);
    
    return success ? 0 : 1;
}