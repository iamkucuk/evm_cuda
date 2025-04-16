#ifndef EVMCPP_LAPLACIAN_PYRAMID_HPP
#define EVMCPP_LAPLACIAN_PYRAMID_HPP

#include <vector>
#include <opencv2/core/mat.hpp> // For cv::Mat

namespace evmcpp {

    /**
     * @brief Generates a Laplacian pyramid for a single image.
     * @param input_image Single image frame (assumed YIQ format).
     * @param level Number of pyramid levels to generate.
     * @param kernel The Gaussian kernel to use for pyrDown/pyrUp.
     * @return A vector of cv::Mat, where each Mat is a level of the Laplacian pyramid.
     */
    std::vector<cv::Mat> generateLaplacianPyramid(
        const cv::Mat& input_image,
        int level,
        const cv::Mat& kernel
    );

    /**
     * @brief Generates Laplacian pyramids for a sequence of images (video frames).
     * @param input_images Sequence of RGB image frames.
     * @param level Number of pyramid levels.
     * @param kernel The Gaussian kernel to use for pyrDown/pyrUp.
     * @return A vector where each element is a Laplacian pyramid (std::vector<cv::Mat>) for a frame.
     */
    std::vector<std::vector<cv::Mat>> getLaplacianPyramids(
        const std::vector<cv::Mat>& input_images,
        int level,
        const cv::Mat& kernel
    );

    /**
     * @brief Applies temporal bandpass filtering and spatial attenuation to pyramids.
     * @param pyramids_batch Collection of Laplacian pyramids (Time x Level x H x W x C).
     * @param level Number of pyramid levels.
     * @param fps Frames per second of the video.
     * @param freq_range Pair [low_freq, high_freq] for bandpass filter.
     * @param alpha Magnification factor.
     * @param lambda_cutoff Spatial wavelength cutoff for attenuation.
     * @param attenuation Factor to attenuate chrominance channels.
     * @return The modified pyramids_batch after filtering and attenuation.
     */
    std::vector<std::vector<cv::Mat>> filterLaplacianPyramids(
        const std::vector<std::vector<cv::Mat>>& pyramids_batch,
        int level,
        double fps,
        const std::pair<double, double>& freq_range,
        double alpha,
        double lambda_cutoff,
        double attenuation
    );

    // End of filterLaplacianPyramids declaration


    /**
     * @brief Reconstructs the magnified image from the original image and the filtered pyramid.
     * @param original_rgb_image The original input image (frame) in RGB format (e.g., CV_8UC3).
     * @param filtered_pyramid The temporally filtered Laplacian pyramid for the corresponding frame.
     * @param kernel The Gaussian kernel used during pyramid construction (needed for pyrUp).
     * @return The reconstructed magnified image in RGB format (CV_8UC3).
     */
    cv::Mat reconstructLaplacianImage(
        const cv::Mat& original_rgb_image,
        const std::vector<cv::Mat>& filtered_pyramid,
        const cv::Mat& kernel
    );

} // namespace evmcpp

#endif // EVMCPP_LAPLACIAN_PYRAMID_HPP