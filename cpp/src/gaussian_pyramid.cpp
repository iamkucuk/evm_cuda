// File: src/gaussian_pyramid.cpp
// Purpose: Implements functions for the Gaussian EVM pathway.

#include "gaussian_pyramid.hpp"
#include "processing.hpp" // Needs rgb2yiq, yiq2rgb, pyrDown, pyrUp
#include <opencv2/imgproc.hpp> // For cv::resize, other potential functions
#include <opencv2/core/hal/interface.h> // For CV types
#include <vector>
#include <cmath>
#include <limits>
#include <iostream> // For error printing

// Define PRINT_ERROR and PRINT_WARNING macros/functions if not globally available
#ifndef PRINT_ERROR
#define PRINT_ERROR(msg) std::cerr << "ERROR: " << msg << std::endl
#endif
#ifndef PRINT_WARNING
#define PRINT_WARNING(msg) std::cout << "WARNING: " << msg << std::endl
#endif


namespace evmcpp {

// --- spatiallyFilterGaussian ---
// TDD_ANCHOR: test_spatiallyFilterGaussian_matches_python
cv::Mat spatiallyFilterGaussian(const cv::Mat& inputRgb, int level, const cv::Mat& kernel) {
    // --- Input Validation ---
    if (inputRgb.empty() || inputRgb.type() != CV_8UC3 || level < 0 || kernel.empty()) {
        PRINT_ERROR("spatiallyFilterGaussian: Invalid input.");
        return cv::Mat();
    }
    if (level == 0) {
        // If level is 0, just convert to YIQ and return
        return rgb2yiq(inputRgb);
    }

    // --- 1. Convert RGB to YIQ (float) ---
    cv::Mat currentFrameYiq;
    try {
        currentFrameYiq = rgb2yiq(inputRgb);
    } catch (const std::exception& e) {
        PRINT_ERROR("spatiallyFilterGaussian: rgb2yiq failed. Error: " + std::string(e.what()));
        return cv::Mat();
    }
    if (currentFrameYiq.empty()) {
        PRINT_ERROR("spatiallyFilterGaussian: rgb2yiq returned empty matrix.");
        return cv::Mat();
    }

    // --- Store shapes for pyrUp target ---
    std::vector<cv::Size> shapes;
    shapes.push_back(currentFrameYiq.size());

    // --- 2. Downsample 'level' times ---
    cv::Mat downsampled = currentFrameYiq.clone(); // Start with the YIQ frame
    // PRINT_WARNING("Start Downsampling: Input shape=" + std::to_string(downsampled.cols) + "x" + std::to_string(downsampled.rows) + " Type=" + std::to_string(downsampled.type()));
    for (int i = 0; i < level; ++i) {
        cv::Mat tempDown;
        try {
            // Use the custom pyrDown from processing.hpp
            tempDown = evmcpp::pyrDown(downsampled, kernel);
        } catch (const std::exception& e) {
             PRINT_ERROR("spatiallyFilterGaussian: pyrDown failed at level " + std::to_string(i) + ". Error: " + std::string(e.what()));
             return cv::Mat();
        }

        if (tempDown.empty() || tempDown.total() == 0) {
            PRINT_ERROR("spatiallyFilterGaussian: pyrDown returned empty or zero-sized matrix at level " + std::to_string(i));
            return cv::Mat();
        }
        downsampled = tempDown;
        shapes.push_back(downsampled.size());
        // PRINT_WARNING("Downsample Level " + std::to_string(i) + ": Output shape=" + std::to_string(downsampled.cols) + "x" + std::to_string(downsampled.rows) + " Type=" + std::to_string(downsampled.type()));
    }

    // --- 3. Upsample 'level' times ---
    cv::Mat reconstructed = downsampled.clone(); // Start with the smallest level
    // PRINT_WARNING("Start Upsampling: Input shape=" + std::to_string(reconstructed.cols) + "x" + std::to_string(reconstructed.rows) + " Type=" + std::to_string(reconstructed.type()));
    for (int i = 0; i < level; ++i) {
        // Target shape is from the corresponding downsampling step
        // shapes[0] = original, shapes[1] = level 0 down, ..., shapes[level] = final down
        // For upsampling level i (0 to level-1), target is shapes[level - 1 - i]
        cv::Size targetShape = shapes[level - 1 - i];
        cv::Mat tempUp;
        try {
             // Use the custom pyrUp from processing.hpp
             tempUp = evmcpp::pyrUp(reconstructed, kernel, targetShape);
        } catch (const std::exception& e) {
             PRINT_ERROR("spatiallyFilterGaussian: pyrUp failed at level " + std::to_string(i) + ". Error: " + std::string(e.what()));
             return cv::Mat();
        }

        if (tempUp.empty()) {
            PRINT_ERROR("spatiallyFilterGaussian: pyrUp returned empty matrix at level " + std::to_string(i));
            return cv::Mat();
        }
        reconstructed = tempUp;
    }

    // --- 4. Final shape check (as per Python code) ---
    if (reconstructed.size() != shapes[0]) {
        // PRINT_WARNING("spatiallyFilterGaussian: Final shape " + std::to_string(reconstructed.cols) + "x" + std::to_string(reconstructed.rows) +
        //               " mismatch. Resizing to original YIQ frame size " + std::to_string(shapes[0].width) + "x" + std::to_string(shapes[0].height) + ".");
        try {
            cv::resize(reconstructed, reconstructed, shapes[0], 0, 0, cv::INTER_LINEAR); // Use linear interpolation for resize
        } catch (const cv::Exception& e) {
             PRINT_ERROR("spatiallyFilterGaussian: cv::resize failed during final shape correction. Error: " + std::string(e.what()));
             return cv::Mat(); // Return empty if resize fails
        }
    }

    return reconstructed; // Return the spatially filtered YIQ frame (CV_32FC3)
}


// --- temporalFilterGaussianBatch ---
// TDD_ANCHOR: test_temporalFilterGaussianBatch_matches_python
std::vector<cv::Mat> temporalFilterGaussianBatch(
    const std::vector<cv::Mat>& spatiallyFilteredBatch,
    float fps,
    float fl,
    float fh,
    float alpha,
    float chromAttenuation
) {
    // --- Input Validation ---
    if (spatiallyFilteredBatch.empty()) {
        PRINT_ERROR("temporalFilterGaussianBatch: Input batch is empty.");
        return {};
    }
    int numFrames = static_cast<int>(spatiallyFilteredBatch.size());
    if (numFrames <= 1) {
        //  PRINT_WARNING("temporalFilterGaussianBatch: Cannot perform temporal filtering with <= 1 frame. Returning empty vector.");
         return {}; // Need multiple frames for FFT
    }
    if (fps <= 0 || fl < 0 || fh <= fl) { // Allow alpha == 0 (just filtering)
        PRINT_ERROR("temporalFilterGaussianBatch: Invalid parameters (fps, freq range).");
        return {};
    }
    cv::Size frameSize = spatiallyFilteredBatch[0].size();
    int frameType = spatiallyFilteredBatch[0].type();
    if (frameType != CV_32FC3) {
         PRINT_ERROR("temporalFilterGaussianBatch: Input frames must be CV_32FC3.");
         return {};
    }
    // Check consistency of size/type across frames (optional but good practice)
    for(size_t i = 1; i < spatiallyFilteredBatch.size(); ++i) {
        if (spatiallyFilteredBatch[i].size() != frameSize || spatiallyFilteredBatch[i].type() != frameType) {
             PRINT_ERROR("temporalFilterGaussianBatch: Inconsistent frame size or type in input batch.");
             return {};
        }
    }

    int rows = frameSize.height;
    int cols = frameSize.width;
    int channels = 3; // YIQ

    // --- Prepare output vector ---
    std::vector<cv::Mat> filteredBatch(numFrames);
    for (int t = 0; t < numFrames; ++t) {
        filteredBatch[t] = cv::Mat::zeros(frameSize, frameType);
    }

    // --- Calculate frequency bins (like np.fft.fftfreq) ---
    std::vector<float> frequencies(numFrames);
    float freqStep = fps / static_cast<float>(numFrames);
    // Correct fftfreq logic: 0, 1*step, 2*step, ..., N/2*step, -(N/2-1)*step, ..., -1*step
    int n_over_2_ceil = (numFrames + 1) / 2; // Ceiling division for Nyquist handling
    for (int i = 0; i < numFrames; ++i) {
        if (i < n_over_2_ceil) {
            frequencies[i] = static_cast<float>(i) * freqStep;
        } else {
            frequencies[i] = static_cast<float>(i - numFrames) * freqStep;
        }
    }

    // --- Find indices closest to fl and fh ---
    int lowIdx = 0;
    int highIdx = 0;
    float minLowDiff = std::numeric_limits<float>::max();
    float minHighDiff = std::numeric_limits<float>::max();
    // Find index for low cutoff (fl)
    for (int i = 0; i < numFrames; ++i) {
        float diffLow = std::abs(frequencies[i] - fl);
        if (diffLow < minLowDiff) {
            minLowDiff = diffLow;
            lowIdx = i;
        }
    }
    // Find index for high cutoff (fh)
    for (int i = 0; i < numFrames; ++i) {
         float diffHigh = std::abs(frequencies[i] - fh);
         if (diffHigh < minHighDiff) {
            minHighDiff = diffHigh;
            highIdx = i;
        }
    }
    // Ensure lowIdx corresponds to the lower frequency magnitude if indices are the same
    if (lowIdx == highIdx && std::abs(fl) > std::abs(fh)) {
        std::swap(lowIdx, highIdx);
    } else if (std::abs(frequencies[lowIdx]) > std::abs(frequencies[highIdx])) {
         // If indices are different, ensure lowIdx corresponds to the frequency closer to 0
         std::swap(lowIdx, highIdx);
    }


    // --- Create frequency mask ---
    // Mask should be complex (CV_32FC2)
    cv::Mat complexMask = cv::Mat::zeros(numFrames, 1, CV_32FC2); // numFrames rows, 1 col, 2 channels (real, imag)
    // Python zeros out [:low] and [high:]. This keeps the range [low:high].
    // Need to map Python indices (potentially negative) to DFT indices (0 to N-1).
    // The frequencies vector already maps DFT index i to its corresponding frequency.
    // We want to keep frequencies f such that fl <= abs(f) <= fh.
    for (int i = 0; i < numFrames; ++i) {
        if (std::abs(frequencies[i]) >= fl && std::abs(frequencies[i]) <= fh) {
            complexMask.at<cv::Vec2f>(i, 0) = cv::Vec2f(1.0f, 0.0f); // Keep this frequency
        }
        // else: leave as zero (0.0f, 0.0f)
    }


    // --- Process each pixel's time series ---
    cv::Mat timeSeries(numFrames, 1, CV_32F); // Reusable buffer for pixel time series
    cv::Mat complexTimeSeries; // Reusable buffer for DFT result
    cv::Mat filteredTimeSeriesReal; // Reusable buffer for IDFT result

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int ch = 0; ch < channels; ++ch) {
                // 1. Extract time series for (r, c, ch)
                for (int t = 0; t < numFrames; ++t) {
                    // Assuming CV_32FC3 input
                    timeSeries.at<float>(t, 0) = spatiallyFilteredBatch[t].at<cv::Vec3f>(r, c)[ch];
                }

                // 2. Perform 1D Forward DFT
                cv::dft(timeSeries, complexTimeSeries, cv::DFT_COMPLEX_OUTPUT); // Output: numFrames x 1, CV_32FC2

                // 3. Apply frequency mask
                // Ensure mask dimensions match complexTimeSeries (should be numFrames x 1, CV_32FC2)
                if (complexMask.rows != complexTimeSeries.rows) {
                     PRINT_ERROR("temporalFilterGaussianBatch: Mask dimension mismatch during DFT processing.");
                     return {}; // Should not happen if logic is correct
                }
                cv::multiply(complexTimeSeries, complexMask, complexTimeSeries); // Element-wise multiplication

                // 4. Perform 1D Inverse DFT
                // Use DFT_REAL_OUTPUT because input to forward DFT was real
                // Use DFT_SCALE to normalize by numFrames
                cv::idft(complexTimeSeries, filteredTimeSeriesReal, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT); // Output: numFrames x 1, CV_32F

                // 5. Apply amplification and attenuation
                float currentAlpha = alpha;
                if (ch > 0) { // Attenuate I and Q channels (indices 1 and 2)
                    currentAlpha *= chromAttenuation;
                }

                // 6. Store result back into output batch
                for (int t = 0; t < numFrames; ++t) {
                    // Multiply the real filtered time series by currentAlpha before storing
                    filteredBatch[t].at<cv::Vec3f>(r, c)[ch] = filteredTimeSeriesReal.at<float>(t, 0) * currentAlpha;
                }
            } // channels
        } // cols
    } // rows

    return filteredBatch;
}


// --- reconstructGaussianFrame ---
// TDD_ANCHOR: test_reconstructGaussianFrame_matches_python
cv::Mat reconstructGaussianFrame(
    const cv::Mat& originalRgb,
    const cv::Mat& filteredYiqSignal
) {
    // --- Input Validation ---
    if (originalRgb.empty() || originalRgb.type() != CV_8UC3 ||
       filteredYiqSignal.empty() || filteredYiqSignal.type() != CV_32FC3 ||
       originalRgb.size() != filteredYiqSignal.size()) {
        PRINT_ERROR("reconstructGaussianFrame: Invalid input.");
        return cv::Mat();
    }

    // --- 1. Convert original RGB to YIQ (float) ---
    cv::Mat originalYiq;
    try {
        originalYiq = rgb2yiq(originalRgb);
    } catch (const std::exception& e) {
        PRINT_ERROR("reconstructGaussianFrame: rgb2yiq failed. Error: " + std::string(e.what()));
        return cv::Mat();
    }
     if (originalYiq.empty()) {
        PRINT_ERROR("reconstructGaussianFrame: rgb2yiq returned empty matrix.");
        return cv::Mat();
    }

    // --- 2. Add filtered signal to original YIQ ---
    cv::Mat combinedYiq;
    try {
        cv::add(originalYiq, filteredYiqSignal, combinedYiq); // Element-wise addition
    } catch (const cv::Exception& e) {
        PRINT_ERROR("reconstructGaussianFrame: cv::add failed. Error: " + std::string(e.what()));
        return cv::Mat();
    }

    // --- 3. Convert combined YIQ back to RGB (float) ---
    cv::Mat reconstructedRgbFloat;
    try {
        reconstructedRgbFloat = yiq2rgb(combinedYiq);
    } catch (const std::exception& e) {
        PRINT_ERROR("reconstructGaussianFrame: yiq2rgb failed. Error: " + std::string(e.what()));
        return cv::Mat();
    }
     if (reconstructedRgbFloat.empty()) {
        PRINT_ERROR("reconstructGaussianFrame: yiq2rgb returned empty matrix.");
        return cv::Mat();
    }

    // --- 4. Clip values to [0, 255] ---
    // Use cv::max and cv::min for element-wise clipping
    // Note: OpenCV's convertTo also performs saturation, but explicit clipping matches Python better.
    cv::Mat lowerBound = cv::Mat::zeros(reconstructedRgbFloat.size(), reconstructedRgbFloat.type());
    cv::Mat upperBound = cv::Mat(reconstructedRgbFloat.size(), reconstructedRgbFloat.type(), cv::Scalar::all(255.0f));
    cv::max(reconstructedRgbFloat, lowerBound, reconstructedRgbFloat); // reconstructed = max(reconstructed, 0)
    cv::min(reconstructedRgbFloat, upperBound, reconstructedRgbFloat); // reconstructed = min(reconstructed, 255)

    // --- 5. Convert to uint8 (CV_8UC3) ---
    cv::Mat reconstructedRgbUint8;
    try {
        reconstructedRgbFloat.convertTo(reconstructedRgbUint8, CV_8UC3); // Handles rounding and saturation
    } catch (const cv::Exception& e) {
        PRINT_ERROR("reconstructGaussianFrame: convertTo CV_8UC3 failed. Error: " + std::string(e.what()));
        return cv::Mat();
    }

    return reconstructedRgbUint8;
}

} // namespace evmcpp