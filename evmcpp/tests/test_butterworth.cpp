#include <gtest/gtest.h>
#include "evmcpp/butterworth.hpp" // Header for the function under test
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath> // For std::fabs

// --- Test Helper Functions ---

// Function to load a 1D vector from a CSV text file (single row)
std::vector<double> loadVectorFromTxt(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open test data file: " + filename);
    }

    std::vector<double> data;
    std::string line;

    // Expecting only one line for coefficient vectors
    if (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        while (std::getline(ss, value_str, ',')) {
            try {
                data.push_back(std::stod(value_str)); // Use stod for double
            } catch (const std::invalid_argument& e) {
                throw std::runtime_error("Invalid number format in file " + filename + ": " + value_str);
            } catch (const std::out_of_range& e) {
                 throw std::runtime_error("Number out of range in file " + filename + ": " + value_str);
            }
        }
    } else {
         throw std::runtime_error("File is empty or could not read line: " + filename);
    }

    if (file.peek() != EOF && std::getline(file, line) && !line.empty()) {
         // Check if there's more than one line of data
         throw std::runtime_error("File contains more than one line of data: " + filename);
    }


    return data;
}

// Function to compare two double vectors element-wise with tolerance
::testing::AssertionResult CompareVectors(const std::vector<double>& vec1, const std::vector<double>& vec2, double tolerance = 1e-8) {
    if (vec1.size() != vec2.size()) {
        return ::testing::AssertionFailure() << "Vector sizes mismatch: "
               << vec1.size() << " vs " << vec2.size();
    }

    double max_diff = 0.0;
    for (size_t i = 0; i < vec1.size(); ++i) {
        double diff = std::fabs(vec1[i] - vec2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
        if (diff > tolerance) {
             return ::testing::AssertionFailure() << "Vectors differ at index " << i
                    << " (" << vec1[i] << " vs " << vec2[i] << ", diff=" << diff
                    << ") by more than tolerance (" << tolerance << "). Max difference: " << max_diff;
        }
    }
     // If loop completes without failure, report max difference for info (optional)
     // std::cout << "[ INFO ] Max difference for vector comparison: " << max_diff << std::endl;
    return ::testing::AssertionSuccess();
}


// --- Test Fixture ---
class ButterworthTest : public ::testing::Test {
protected:
    const std::string data_dir = "data/"; // Relative to build/tests directory
    const double test_fs = 30.0;
    const double low_cutoff = 0.4;
    const double high_cutoff = 3.0; // Used for the 'high' test which uses lowpass design
    const int test_order = 1;

    // Reference coefficients
    std::vector<double> ref_b_low, ref_a_low;
    std::vector<double> ref_b_high, ref_a_high;


    void SetUp() override {
        // Load reference data
        try {
            ref_b_low = loadVectorFromTxt(data_dir + "butter_low_b.txt");
            ref_a_low = loadVectorFromTxt(data_dir + "butter_low_a.txt");
            ref_b_high = loadVectorFromTxt(data_dir + "butter_high_b.txt");
            ref_a_high = loadVectorFromTxt(data_dir + "butter_high_a.txt");
        } catch (const std::exception& e) {
            GTEST_FAIL() << "Failed to load Butterworth test data: " << e.what();
        }
    }
};

// --- Test Cases ---

TEST_F(ButterworthTest, LowPassCoeffs) {
    ASSERT_FALSE(ref_b_low.empty());
    ASSERT_FALSE(ref_a_low.empty());

    std::pair<std::vector<double>, std::vector<double>> result;
    ASSERT_NO_THROW(result = evmcpp::calculateButterworthCoeffs(test_order, low_cutoff, "low", test_fs));

    EXPECT_TRUE(CompareVectors(result.first, ref_b_low)) << "Numerator (b) coefficients mismatch for low-pass.";
    EXPECT_TRUE(CompareVectors(result.second, ref_a_low)) << "Denominator (a) coefficients mismatch for low-pass.";
}

TEST_F(ButterworthTest, HighPassCoeffs) {
    // Note: Python code uses lowpass design for the high frequency cutoff
    ASSERT_FALSE(ref_b_high.empty());
    ASSERT_FALSE(ref_a_high.empty());

    std::pair<std::vector<double>, std::vector<double>> result;
    // We test the C++ 'low' pass design function with the high cutoff frequency,
    // matching how the Python code generated the reference data.
    ASSERT_NO_THROW(result = evmcpp::calculateButterworthCoeffs(test_order, high_cutoff, "low", test_fs));

    EXPECT_TRUE(CompareVectors(result.first, ref_b_high)) << "Numerator (b) coefficients mismatch for high-pass (using lowpass design).";
    EXPECT_TRUE(CompareVectors(result.second, ref_a_high)) << "Denominator (a) coefficients mismatch for high-pass (using lowpass design).";
}

// TODO: Add tests for actual high-pass, band-pass, band-stop once implemented