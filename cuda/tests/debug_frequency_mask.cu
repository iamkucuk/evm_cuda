#include <iostream>
#include <cmath>

int main() {
    std::cout << "=== Frequency Mask Debug ===" << std::endl;
    
    const float fps = 30.0f;
    const float fl = 0.8f;
    const float fh = 1.0f;
    const int dft_size = 512;
    
    std::cout << "Parameters: fps=" << fps << ", range=" << fl << "-" << fh << " Hz" << std::endl;
    std::cout << "DFT size: " << dft_size << ", R2C bins: " << (dft_size/2 + 1) << std::endl;
    
    std::cout << "\nFrequency analysis for R2C format:" << std::endl;
    std::cout << "Bin\tFreq (Hz)\tAction" << std::endl;
    
    for (int i = 0; i <= dft_size/2; i++) {
        float freq = (float)i * fps / (float)dft_size;
        bool should_pass = (freq >= fl && freq <= fh);
        
        std::cout << i << "\t" << freq << "\t\t" << (should_pass ? "PASS" : "BLOCK");
        
        // Mark critical frequencies
        if (i == 0) std::cout << " (DC)";
        if (freq >= 0.79 && freq <= 0.81) std::cout << " (near fl)";
        if (freq >= 0.99 && freq <= 1.01) std::cout << " (near fh)";
        if (freq >= 1.19 && freq <= 1.21) std::cout << " (test freq)";
        
        std::cout << std::endl;
    }
    
    // Find specific test frequencies
    std::cout << "\nSpecific test frequency analysis:" << std::endl;
    float test_freqs[] = {0.0f, 0.5f, 0.8f, 0.9f, 1.0f, 1.2f, 15.0f};
    
    for (float test_freq : test_freqs) {
        float exact_bin = test_freq * dft_size / fps;
        int bin = (int)round(exact_bin);
        float actual_freq = (float)bin * fps / (float)dft_size;
        bool should_pass = (test_freq >= fl && test_freq <= fh);
        bool would_pass = (actual_freq >= fl && actual_freq <= fh);
        
        std::cout << "Target " << test_freq << " Hz -> bin " << exact_bin 
                 << " (rounded to " << bin << ") -> actual " << actual_freq << " Hz";
        std::cout << " | Expected: " << (should_pass ? "PASS" : "BLOCK");
        std::cout << " | Actual: " << (would_pass ? "PASS" : "BLOCK");
        if (should_pass != would_pass) std::cout << " ❌ MISMATCH";
        std::cout << std::endl;
    }
    
    return 0;
}