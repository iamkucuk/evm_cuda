#include "cuda_evm.cuh"
#include <iostream>
#include <string>
#include <cstdlib> // For atoi, atof
#include <stdexcept>

void print_usage(const char* program_name) {
    std::cout << "Eulerian Video Magnification (CUDA Implementation)" << std::endl;
    std::cout << "Usage: " << program_name << " [options]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  -i, --input <file>       Input video file (required)" << std::endl;
    std::cout << "  -o, --output <file>      Output video file (required)" << std::endl;
    std::cout << "  -l, --levels <int>       Number of pyramid levels [default: 4]" << std::endl;
    std::cout << "  -a, --alpha <float>      Magnification factor [default: 10]" << std::endl;
    std::cout << "  -c, --cutoff <float>     Spatial wavelength cutoff [default: 16]" << std::endl;
    std::cout << "  -fl, --freq-low <float>  Low frequency cutoff for bandpass [default: 0.05]" << std::endl;
    std::cout << "  -fh, --freq-high <float> High frequency cutoff for bandpass [default: 0.4]" << std::endl;
    std::cout << "  -ca, --chrom-att <float> Chrominance attenuation [default: 0.1]" << std::endl;
    std::cout << "  -h, --help               Display this help message" << std::endl;
}

int main(int argc, char* argv[]) {
    // Default parameters
    std::string input_filename;
    std::string output_filename;
    int pyramid_levels = 4;
    double alpha = 10.0;
    double lambda_cutoff = 16.0;
    double fl = 0.05;
    double fh = 0.4;
    double chrom_attenuation = 0.1;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "-i" || arg == "--input") {
            if (i + 1 < argc) {
                input_filename = argv[++i];
            } else {
                std::cerr << "Error: -i/--input requires a filename argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                output_filename = argv[++i];
            } else {
                std::cerr << "Error: -o/--output requires a filename argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-l" || arg == "--levels") {
            if (i + 1 < argc) {
                pyramid_levels = atoi(argv[++i]);
            } else {
                std::cerr << "Error: -l/--levels requires an integer argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-a" || arg == "--alpha") {
            if (i + 1 < argc) {
                alpha = atof(argv[++i]);
            } else {
                std::cerr << "Error: -a/--alpha requires a float argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-c" || arg == "--cutoff") {
            if (i + 1 < argc) {
                lambda_cutoff = atof(argv[++i]);
            } else {
                std::cerr << "Error: -c/--cutoff requires a float argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-fl" || arg == "--freq-low") {
            if (i + 1 < argc) {
                fl = atof(argv[++i]);
            } else {
                std::cerr << "Error: -fl/--freq-low requires a float argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-fh" || arg == "--freq-high") {
            if (i + 1 < argc) {
                fh = atof(argv[++i]);
            } else {
                std::cerr << "Error: -fh/--freq-high requires a float argument" << std::endl;
                return 1;
            }
        }
        else if (arg == "-ca" || arg == "--chrom-att") {
            if (i + 1 < argc) {
                chrom_attenuation = atof(argv[++i]);
            } else {
                std::cerr << "Error: -ca/--chrom-att requires a float argument" << std::endl;
                return 1;
            }
        }
        else {
            std::cerr << "Error: Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    
    // Validate required parameters
    if (input_filename.empty()) {
        std::cerr << "Error: Input filename is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    if (output_filename.empty()) {
        std::cerr << "Error: Output filename is required" << std::endl;
        print_usage(argv[0]);
        return 1;
    }
    
    // Validate parameter ranges
    if (pyramid_levels < 1) {
        std::cerr << "Error: Number of pyramid levels must be at least 1" << std::endl;
        return 1;
    }
    
    if (alpha <= 0.0) {
        std::cerr << "Error: Alpha must be positive" << std::endl;
        return 1;
    }
    
    if (lambda_cutoff <= 0.0) {
        std::cerr << "Error: Lambda cutoff must be positive" << std::endl;
        return 1;
    }
    
    if (fl < 0.0 || fl >= fh) {
        std::cerr << "Error: Low frequency must be non-negative and less than high frequency" << std::endl;
        return 1;
    }
    
    if (fh <= 0.0 || fh > 1.0) {
        std::cerr << "Error: High frequency must be between 0 and 1" << std::endl;
        return 1;
    }
    
    if (chrom_attenuation < 0.0 || chrom_attenuation > 1.0) {
        std::cerr << "Error: Chrominance attenuation must be between 0 and 1" << std::endl;
        return 1;
    }
    
    // Print processing parameters
    std::cout << "=== Eulerian Video Magnification (CUDA) ===" << std::endl;
    std::cout << "Input file:            " << input_filename << std::endl;
    std::cout << "Output file:           " << output_filename << std::endl;
    std::cout << "Number of levels:      " << pyramid_levels << std::endl;
    std::cout << "Alpha (amplification): " << alpha << std::endl;
    std::cout << "Lambda cutoff:         " << lambda_cutoff << std::endl;
    std::cout << "Frequency range:       [" << fl << ", " << fh << "]" << std::endl;
    std::cout << "Chrom. attenuation:    " << chrom_attenuation << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        // Process the video
        evmcuda::process_video_laplacian(
            input_filename,
            output_filename,
            pyramid_levels,
            alpha,
            lambda_cutoff,
            fl,
            fh,
            chrom_attenuation
        );
        
        std::cout << "Processing complete! Output saved to: " << output_filename << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during processing: " << e.what() << std::endl;
        return 1;
    }
}