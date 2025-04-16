# ------------------------------------------------------------------------------
# Stage 1: Build OpenCV with optional CUDA support
# ------------------------------------------------------------------------------

# Choose the base image (NVIDIA CUDA images only needed for GPU builds)
# Select appropriate tags for CUDA version, cuDNN version, and OS.
# Using 'devel' tag as it includes compiler (nvcc) and headers.
# Find tags here: https://hub.docker.com/r/nvidia/cuda/tags
# ARG CUDA_VERSION=11.8.0
# ARG CUDNN_VERSION=8
ARG OS_VERSION=ubuntu20.04
# FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${OS_VERSION} AS builder
FROM ubuntu:${OS_VERSION} AS builder

# Set ARG for OpenCV version (can be overridden during build)
ARG OPENCV_VERSION=4.8.0

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# Install essential build tools and OpenCV dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    pkg-config \
    # Image I/O libs
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev \
    libopenexr-dev \
    # Video I/O libs (FFmpeg)
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavresample-dev \
    libv4l-dev \
    # Parallelism and linear algebra
    libtbb-dev \
    libeigen3-dev \
    libatlas-base-dev \
    gfortran \
    # GUI (optional, but sometimes required by dependencies)
    libgtk-3-dev \
    # Python
    python3-dev \
    python3-pip \
    python3-numpy \
    # Other utils
    ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install common Python packages (optional, but useful for development)
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir numpy # Ensure numpy is installed via pip

# Download OpenCV and OpenCV Contrib source code
WORKDIR /opt
RUN wget https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -O opencv.zip && \
    wget https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -O opencv_contrib.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    rm opencv.zip opencv_contrib.zip && \
    mv opencv-${OPENCV_VERSION} opencv && \
    mv opencv_contrib-${OPENCV_VERSION} opencv_contrib

# Configure OpenCV build with CMake
# IMPORTANT: Adjust CUDA_ARCH_BIN to match the Compute Capability of your target GPU(s (if building with CUDA))
#            Find your GPU's Compute Capability: https://developer.nvidia.com/cuda-gpus (optional)
#            Examples: Pascal (6.x), Volta (7.0), Turing (7.5), Ampere (8.0, 8.6), Ada (8.9), Hopper (9.0)
#            Providing multiple archs increases compatibility but also build time/size.
WORKDIR /opt/opencv/build

# ***** CORRECTED SECTION *****
RUN cmake \
    -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    # --- CUDA Specific --- (remove or comment out for CPU-only builds)
    #    -D WITH_CUDA=ON \
    #    -D OPENCV_DNN_CUDA=ON \
    #    -D ENABLE_FAST_MATH=1 \
    #    -D CUDA_FAST_MATH=1 \
    #    -D CUDA_ARCH_BIN="8.6" \
    #    -D CUDA_ARCH_PTX="" \
    # --- Modules & Features ---
    -D OPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib/modules \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D WITH_TBB=ON \
    -D WITH_V4L=ON \
    -D WITH_OPENGL=ON \
    -D WITH_FFMPEG=ON \
    # --- Build Options ---
    -D BUILD_EXAMPLES=OFF \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_DOCS=OFF \
    # --- Python Bindings ---
    -D BUILD_opencv_python3=ON \
    -D PYTHON3_EXECUTABLE=$(which python3) \
    -D PYTHON3_INCLUDE_PATH=$(python3 -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())") \
    ..
# ***** END CORRECTED SECTION *****

# Build and install OpenCV
# Use all available CPU cores for compilation (-j$(nproc))
RUN make -j$(nproc) && make install && ldconfig

# ------------------------------------------------------------------------------
# Stage 2: Final Development Environment
# ------------------------------------------------------------------------------

# Use the same base image to ensure compatibility
# FROM nvidia/cuda:${CUDA_VERSION}-cudnn${CUDNN_VERSION}-devel-${OS_VERSION} AS final
FROM ubuntu:${OS_VERSION} AS final

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
# Ensure CUDA paths are set (usually handled by base image, but explicit is safe)
# ENV PATH=/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# Install runtime dependencies needed by OpenCV or for general development
# (Subset of build dependencies + any runtime specific needs)
# Adjust package versions if needed for your OS (e.g., libopenexr, libavcodec)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Runtime libs for OpenCV compiled features
    libjpeg8 \
    libpng16-16 \
    libtiff5 \
    libwebp6 \
    libopenexr24 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libavresample4 \
    libgtk-3-0 \
    libtbb2 \
    libv4l-0 \
    # Python and common tools
    python3 \
    python3-pip \
    python3-numpy \
    git \
    vim \
    tmux \
    ca-certificates \
    make \
    cmake \
    build-essential \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Copy built OpenCV libraries and headers from the builder stage
COPY --from=builder /usr/local /usr/local/

# Refresh library cache
RUN ldconfig

# Install Python packages (can add more here)
# Note: numpy might already be installed by apt or copied python bindings.
# Using pip ensures specific versions if needed.
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir \
    numpy
    # Add other useful python packages:
    # matplotlib \
    # scipy \
    # pandas \
    # scikit-learn \
    # scikit-image \
    # jupyterlab

# Set default working directory
WORKDIR /workspace

# Set the default command to bash (useful for development)
CMD ["/bin/bash"]