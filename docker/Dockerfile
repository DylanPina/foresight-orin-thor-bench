# =============================================================================
# Base Image and Build Arguments
#  - Docker container must match jetpack version installed on Jetson
#    - Image list: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/l4t-base/tags
#  - ROS2 Humble is used for Jetson Orin OpenCV and Python version compatibility
#  - For native builds: ARM64/Jetson uses l4t-jetpack, AMD64 uses Ubuntu 22.04
# =============================================================================

# Build arguments
ARG ros_distro=humble
ARG gid
ARG uid
ARG username
ARG install_cuda=false
ARG is_jetson=false
ARG BASE_STAGE_NAME=base_amd64

# =============================================================================
# Stage 1: Platform-Specific Base Images
# =============================================================================
# For Robot (ARM64/Jetson): Use NVIDIA JetPack base image
# Note: Stage name uses "arm64" to match Docker's TARGETARCH variable
ARG BASE_IMAGE_ARM64=nvcr.io/nvidia/l4t-jetpack:r36.4.0
FROM ${BASE_IMAGE_ARM64} AS base_arm64
# JetPack container: includes CUDA, cuDNN, TensorRT, VPI, Jetson Multimedia, and so on
# Alternative ML container with PyTorch, TensorFlow pre-installed:
# FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3 AS base-arm64

ARG is_jetson
ARG install_cuda

# NOTE that while the package repositories are not enabled inside the container, jetpack is already installed.
# To get a list of all packages, check the host by running:
# cat /var/lib/apt/lists/repo.download.nvidia.com_jetson_common_dists_r36.4_main_binary-arm64_Packages |grep cudnn
# Install the jetson package repository (only on actual Jetson devices)
RUN if [ "$is_jetson" = "true" ]; then \
        apt-key adv --fetch-key https://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
        echo "deb https://repo.download.nvidia.com/jetson/common r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
        echo "deb https://repo.download.nvidia.com/jetson/t234 r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list && \
        echo "deb https://repo.download.nvidia.com/jetson/ffmpeg r36.4 main" >> /etc/apt/sources.list.d/nvidia-l4t-apt-source.list; \
    fi

# Install cuDNN (only when CUDA is available, required for PyTorch)
# NOTE cudnn8 is not compatible with latest PyTorch on Jetson, so we install cudnn9
RUN if [ "$install_cuda" = "true" ]; then \
        apt-get update && apt-get install -y \
        libcudnn9 \
        libcudnn9-dev \
        && rm -rf /var/lib/apt/lists/*; \
    fi

# For Desktop (AMD64/x86_64): Use Ubuntu 22.04 with optional CUDA
# Note: Stage name uses "amd64" to match Docker's TARGETARCH variable
FROM ubuntu:22.04 AS base_amd64

ARG install_cuda

# Install CUDA toolkit for Desktop only if install_cuda is true
# This provides GPU support on Desktop machines with NVIDIA GPUs
COPY scripts/detect_gpu_setup.sh /tmp/detect_gpu_setup.sh
RUN if [ "$install_cuda" = "true" ]; then \
        . /tmp/detect_gpu_setup.sh && \
        apt-get update && apt-get install -y \
            wget \
            gnupg2 \
            && rm -rf /var/lib/apt/lists/* && \
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        rm cuda-keyring_1.1-1_all.deb && \
        apt-get update && apt-get install -y \
            ${CUDA_TOOLKIT_PKG} \
            libcudnn8 \
            libcudnn8-dev \
            && rm -rf /var/lib/apt/lists/*; \
    fi

# =============================================================================
# Stage 2: Select Base - Auto-detects platform during native build
# =============================================================================
# When building natively on Robot (ARM64), this will pull from base-arm64
# When building natively on Desktop (AMD64), this will pull from base-amd64
ARG BASE_STAGE_NAME
FROM $BASE_STAGE_NAME AS base-selected

ARG ros_distro=humble
ARG gid
ARG uid
ARG username

# =============================================================================
# Stage 3: Base Setup - Locale and Essential Tools
# =============================================================================
FROM base-selected AS base-setup

# Set non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive

# Setup Locale
RUN apt-get update && apt-get install -y locales && \
    locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US:en
ENV LC_ALL=en_US.UTF-8

# Install base dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    gnupg2 \
    lsb-release \
    iproute2 \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Stage 4: ROS2 Installation
# =============================================================================
FROM base-setup AS ros2-install

ARG ros_distro
ARG is_jetson

# Setup Isaac ROS2 APT repository (ARM64 only, will fail gracefully on AMD64)
RUN if [ "$is_jetson" = "true" ]; then \
  wget -qO - https://isaac.download.nvidia.com/isaac-ros/repos.key | apt-key add - && \
  grep -qxF "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" /etc/apt/sources.list || \
  echo "deb https://isaac.download.nvidia.com/isaac-ros/release-3 $(lsb_release -cs) release-3.0" | tee -a /etc/apt/sources.list || true; \
fi

# Setup Official ROS2 APT repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install Core ROS 2 Humble packages
RUN apt-get update && apt-get install -y \
    ros-${ros_distro}-desktop-full \
    ros-${ros_distro}-rmw-cyclonedds-cpp \
    ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Isaac ROS common dependencies and VPI (ARM64 only, skip on AMD64)
# Note: VPI libraries are also part of the Jetson host system and will be mounted.
# This ensures any additional dependencies are met inside the container.
RUN if [ "$is_jetson" = "true" ]; then \
    apt-get update && \
    (apt-get install -y ros-${ros_distro}-isaac-ros-common || echo "Isaac ROS not available on this platform") && \
    rm -rf /var/lib/apt/lists/*; \
fi

# Initialize rosdep
RUN rosdep init && \
    rosdep update

# =============================================================================
# Stage 5: Hardware Dependencies (Cameras, Sensors)
# =============================================================================
FROM ros2-install AS hardware-deps

# Re-declare build arguments for this stage
ARG install_cuda
ARG is_jetson
COPY scripts/install_realsense_library.sh external* /tmp/librealsense/
# RUN if [ "$is_jetson" = "true" ]; then \
#   if [ "$install_cuda" = "true" ]; then \
#     apt-get update && apt-get install -y kmod && \
#     /tmp/librealsense/install_realsense_library.sh && \
#     rm /tmp/librealsense/install_realsense_library.sh; \
#   else \
#     echo "Skipping Realsense installation (CUDA not found)"; \
#   fi; \
# else \
#   echo "Skipping Realsense installation (Not on Jetson)"; \
# fi

# Azure Kinect SDK
COPY scripts/install_azure_kinect_sdk.sh /tmp/install_azure_kinect_sdk.sh
RUN /tmp/install_azure_kinect_sdk.sh && rm /tmp/install_azure_kinect_sdk.sh

# MPU6050 Dependencies
RUN if [ "$is_jetson" = "true" ]; then \
  apt-get update && apt-get install -y \
  libi2c-dev \
  && rm -rf /var/lib/apt/lists/*; \
fi

# =============================================================================
# Stage 6: Application Dependencies
# =============================================================================
FROM hardware-deps AS app-deps

# Audio capture dependencies (for spot_audio: Azure Kinect mic array)
# RUN apt-get update && apt-get install -y \
#     libportaudio2 \
#     alsa-utils \
#     && rm -rf /var/lib/apt/lists/*
RUN apt-get update && apt-get install -y \
    libportaudio2 \
    portaudio19-dev \
    alsa-utils \
    && rm -rf /var/lib/apt/lists/*


# =============================================================================
# VectorNav VN-100/200/300 ROS 2 driver (build from source)
# =============================================================================
ENV VN_WS=/opt/vn_ws

# Clone ROS 2 branch, resolve deps, build
RUN set -e \
 && mkdir -p "$VN_WS/src" \
 && cd "$VN_WS/src" \
 && git clone -b ros2_fast --depth 1 https://github.com/ut-amrl/vectornav.git \
 && cd "$VN_WS" \
 && . /opt/ros/${ros_distro}/setup.sh \
 && rosdep update \
 && rosdep install --from-paths src --ignore-src -y -r \
 && colcon build --merge-install --cmake-args -DCMAKE_BUILD_TYPE=Release

# Source the overlay in login + interactive shells
RUN printf '# VectorNav overlay\n[ -f %s/local_setup.bash ] && . %s/local_setup.bash\n' \
           "$VN_WS/install" "$VN_WS/install" \
      | tee /etc/profile.d/10-vectornav.sh >/dev/null \
 && chmod 0644 /etc/profile.d/10-vectornav.sh \
 && printf '\n# VectorNav overlay\n[ -f %s/local_setup.bash ] && . %s/local_setup.bash\n' \
           "$VN_WS/install" "$VN_WS/install" \
      >> /etc/bash.bashrc

# =============================================================================
# Spot Dependencies & OpenCV
# =============================================================================
COPY scripts/install_spot_ros2.sh /tmp/install_spot_ros2.sh
COPY scripts/spot_wrapper_requirements.txt /tmp/spot_wrapper_requirements.txt

# Prefer ROS/Ubuntu OpenCV (matches cv_bridge) and block pip wheels
RUN apt-get update && apt-get install -y \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i -E '/^\s*opencv-(contrib-)?python(-headless)?([<=>].*)?\s*$/d' /tmp/spot_wrapper_requirements.txt

# Run the Spot install script
RUN /tmp/install_spot_ros2.sh --arm64

# Safety net: remove any OpenCV wheels that might have been pulled indirectly
RUN python3 -m pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# Additional ROS2 Packages (Localization, Transport)
RUN apt-get update && apt-get install -y \
    ros-${ros_distro}-bondpy \
    ros-${ros_distro}-controller-interface \
    ros-${ros_distro}-forward-command-controller \
    ros-${ros_distro}-imu-sensor-broadcaster \
    ros-${ros_distro}-robot-localization \
    ros-${ros_distro}-point-cloud-transport \
    ros-${ros_distro}-velodyne \
    ros-${ros_distro}-image-transport \
    ros-${ros_distro}-image-transport-plugins \
    && rm -rf /var/lib/apt/lists/*

# Serial dependencies
RUN apt-get update && apt-get install -y \
    python3-serial libserialport0 \
    && rm -rf /var/lib/apt/lists/*

# GStreamer (for camera streaming)
RUN apt-get update && apt-get install -y \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  gstreamer1.0-tools \
  gstreamer1.0-x \
  && rm -rf /var/lib/apt/lists/*

# Azure Kinect ROS Driver
WORKDIR /opt/azure_kinect_ws
RUN apt-get update && \
    mkdir -p src && \
    cd src && \
    git clone -b humble https://github.com/microsoft/Azure_Kinect_ROS_Driver.git && \
    cd .. && \
    . /opt/ros/${ros_distro}/setup.sh && \
    rosdep install --from-paths src --ignore-src -r -y && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release && \
    rm -rf /var/lib/apt/lists/*

# Source Azure Kinect Driver in bashrc
RUN echo "source /opt/azure_kinect_ws/install/setup.bash" >> /etc/bash.bashrc

# EGL, OpenGL libraries (for GUI applications)
RUN apt-get update && apt-get install -y \
  libegl1 \
  libegl1-mesa \
  libgles2-mesa \
  libgles2 \
  libglvnd-dev \
  libgl1-mesa-glx \
  libgl1-mesa-dri \
  mesa-utils-extra \
  && rm -rf /var/lib/apt/lists/*

# Install Xfce4 and TurboVNC
RUN apt-get update && apt-get install -y \
    xfce4 \
    xfce4-goodies \
    dbus-x11 \
    && rm -rf /var/lib/apt/lists/*

ARG TARGETARCH
RUN if [ "$TARGETARCH" = "amd64" ]; then \
      wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_amd64.deb -O /tmp/turbovnc.deb; \
    elif [ "$TARGETARCH" = "arm64" ]; then \
      wget https://github.com/TurboVNC/turbovnc/releases/download/3.2.1/turbovnc_3.2.1_arm64.deb -O /tmp/turbovnc.deb; \
    fi && \
    dpkg -i /tmp/turbovnc.deb && \
    rm /tmp/turbovnc.deb && \
    ln -s /opt/TurboVNC/bin/* /usr/local/bin/

# Copy VNC startup script
COPY scripts/vnc /usr/local/bin/vnc
RUN chmod +x /usr/local/bin/vnc

# =============================================================================
# Stage 7: Python and ML Dependencies
# =============================================================================
FROM app-deps AS python-ml

ARG install_cuda
ARG PT_VER=2.9.1
ARG VIS_VER=0.24.1
ARG AUD_VER=2.9.1

# pip
RUN apt-get update && apt-get install -y \
  python3-pip \
  python3-venv \
  python3-setuptools \
  python3-wheel \
  python3-pyqt5 \
  python3-colcon-clean \
  libgfortran5 \
  && rm -rf /var/lib/apt/lists/*
#RUN python3 -m ensurepip --default-pip && python3 -m pip install --upgrade pip

RUN pip3 install --no-cache-dir sounddevice==0.5.1

# pytorch (CUDA 12.6 / vLLM 0.16 compatible versions)
ARG install_cuda
ARG is_jetson
# if this step fails, run `docker builder prune`
RUN --mount=type=cache,target=/var/cache/downloads --mount=type=cache,target=/root/.cache/pip \
  if [ "$is_jetson" = "true" ]; then \
    # For Jetson: install gfortran and compatible libraries, then use NVIDIA's PyTorch \
    # Installation instructions: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html
    # Available versions are listed by jetpack version here: https://developer.download.nvidia.com/compute/redist/jp/
    if [ ! -f /var/cache/downloads/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb ]; then \
        wget -P /var/cache/downloads https://developer.download.nvidia.com/compute/cusparselt/0.8.1/local_installers/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb; \
    fi && \
    dpkg -i /var/cache/downloads/cusparselt-local-tegra-repo-ubuntu2204-0.8.1_0.8.1-1_arm64.deb && \
    cp /var/cusparselt-local-tegra-repo-ubuntu2204-0.8.1/cusparselt-*-keyring.gpg /usr/share/keyrings/ && \
    apt-get update && apt-get install -y gfortran libopenblas-dev cusparselt && rm -rf /var/lib/apt/lists/* && \
    pip3 install --no-cache-dir torch==${PT_VER} torchvision==${VIS_VER} torchaudio==${AUD_VER} --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    pip3 install --no-cache-dir nvidia-cudss-cu12 --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 && \
    pip3 install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126; \
  elif [ "$install_cuda" = "true" ]; then \
    pip3 install --no-cache-dir torch==${PT_VER}+cu126 torchvision==${VIS_VER} torchaudio==${AUD_VER} --index-url https://download.pytorch.org/whl/cu126 && \
    pip3 install --no-cache-dir nvidia-cudss-cu12; \
  else \
    # CPU Fallback
    pip3 install torch==${PT_VER} torchvision==${VIS_VER} torchaudio==${AUD_VER} --index-url https://download.pytorch.org/whl/cpu; \
  fi

RUN python3 - <<'PY'
from pathlib import Path
import sys

# Search broadly because wheel layouts can vary by build/index.
search_roots = [
    Path("/usr/local/lib/python3.10/dist-packages"),
    Path("/usr/local/lib/python3.10/site-packages"),
    Path("/usr/local/lib"),
    Path("/usr/lib"),
]
targets = []
for root in search_roots:
    if root.exists():
        targets.extend(root.rglob("libcudss.so*"))

# Keep only real files/symlinks, unique and deterministic order.
targets = sorted({p.resolve() if p.is_symlink() else p for p in targets if p.exists()})
if not targets:
    raise RuntimeError(
        "No libcudss shared objects found after installing nvidia-cudss-cu12. "
        "Verify the wheel was installed for aarch64 and from the Jetson index."
    )

for lib in targets:
    out = Path("/usr/local/lib") / lib.name
    if out.exists() or out.is_symlink():
        out.unlink()
    out.symlink_to(lib)
    print(f"linked {out} -> {lib}")
PY
RUN ldconfig

RUN python3 - <<'PY'
import torch
import ctypes.util
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("libcudss found:", ctypes.util.find_library("cudss"))
print("cuda available:", torch.cuda.is_available())
PY

# leg_detector
RUN --mount=type=cache,target=/root/.cache/pip pip3 install pykalman

# =============================================================================
# Stage 8: vLLM and VLM Dependencies
# =============================================================================
FROM python-ml AS vllm-deps

ARG install_cuda
ARG is_jetson

# Build vLLM 0.16.0 from source and install VLM Python deps
RUN apt-get update && apt-get install -y \
  git \
  && rm -rf /var/lib/apt/lists/*

# Triton JIT requires CUDA compiler toolchain (ptxas) at runtime.
# Jetson base images often include CUDA runtime libs but not compiler binaries.
# RUN if [ "$is_jetson" = "true" ]; then \
#       apt-get update && \
#       (apt-get install -y cuda-nvcc-12-6 || apt-get install -y cuda-toolkit-12-6) && \
#       command -v ptxas && ptxas --version && \
#       rm -rf /var/lib/apt/lists/*; \
#     fi

# vLLM 0.16.0 build-time requirements from pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-cache-dir --upgrade \
  "pip<25.0.0" \
  "setuptools>=77.0.3,<80.0.0" \
  wheel \
  "packaging>=24.2" \
  jinja2 \
  "cmake>=3.26" \
  ninja \
  "typing_extensions>=4.12.2"

# Runtime deps for Qwen3-VL + vLLM
RUN --mount=type=cache,target=/root/.cache/pip pip3 install --no-cache-dir \
  "numpy==1.26.4" \
  "networkx>=3.2.1" \
  "scipy>=1.11.4" \
  setuptools_scm \
  "transformers==4.57.1" \
  "qwen-vl-utils==0.0.14" \
  "flashinfer-python==0.6.3" \
  "flashinfer-cubin==0.6.3"

# Triton is required by vllm_flash_attn rotary kernels on this stack.
RUN --mount=type=cache,target=/root/.cache/pip \
  if [ "$is_jetson" = "true" ]; then \
    pip3 install --no-cache-dir "triton==3.5.1" --index-url https://pypi.jetson-ai-lab.io/jp6/cu126; \
  else \
    pip3 install --no-cache-dir "triton==3.5.1"; \
  fi

# CUDA environment variables for JIT compilation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
ENV CPATH=/usr/local/cuda/include:/usr/local/cuda/targets/aarch64-linux/include:${CPATH}
ENV LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:${LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST=8.7
# ENV MAX_JOBS=6

RUN git clone --depth 1 --branch v0.16.0 https://github.com/vllm-project/vllm.git /opt/vllm

WORKDIR /opt/vllm

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "$is_jetson" = "true" ]; then \
      pip3 install --no-cache-dir -r requirements/build.txt --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126 && \
      pip3 install --no-cache-dir -r requirements/common.txt --extra-index-url https://pypi.jetson-ai-lab.io/jp6/cu126; \
    else \
      pip3 install --no-cache-dir -r requirements/build.txt && \
      pip3 install --no-cache-dir -r requirements/common.txt; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    VLLM_TARGET_DEVICE=cuda VLLM_MAIN_CUDA_VERSION=12.6 pip3 install --no-build-isolation .

# Smoke check key runtime imports early so failures happen at build time.
RUN python3 - <<'PY'
import importlib.metadata as metadata
import vllm
import transformers
import qwen_vl_utils
import flashinfer
import triton
print("vllm:", vllm.__version__)
print("transformers:", transformers.__version__)
print("qwen-vl-utils:", metadata.version("qwen-vl-utils"))
print("flashinfer-python:", metadata.version("flashinfer-python"))
print("flashinfer-cubin:", metadata.version("flashinfer-cubin"))
print("triton:", triton.__version__)
PY

# FROM vllm-deps AS nav-slam
# FROM nav-slam AS utilities

# =============================================================================
# Stage 9: Navigation and SLAM Libraries
# =============================================================================
FROM vllm-deps AS nav-slam

ARG ros_distro

# Libraries for graph nav and enml
RUN apt-get update && apt-get install -y \
  libfreeimage3 \
  libfreeimage-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libeigen3-dev \
  libjpeg8-dev \
  libgoogle-perftools-dev \
  libsuitesparse-dev \
  libatlas-base-dev \
  libblas-dev \
  libopencv-dev \
  liblapack-dev \
  libceres-dev \
  libtbb-dev \
  libncurses5-dev \
  libpopt-dev \
  libpcl-dev \
  ros-${ros_distro}-ament-cmake \
  cimg-dev \
  && rm -rf /var/lib/apt/lists/*

# Graph navigation OSRM dependencies
# NOTE: v6.0.0 currently fails to compile with GCC 11 in this environment.
# Use v5.27.1, which is known to build reliably on Ubuntu/GCC 11.

RUN apt-get update && apt-get install -y \
  libbz2-dev \
  libxml2-dev \
  libzip-dev \
  libboost-all-dev \
  lua5.2 \
  liblua5.2-dev \
  libtbb-dev \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
RUN git clone --depth 1 --branch v5.27.1 https://github.com/Project-OSRM/osrm-backend.git && \
    cd osrm-backend && \
    sed -i 's/cmake_policy(SET CMP0048 OLD)/cmake_policy(SET CMP0048 NEW)/' CMakeLists.txt && \
    sed -i 's/^add_warning(error).*/# add_warning(error) # disabled for GCC 11 compatibility/' cmake/warnings.cmake && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Wno-dev && \
    cmake --build . && \
    cmake --build . --target install

# =============================================================================
# iceoryx for shared memory communication (optional)
# =============================================================================
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    git \
    libacl1-dev \
    libncurses5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /tmp
RUN git clone https://github.com/eclipse-iceoryx/iceoryx.git
WORKDIR /tmp/iceoryx
RUN cmake -Bbuild -Hiceoryx_meta -DBUILD_ALL=ON -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target install -j$(nproc)

# =============================================================================
# Superodom dependencies
# =============================================================================
WORKDIR /tmp
RUN git clone http://github.com/strasdat/Sophus.git && \
      cd Sophus && git checkout 97e7161 && \
      mkdir build && cd build && \
      cmake .. -DBUILD_TESTS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
      make -j8 && sudo make install

WORKDIR /tmp
RUN git clone https://github.com/borglab/gtsam.git && \
      cd gtsam && git checkout 4abef92 && \
      mkdir build && cd build && \
      cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
      make -j8 && sudo make install

# Livox-SDK2
WORKDIR /tmp
RUN git clone https://github.com/Livox-SDK/Livox-SDK2.git && \
      cd Livox-SDK2 && \
      mkdir build && cd build && \
      cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
      make -j8 && sudo make install

# =============================================================================
# Webviz dependencies
# =============================================================================
RUN apt-get update && apt-get install -y \
  qtbase5-dev qt5-qmake qtbase5-dev-tools libqt5websockets5-dev \
  python3-colcon-common-extensions libgtest-dev \
  && rm -rf /var/lib/apt/lists/*

# =============================================================================
# LeLAn dependencies
# =============================================================================
RUN pip3 install \
  openai-clip \
  vit_pytorch

# clone and pip install diffusion policy (https://github.com/real-stanford/diffusion_policy.git)
WORKDIR /tmp
RUN git clone https://github.com/real-stanford/diffusion_policy.git
WORKDIR /tmp/diffusion_policy
RUN pip install -e .

# =============================================================================
# Stage 10: Utilities and Additional Scripts
# =============================================================================
FROM nav-slam AS utilities
# FROM vllm-deps AS utilities

ARG is_jetson

# Utils and additional dependencies
RUN apt-get update && apt-get install -y \
  vim \
  tmux \
  tmuxinator \
  && rm -rf /var/lib/apt/lists/*

# Jetson-stats (ARM64 only, will fail gracefully on AMD64)
RUN --mount=type=cache,target=/root/.cache/pip if [ "$is_jetson" = "true" ]; then \
    pip3 install jetson-stats || echo "jetson-stats not available on this platform"; \
fi

# Visualization dependencies
RUN apt-get update && apt-get install -y \
    ros-${ros_distro}-foxglove-bridge \
    ros-${ros_distro}-foxglove-msgs \
    ros-${ros_distro}-visualization-msgs \
    ros-${ros_distro}-grid-map-msgs \
    ros-${ros_distro}-grid-map-core \
    ros-${ros_distro}-grid-map-ros \
    ros-${ros_distro}-tf-transformations \
    && rm -rf /var/lib/apt/lists/*

# ROS2 and other dependencies installation scripts
COPY scripts/ros2_deps.sh /tmp/ros2_deps.sh
RUN /tmp/ros2_deps.sh && rm /tmp/ros2_deps.sh

# Installs cotnav dependencies
COPY scripts/install_cotnav_deps.sh /tmp/install_cotnav_deps.sh
RUN /tmp/install_cotnav_deps.sh && rm /tmp/install_cotnav_deps.sh


# =============================================================================
# Stage 11: Final - User Creation and Configuration
# =============================================================================
FROM utilities AS final

ARG uid
ARG gid
ARG username
ARG ros_distro

ENV UID=$uid
ENV GID=$gid
ENV USERNAME=$username
ENV ROS_DISTRO=$ros_distro

# Create user and group, then configure sudo
RUN set -e && \
    # Create group if it doesn't exist
    getent group ${GID} || groupadd -g ${GID} ${USERNAME} && \
    # Handle user creation/modification
    if id ${UID} >/dev/null 2>&1; then \
        # User with this UID exists, modify it
        existing_user=$(id -nu ${UID}) && \
        usermod -l ${USERNAME} -d /home/${USERNAME} -m $existing_user 2>/dev/null || true && \
        usermod -g ${GID} ${USERNAME} 2>/dev/null || true; \
    else \
        # Create new user
        useradd -u ${UID} -g ${GID} -m -s /bin/bash ${USERNAME}; \
    fi && \
    # Ensure home directory exists and has correct ownership
    mkdir -p /home/${USERNAME} && \
    chown -R ${UID}:${GID} /home/${USERNAME} && \
    # Add to sudo group and configure passwordless sudo
    usermod -aG sudo ${USERNAME} && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    visudo -c && \
    # Add the user to the video group for access to video devices
    usermod -aG video ${USERNAME}

# Create groups matching the host's group IDs, handling existing groups
RUN set -e && \
    # Handle i2c group (host GID: 116)
    if getent group i2c >/dev/null 2>&1 && getent group 116 >/dev/null 2>&1; then \
        # Both i2c name and GID 116 exist - check if they're the same group
        existing_group=$(getent group 116 | cut -d: -f1) && \
        if [ "$existing_group" != "i2c" ]; then \
            # Different groups: delete the one with wrong GID, rename the one at 116
            groupdel i2c && \
            groupmod -n i2c $existing_group; \
        fi; \
    elif getent group 116 >/dev/null 2>&1; then \
        # Group with GID 116 exists, rename it to i2c
        existing_group=$(getent group 116 | cut -d: -f1) && \
        groupmod -n i2c $existing_group; \
    elif getent group i2c >/dev/null 2>&1; then \
        # i2c group exists with different GID, change its GID to 116
        groupmod -g 116 i2c; \
    else \
        # Create new i2c group
        groupadd -g 116 i2c; \
    fi && \
    # Handle dialout group (host GID: 20)
    if getent group dialout >/dev/null 2>&1 && getent group 20 >/dev/null 2>&1; then \
        # Both dialout name and GID 20 exist - check if they're the same group
        existing_group=$(getent group 20 | cut -d: -f1) && \
        if [ "$existing_group" != "dialout" ]; then \
            # Different groups: delete the one with wrong GID, rename the one at 20
            groupdel dialout && \
            groupmod -n dialout $existing_group; \
        fi; \
    elif getent group 20 >/dev/null 2>&1; then \
        # Group with GID 20 exists, rename it to dialout
        existing_group=$(getent group 20 | cut -d: -f1) && \
        groupmod -n dialout $existing_group; \
    elif getent group dialout >/dev/null 2>&1; then \
        # dialout group exists with different GID, change its GID to 20
        groupmod -g 20 dialout; \
    else \
        # Create new dialout group
        groupadd -g 20 dialout; \
    fi && \
    # Handle input group (host GID: 104 or similar, for joystick/gamepad access)
    # Note: The input group GID varies by system, we'll create it if it doesn't exist
    if ! getent group input > /dev/null 2>&1; then \
        groupadd -r input; \
    fi && \
    # Add user to i2c, dialout, and input groups
    usermod -aG i2c,dialout,input,audio ${USERNAME}

# Fix permissions for Azure Kinect workspace
# This allows the non-root user to generate the URDF file at runtime
RUN chown -R ${UID}:${GID} /opt/azure_kinect_ws

# Copy bash_profile files from docker directory
COPY docker/bash_profile.shared /tmp/
COPY docker/bash_profile.robot /tmp/
COPY docker/bash_profile.desktop /tmp/
# Copy setup script
COPY scripts/setup_bash_profile.sh /tmp/
RUN chmod +x /tmp/setup_bash_profile.sh
# Combine shared and platform-specific bash_profile based on architecture
RUN ARCH=$(dpkg --print-architecture) && \
    /tmp/setup_bash_profile.sh ${ARCH} ${USERNAME} ${ROS_DISTRO} ${GID} && \
    rm -f /tmp/bash_profile.shared /tmp/bash_profile.robot /tmp/bash_profile.desktop /tmp/setup_bash_profile.sh

RUN mkdir -p /opt/spot_audio_models && \
    chown -R ${UID}:${GID} /opt/spot_audio_models

# Create .tmux.conf
RUN echo "set -g mouse on" >> /home/${USERNAME}/.tmux.conf && \
    echo "set -g history-limit 10000" >> /home/${USERNAME}/.tmux.conf && \
    chown ${USERNAME}:${GID} /home/${USERNAME}/.tmux.conf

# =============================================================================
# Switch to User Context
# =============================================================================

USER $username
ENV HOME=/home/$username
WORKDIR /home/$username

# Install user level pip dependencies
# Used for av_imitation
# Need this very specific verison of numpy for pytorch and cv_bridge compatibility
#   - PyTorch: 1.26.4 provides the 0x10 API version it requires.
#   - ROS 2: NumPy 1.26.4 is backward compatible with modules compiled against 1.21, so cv_bridge will continue to work.
RUN --mount=type=cache,target=/tmp/pip-cache,mode=0777 PIP_CACHE_DIR=/tmp/pip-cache pip3 install \
    "numpy==1.26.4" \
    onnx \
    onnxruntime \
    flask \
    tensorboard \
    pygame \
    soxr==0.5.0 \
    faster-whisper==1.1.0 \
    ctranslate2==4.5.0

# Pre-download Whisper + Silero VAD models (no runtime network dependency)
# Note: torch.hub.load for silero-vad requires torchaudio (not available on Jetson),
# so we download the repo zip and copy the JIT file directly.
# RUN mkdir -p /opt/spot_audio_models && \
#     python3 -c "from faster_whisper import WhisperModel; WhisperModel('base', download_root='/opt/spot_audio_models/whisper-base', device='cpu', compute_type='int8')" && \
#     python3 -c "import urllib.request, zipfile, shutil, os; urllib.request.urlretrieve('https://github.com/snakers4/silero-vad/archive/refs/tags/v5.1.zip', '/tmp/silero_vad.zip'); z = zipfile.ZipFile('/tmp/silero_vad.zip'); jit = [f for f in z.namelist() if f.endswith('silero_vad.jit')][0]; z.extract(jit, '/tmp/silero_extract'); shutil.copy2(os.path.join('/tmp/silero_extract', jit), '/opt/spot_audio_models/silero_vad.jit'); z.close()" && \
#     ls -la /opt/spot_audio_models/whisper-base/ && \
#     test -f /opt/spot_audio_models/silero_vad.jit && \
#     echo "Model provisioning OK"

# =============================================================================
# Setup Unicast
# =============================================================================
COPY scripts/setup_unicast.sh /home/${username}/setup_unicast.sh
RUN /home/${username}/setup_unicast.sh && rm /home/${username}/setup_unicast.sh
