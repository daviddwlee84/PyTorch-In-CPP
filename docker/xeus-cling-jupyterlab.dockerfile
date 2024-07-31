# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive

# Install basic tools and dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    cmake \
    # vim \
    build-essential \
    libzmq3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
RUN wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O Miniforge3-Linux-x86_64.sh \
    && bash Miniforge3-Linux-x86_64.sh -b -p /opt/conda \
    && rm Miniforge3-Linux-x86_64.sh

# Update conda and install mamba
ENV PATH /opt/conda/bin:$PATH
RUN conda update -y -n base -c defaults conda \
    && conda install -y mamba -c conda-forge

# Install xeus-cling and JupyterLab
RUN mamba install -y -c conda-forge xeus-cling jupyterlab

# Install library
# https://anaconda.org/conda-forge/mkl
# https://anaconda.org/conda-forge/libtorch
# https://anaconda.org/conda-forge/xtensor-blas
# https://anaconda.org/conda-forge/nlohmann_json/ (not working)
# https://json.nlohmann.me/integration/package_managers/#conda
# https://github.com/conda-forge/nlohmann_json-feedstock
RUN mamba install -y -c conda-forge xtensor xtensor-blas nlohmann_json=3.11.2 xtl mkl libtorch
# This may update libtorch and downgrade mkl
RUN mamba install -y pytorch torchvision torchaudio cpuonly -c pytorch
# RUN mamba install -c intel mkl mkl-devel mkl-static mkl-include
# RUN mamba install mkl mkl-devel mkl-static mkl-include
# RUN mamba install -c pytorch mkl mkl-devel mkl-static mkl-include
RUN mamba install -c pytorch mkl=2023.2.0 mkl-include=2023.2.0

# Optionally, expose the JupyterLab port
EXPOSE 8888

ENV CONDA_PREFIX /opt/conda

WORKDIR /workspace

# Set the default command to start JupyterLab
CMD ["jupyter-lab", "--ip=0.0.0.0", "--no-browser", "--allow-root"]
