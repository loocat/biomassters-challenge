ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.7.1
FROM ${BASE_IMAGE}

## Base packages for ubuntu
RUN apt-get clean && \
    apt-get update -qq && \
    apt-get install -y \
        sudo \
        gosu \
        git \
        wget \
        bzip2 \
        htop \
        nano \
        g++ \
        gcc \
        make \
        build-essential \
        software-properties-common \
        apt-transport-https \
        libhdf5-dev \
        libgl1-mesa-glx \
        openmpi-bin \
        graphviz \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# ## Download and install miniforge
# RUN wget -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
# RUN /bin/bash /tmp/miniforge.sh -bf -p /opt/conda && \
#     rm /tmp/miniforge.sh && \
#     echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh
# ENV PATH /opt/conda/bin:$PATH

# download and install mambaforge
# RUN wget -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh
RUN wget -O /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-pypy3-Linux-aarch64.sh
RUN /bin/bash /tmp/miniforge.sh -bf -p /opt/conda && \
    rm /tmp/miniforge.sh && \
    echo "export PATH=/opt/conda/bin:$PATH" > /etc/profile.d/conda.sh

#
# Install python and upgrade pip version
# RUN conda install -c conda-forge -y 'python>=3.6,<3.7' && \
#     pip install --upgrade pip
ENV PATH=/opt/conda/bin:$PATH
RUN conda create --no-default-packages -n ml python=3.6.9
SHELL ["conda", "run", "-n", "ml", "/bin/bash", "-c"]
RUN conda init --system bash
RUN echo `python --version` && \
    echo `python3 --version`
# RUN conda install -y -n ml \
#         -c conda-canary -c defaults -c conda-forge \
#         rasterio && \
#     conda clean --all --yes

#
# install prerequisites (many of these are for numpy)
#
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        python3-dev \
        python3-numpy \
        libopenblas-dev \
        libopenmpi-dev \
        openmpi-bin \
        openmpi-common \
        gfortran \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN pip3 install --no-cache-dir setuptools Cython wheel
RUN pip3 install --no-cache-dir --verbose numpy

#
# Install pytorch 1.10.0
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
#
ARG PYTORCH_URL=https://nvidia.box.com/shared/static/fjtbno0vpo676a25cgvuqc1wty0fkkg6.whl
ARG PYTORCH_WHL=torch-1.10.0-cp36-cp36m-linux_aarch64.whl

# ARG PYTORCH_URL=https://nvidia.box.com/shared/static/3ibazbiwtkl181n95n9em3wtrca7tdzp.whl
# ARG PYTORCH_WHL=torch-1.5.0-cp36-cp36m-linux_aarch64.whl

# ARG PYTORCH_URL=https://nvidia.box.com/shared/static/lufbgr3xu2uha40cs9ryq1zn4kxsnogl.whl
# ARG PYTORCH_WHL=torch-1.2.0-cp36-cp36m-linux_aarch64.whl

RUN wget --quiet --show-progress --progress=bar:force:noscroll --no-check-certificate ${PYTORCH_URL} -O ${PYTORCH_WHL} && \
    pip3 install --no-cache-dir --verbose ${PYTORCH_WHL} && \
    rm ${PYTORCH_WHL}

#
# Install torchvision 0.11.1
#
ARG TORCHVISION_VERSION=v0.11.1
# ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2;8.7;10.2"
ARG TORCH_CUDA_ARCH_LIST="5.3;6.2;7.2"

RUN printenv && \
    echo "torchvision version = $TORCHVISION_VERSION" && \
    echo "TORCH_CUDA_ARCH_LIST = $TORCH_CUDA_ARCH_LIST"

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git \
        build-essential \
        libjpeg-dev \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV OPENBLAS_CORETYPE=ARMV8
RUN git clone https://github.com/pytorch/vision torchvision && \
    cd torchvision && \
    git checkout ${TORCHVISION_VERSION} && \
    OPENBLAS_CORETYPE=ARMV8 python3 setup.py install && \
    cd ../ && \
    rm -rf torchvision

# prevent python version conflict: future annotation
RUN pip3 install --no-cache-dir --verbose 'pillow<8'


#
# PyCUDA
#
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN echo "$PATH" && echo "$LD_LIBRARY_PATH"

RUN pip3 install --no-cache-dir --verbose pycuda six


# 
# install OpenCV (with CUDA)
#
ARG OPENCV_URL=https://nvidia.box.com/shared/static/5v89u6g5rb62fpz4lh0rz531ajo2t5ef.gz
ARG OPENCV_DEB=OpenCV-4.5.0-aarch64.tar.gz

COPY scripts/opencv_install.sh /tmp/opencv_install.sh
RUN cd /tmp && ./opencv_install.sh ${OPENCV_URL} ${OPENCV_DEB}


# # #
# # # tensorflow 2.7.0 for python3
# # #
# # ENV PYTHON3_VERSION=3.6
# # COPY --from=tensorflow \
# #     /usr/local/lib/python${PYTHON3_VERSION}/dist-packages/ \
# #     /usr/local/lib/python${PYTHON3_VERSION}/dist-packages/
RUN apt update && \
    apt install -y \
        libhdf5-serial-dev \
        hdf5-tools \
        libhdf5-dev \
        zlib1g-dev \
        zip \
        libjpeg8-dev \
        liblapack-dev \
        libblas-dev \
        gfortran && \
    rm -rf /var/lib/apt/lists/* \
    apt-get clean

#
# python pip packages
#
# # RUN pip3 install --no-cache-dir --verbose testresources
# RUN pip3 install --no-cache-dir --verbose 'h5py<=3.1.0'
# # RUN pip3 install --no-cache-dir --verbose keras_preprocessing==1.1.2 keras_applications==1.0.8 gast==0.4.0
# RUN pip3 install --no-cache-dir --verbose \
#     keras_preprocessing keras_applications gast \
#     opt_einsum astunparse flatbuffers
# # RUN pip3 install --no-cache-dir --ignore-installed pybind11 
# # RUN pip3 install --no-cache-dir --verbose onnx
# # RUN pip3 install --no-cache-dir --verbose scipy
# # RUN pip3 install --no-cache-dir --verbose scikit-learn
# # RUN pip3 install --no-cache-dir --verbose pandas
# # RUN pip3 install --no-cache-dir --verbose pycuda
# # RUN pip3 install --no-cache-dir --verbose numba
RUN pip3 install --no-cache-dir --verbose \
    --no-deps \
    --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v461 \
    tensorflow==2.7.0

# #
# # Setup order is important (GDAL, Rasterio, OpenCV)! - otherwise it won't import due to dependency conflict
# RUN add-apt-repository ppa:ubuntugis/ppa \
# && apt-get update \
# && apt-get install -y python-numpy gdal-bin libgdal-dev \
# && pip install \
#    "opencv-python>=3.4.1.5,<3.5" \
#    "rasterio>=1.0b2,<1.3"
RUN apt update && \
    apt install -y gdal-bin libgdal-dev && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean

# RUN pip3 install --no-cache-dir --verbose rasterio==1.8

# ### Build libspatialindex from source because conda's libspatialindex conflicts with GDAL
# #RUN wget http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz -O /tmp/spatialindex-src.tar.gz && \
# #    tar -xvf /tmp/spatialindex-src.tar.gz -C /tmp
# #WORKDIR /tmp/spatialindex-src-1.8.5
# #RUN ./configure && make && make install && ldconfig && pip install Rtree==0.8.3

ENV USERNAME=user
ENV USER_UID=1000
ENV USER_GID=$USER_UID

RUN echo "$USERNAME $USER_UID $USER_GID"

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# ********************************************************
# * Anything else you want to do like clean up goes here *
# ********************************************************

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

RUN echo 'export PATH=$HOME/.local/bin:$PATH' >> ~/.bashrc && \
    echo 'export LC_ALL=C.UTF-8' >> ~/.bashrc && \
    echo 'export LANG=C.UTF-8' >> ~/.bashrc

# grant $USERNAME to access cuda
RUN sudo usermod -aG video $USERNAME && newgrp

# activate conda environment: ml
RUN conda init bash \
    && echo 'export OPENBLAS_CORETYPE=ARMV8' >> ~/.bashrc \
    && echo 'conda activate ml' >> ~/.bashrc
