# Use a base image with the desired Python version and CUDA support
FROM pytorch/pytorch:1.9.0-cuda11.1-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    build-essential



# Set the working directory
RUN mkdir -p /home/damian/dev/thesis/lightweight-human-pose-estimation-3d-demo.pytorch
WORKDIR /home/damian/dev/thesis/lightweight-human-pose-estimation-3d-demo.pytorch

# Copy the repository
COPY . .

# Install Python dependencies
RUN pip install -r requirements.txt

# Download the pre-trained model
RUN wget -O models/human-pose-estimation-3d.pth "https://drive.google.com/uc?export=download&id=1niBUbUecPhKt3GyeDNukobL4OQ3jqssH"


# Copy gmake and cc binaries
#COPY /usr/bin/gmake /usr/bin/gmake
#COPY /usr/bin/cc /usr/bin/cc

# Set environment variables for the C compiler
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
# Build pose_extractor module
RUN python setup.py build_ext

# Add build folder to PYTHONPATH
ENV PYTHONPATH="/home/damian/dev/thesis/social-robotics-gaze-detection/pose_extractor/build/:$PYTHONPATH"

# Set OpenVINO environment variables
#ENV OpenVINO_INSTALL_DIR="/opt/intel/openvino"
#ENV LD_LIBRARY_PATH="${OpenVINO_INSTALL_DIR}/deployment_tools/inference_engine/external/tbb/lib:${LD_LIBRARY_PATH}"

# Convert checkpoint to ONNX with OpenVINO
#RUN source ${OpenVINO_INSTALL_DIR}/bin/setupvars.sh && \
#    python scripts/convert_to_onnx.py --checkpoint-path models/human-pose-estimation-3d.pth

# Convert ONNX to OpenVINO format
#RUN python ${OpenVINO_INSTALL_DIR}/deployment_tools/model_optimizer/mo.py --input_model models/human-pose-estimation-3d.onnx --input=data --mean_values=data[128.0,128.0,128.0] --scale_values=data[255.0,255.0,255.0] --output=features,heatmaps,pafs

# Install CUDA and cuDNN for TensorRT
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-compiler-11-1 \
    cuda-libraries-11-1 \
    libcudnn8=8.2.4.15-1+cuda11.1 \
    libcudnn8-dev=8.2.4.15-1+cuda11.1

# Install torch2trt for TensorRT
RUN python -m pip install nvidia-pyindex && \
    pip install nvidia-tensorrt==7.2.1.6 && \
    pip install torch2trt

# Convert checkpoint to TensorRT format
RUN python scripts/convert_to_trt.py --checkpoint-path models/human-pose-estimation-3d.pth

# Set the entrypoint command
CMD ["python", "demo.py", "--model", "models/human-pose-estimation-3d.pth", "--video", "0"]
