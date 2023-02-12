FROM nvcr.io/nvidia/pytorch:21.03-py3
LABEL maintainer="Hugging Face"

ARG DEBIAN_FRONTEND=noninteractive

# Example: `cu102`, `cu113`, etc.
ARG CUDA='cu113'

RUN apt -y update
RUN apt install -y libaio-dev
RUN python3 -m pip install --no-cache-dir --upgrade pip

ARG REF=main
RUN git clone https://github.com/huggingface/transformers && cd transformers && git checkout $REF

# Install **nightly** release PyTorch (flag `--pre`)
# (PyTorch must be installed before pre-compiling any DeepSpeed c++/cuda ops.)
# (https://www.deepspeed.ai/tutorials/advanced-install/#pre-install-deepspeed-ops)
RUN python3 -m pip install --no-cache-dir -U --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/$CUDA

RUN python3 -m pip install --no-cache-dir ./transformers[deepspeed-testing]

# Pre-build **nightly** release of DeepSpeed, so it would be ready for testing (otherwise, the 1st deepspeed test will timeout)
RUN python3 -m pip uninstall -y deepspeed
# This has to be run inside the GPU VMs running the tests. (So far, it fails here due to GPU checks during compilation.)
# Issue: https://github.com/microsoft/DeepSpeed/issues/2010
# RUN git clone https://github.com/microsoft/DeepSpeed && cd DeepSpeed && rm -rf build && \
#    DS_BUILD_CPU_ADAM=1 DS_BUILD_AIO=1 DS_BUILD_UTILS=1 python3 -m pip install . --global-option="build_ext" --global-option="-j8" --no-cache -v --disable-pip-version-check 2>&1

# When installing in editable mode, `transformers` is not recognized as a package.
# this line must be added in order for python to be aware of transformers.
RUN cd transformers && python3 setup.py develop

# Disable for now as deepspeed is not installed above. To be enabled once the issue is fixed.
# RUN python3 -c "from deepspeed.launcher.runner import main"
