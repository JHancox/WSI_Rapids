FROM rapidsai/rapidsai-core:21.08-cuda11.2-runtime-ubuntu20.04-py3.8
RUN conda update -n base -c defaults conda \
  && source activate rapids \
  && conda install -c conda-forge dask-image \
  && apt-get update \
  && apt-get -y install gcc \
  && pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html \
  && pip install napari-lazy-openslide \
  && pip install cucim \
  && pip install termcolor \
  && conda install -c bioconda openslide
