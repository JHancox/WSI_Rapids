FROM nvcr.io/nvidia/rapidsai/rapidsai:0.19-cuda11.0-runtime-ubuntu18.04

RUN source activate rapids \
  && conda update -n base -c defaults conda \
  && conda install pytorch==1.7.1 \
  && conda install -c conda-forge dask-image \
  && apt-get update \
  && apt-get -y install gcc \
  && pip install napari-lazy-openslide \
  && pip install cucim \
  && pip install termcolor \
  && conda install -c bioconda openslide
