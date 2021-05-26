FROM nvcr.io/nvidia/rapidsai/rapidsai:0.19-cuda11.0-runtime-ubuntu18.04
#FROM wsi_rapids:v2
#RUN source activate rapids \
#  && conda update -n base -c defaults conda \
#  && conda install pytorch==1.7.1 \
#  && conda install -c conda-forge dask-image 
RUN source activate rapids \
  && conda update -n base -c defaults conda \
  && conda install pytorch==1.7.1 \
  && conda install -c conda-forge dask-image \
  && apt-get update \
  && apt-get -y install gcc \
  && pip install napari-lazy-openslide \
  && pip install cucim \
  && conda install -c bioconda openslide
  
#  && conda list \ 
#ENV VIRTUAL_ENV /opt/conda/envs/rapids
#RUN python3 -m rapids --python=/usr/bin/python3 $VIRTUAL_ENV
#ENV PATH "$VIRTUAL_ENV/bin:$PATH"
#SHELL ["conda", "run", "-n", "rapids", "conda", "list"]
#RUN conda env list
#RUN echo "source activate rapids" >> ~/.bashrc && conda list
#RUN conda init bash && . ~/.bashrc  && conda activate rapids  && conda install pytorch==1.7.1 -c #pytorch
#  
#  && apt-get update \
#  && apt-get install gcc \
#  && apt-get install openslide \
#  && pip install napari-lazy-openslide
#RUN /bin/bash -c "conda activate rapids" && conda install pytorch==1.7.1 -c pytorch && conda install #-c conda-forge dask-image
#RUN apt-get update && apt-get install gcc && apt-get install openslide
#RUN pip install napari-lazy-openslide

