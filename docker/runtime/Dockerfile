ARG BASE_IMAGE=ubuntu:20.04
ARG DEV_IMAGE=parflow-development

FROM ${DEV_IMAGE} AS devimage
FROM ${BASE_IMAGE}


# Non interactive mode
env DEBIAN_FRONTEND noninteractive

# Fetch the latest definitions of packages
run apt-get update && \
  ln -fs /usr/share/zoneinfo/America/New_York /etc/localtime && \
  apt-get install -y tzdata && \
  dpkg-reconfigure --frontend noninteractive tzdata && \
  apt-get install -y \
    curl \
    libcurl4 \
    git \
    vim \
    libopenblas-dev \
    liblapack-dev \
    openssh-client \
    openssh-server \
    openmpi-bin \
    libopenmpi-dev \
    python3 \
    python3-pip \
    python3-venv \
    tcl-dev \
    tk-dev

RUN groupadd -g 1000 ubuntu && \
  useradd -u 1000 -g ubuntu -m -N -s /bin/bash ubuntu

USER ubuntu
WORKDIR /home/ubuntu/

COPY --from=devimage /opt/parflow /opt/parflow
COPY --from=devimage /opt/hypre   /opt/hypre
COPY --from=devimage /opt/netcdf  /opt/netcdf
COPY --from=devimage /opt/hdf5    /opt/hdf5
COPY --from=devimage /opt/silo    /opt/silo

RUN pip3 install -r /opt/parflow/python/requirements_all.txt

ENV HYPRE_DIR /opt/hypre
ENV PARFLOW_DIR /opt/parflow
ENV PYTHONPATH /opt/parflow/python

ENTRYPOINT ["python3"]
