FROM ubuntu:trusty

WORKDIR /snorkel
SHELL ["/bin/bash", "-c"]
ARG DEBIAN_FRONTEND=noninteractive

COPY install/cleanup.sh ./
COPY install/apt_pkgs.sh ./
RUN ./apt_pkgs.sh \
 && ./cleanup.sh

COPY install/corenlp.sh ./
RUN ./corenlp.sh \
 && ./cleanup.sh

COPY install/conda.sh ./
RUN ./conda.sh \
 && ./cleanup.sh

COPY install/miniconda_env.sh ./
COPY install/python_pkgs.sh ./
COPY python-package-requirement.txt ./
RUN ./python_pkgs.sh \
 && ./cleanup.sh

COPY . .
RUN git submodule update --init --recursive && ls
ENTRYPOINT /snorkel/docker/entry.sh
