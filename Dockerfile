FROM python:2

WORKDIR /snorkel
SHELL ["/bin/bash", "-c"]

COPY python-package-requirement.txt ./
COPY docker/install.sh ./docker/install.sh
RUN source docker/install.sh &&\
 install_python_pkgs &&\
 cleanup

COPY install-parser.sh ./
RUN source docker/install.sh &&\
 install_corenlp &&\
 cleanup

COPY . .
RUN git submodule update --init --recursive
ENTRYPOINT /snorkel/docker/entry.sh
