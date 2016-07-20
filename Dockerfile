# Dockerfile for HazyResearch/Snorkel
#
# (Optional) You can create a fresh image with:
# $ BRANCH=snorkel-containers
# $ docker build -t hazyresearch/snorkel https://github.com/HazyResearch/snorkel/raw/$BRANCH/Dockerfile --build-arg BRANCH=$BRANCH
#
# Run the Snorkel docker container with:
# $ cd /path/to/your/notebooks
# $ docker run -it -p12345:8888 --volume $PWD:/work --rm hazyresearch/snorkel
#
# Finally, point your browser to http://localhost:12345

FROM jupyter/datascience-notebook
MAINTAINER deepdive-dev@googlegroups.com

# install dependencies of Snorkel's dependencies
USER root
RUN apt-get update && apt-get install -qy \
        python-dev libxml2-dev libxslt1-dev zlib1g-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
USER jovyan

# grab a shallow clone of Snorkel
ARG BRANCH=master
ENV BRANCH=$BRANCH
RUN git clone https://github.com/HazyResearch/snorkel.git \
        --branch $BRANCH \
        --depth 10 \
 && cd snorkel \
 && git submodule update --init --recursive

# set up Snorkel
WORKDIR snorkel
RUN pip2 install --requirement requirements.txt \
 && pip3 install --requirement requirements.txt
RUN ./install-parser.sh
WORKDIR ..

ENV SNORKELHOME=/home/jovyan/work/snorkel

RUN ln -sfn /work . \
 && mkdir -p \
    'snorkel IS NOT PERSISTENT!!! YOUR CHANGES WILL DISAPPEAR!!!'/'PLEASE GO BACK' \
    'work is the right place to keep your files with the --volume flag'/'PLEASE GO BACK' \
 && chmod -R a-w 'snorkel '* 'work '* \
 && chmod a-w .
