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

# Install Java 8 for CoreNLP
# See: http://www.webupd8.org/2014/03/how-to-install-oracle-java-8-in-debian.html
USER root
RUN echo "deb http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee /etc/apt/sources.list.d/webupd8team-java.list \
 && echo "deb-src http://ppa.launchpad.net/webupd8team/java/ubuntu xenial main" | tee -a /etc/apt/sources.list.d/webupd8team-java.list \
 && apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys EEA14886 \
 && echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | sudo /usr/bin/debconf-set-selections \
 && apt-get update && apt-get install -qy \
        oracle-java8-installer \
        oracle-java8-set-default \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
USER jovyan

# Install dependencies of Snorkel's dependencies
# (lxml, matplotlib)
USER root
RUN apt-get update && apt-get install -qy \
        python-dev libxml2-dev libxslt1-dev zlib1g-dev \
        pkg-config libfreetype6-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*
USER jovyan

# Install python2
RUN conda install --yes python=2.7

# Install binary distribution of scientific python modules
ARG NUMPY=1.11
ARG SCIPY=0.17
RUN conda install --yes numpy=$NUMPY scipy=$SCIPY matplotlib pip

# Install theano
RUN conda install --yes theano

# Grab a shallow clone of Snorkel
ARG BRANCH=master
ENV BRANCH=$BRANCH
RUN git clone https://github.com/HazyResearch/snorkel.git \
        --branch $BRANCH \
        --depth 10 \
 && cd snorkel \
 && git submodule update --init --recursive

# Set up Snorkel: installing CoreNLP, Mindbender (part of DeepDive), etc.
WORKDIR snorkel
RUN ./install-parser.sh \
 && rm -f stanford-corenlp-*.zip
RUN bash -euc 'PREFIX="$PWD"/deepdive bash <(curl -fsSL git.io/getdeepdive || wget -qO- git.io/getdeepdive) deepdive_from_release' \
 && rm -f deepdive-*.tar.gz
RUN pip2 install --requirement requirements.txt \
 && pip3 install --requirement requirements.txt
WORKDIR ..

ENV SNORKELHOME=/home/jovyan/work/snorkel

RUN ln -sfn /work . \
 && mkdir -p \
    'snorkel IS NOT PERSISTENT!!! YOUR CHANGES WILL DISAPPEAR!!!'/'PLEASE GO BACK' \
    'work is the right place to keep your files with the --volume flag'/'PLEASE GO BACK' \
 && chmod -R a-w 'snorkel '* 'work '* \
 && chmod a-w .
