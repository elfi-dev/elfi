FROM ubuntu:bionic

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz make python3-scipy jupyter python3-matplotlib
RUN pip3 install graphviz

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

WORKDIR /elfi0
ADD . /elfi0

RUN pip3 install -e .
RUN pip3 install -r requirements-dev.txt

# Note: The created image contains a static version of ELFI. To use the live, up-to-date version
# you should mount the elfi directory when running the container ('make docker' does it for you).
