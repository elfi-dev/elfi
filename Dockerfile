FROM ubuntu:bionic

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y graphviz make python3-scipy jupyter python3-matplotlib
RUN pip3 install graphviz

WORKDIR /elfi
ADD . /elfi

RUN pip3 install -e .
RUN pip3 install -r requirements-dev.txt
