FROM ubuntu:bionic

RUN apt-get update && apt-get install -y graphviz make python3-pip
RUN pip3 install numpy graphviz

WORKDIR /elfi
ADD . /elfi

RUN pip3 install -e .
RUN pip3 install -r requirements-dev.txt

# default matplotlib to Agg backend
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" >> /root/.config/matplotlib/matplotlibrc
