FROM python:3.6

RUN apt-get update && apt-get install -y graphviz

WORKDIR /elfi
ADD . /elfi

RUN pip install numpy graphviz
RUN pip install -e .

# matplotlib unable to show figures
RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" >> /root/.config/matplotlib/matplotlibrc

CMD ["/bin/bash"]
