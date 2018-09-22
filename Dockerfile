FROM ubuntu:18.04

RUN apt-get update && \ 
    apt-get install -y python3 python3-pip && \
    apt-get install -y make build-essential git && \
    mkdir -p /home/hervaldeepmailing && \
    mkdir -p /home/hervaldeepmailing/logs && \
    mkdir -p /home/hervaldeepmailing/output.source

ENV LANG en_US.utf8

COPY requirements.txt /home/hervaldeepmailing
COPY start_docker.sh /home/hervaldeepmailing
COPY output.source /home/hervaldeepmailing/output.source

WORKDIR /home/hervaldeepmailing

RUN git clone --recursive https://github.com/dmlc/xgboost.git

WORKDIR /home/hervaldeepmailing/xgboost

RUN make -j4

WORKDIR /home/hervaldeepmailing

RUN pip3 install -r /home/hervaldeepmailing/requirements.txt

VOLUME /home/hervaldeepmailing/data

CMD ./start_docker.sh

