FROM josifovski/vtprl:latest as base 

WORKDIR /home 

RUN apt-get update && apt-get  install -y python3-tk

ADD docker/requirements.txt /home/requirements.txt
RUN python3 -m pip install -r /home/requirements.txt


ADD docker/start.sh ./start.sh
