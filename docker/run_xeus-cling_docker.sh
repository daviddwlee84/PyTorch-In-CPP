#!/bin/bash
# https://stackoverflow.com/questions/17466699/how-do-i-build-a-dockerfile-if-the-name-of-the-dockerfile-isnt-dockerfile
# docker build -t xeus-cling -f xeus-cling.dockerfile .
# docker run --rm -p 8888:8888 --workdir /notebooks -it xeus-cling jupyter notebook --allow-root --ip 0.0.0.0

docker build -t xeus-cling-jupyterlab -f xeus-cling-jupyterlab.dockerfile .
docker run --rm -p 8888:8888 -v `pwd`/..:/workspace -it xeus-cling-jupyterlab
