# https://hub.docker.com/r/continuumio/miniconda3
FROM continuumio/miniconda3
# https://hub.docker.com/r/frolvlad/alpine-miniconda3
# FROM frolvlad/alpine-miniconda3

RUN conda install -y -c conda-forge bash jupyter jupyter_contrib_nbextensions
RUN conda install -y -c conda-forge xeus-cling xtensor
RUN mkdir /notebooks