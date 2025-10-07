# hash:sha256:bf148e88e61f323d18318b088218e45f06392959a83cc8fcb0951fb2e78a8fb9
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN pip install -U --no-cache-dir \
    geopandas==1.0.1 \
    geopy==2.4.1 \
    numpy==2.0.2 \
    pandas==2.2.3 \
    scipy==1.13.1
