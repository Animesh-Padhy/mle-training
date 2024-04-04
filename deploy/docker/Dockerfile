FROM ubuntu

RUN apt-get update && \
    apt-get install -y wget && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH /opt/conda/bin:$PATH

RUN conda init && \
    conda update -n base -c defaults conda -y

COPY . /house-price-prediction
WORKDIR /house-price-prediction

RUN conda install -y \
    six \
    urllib3 \
    pandas \
    joblib \
    scikit-learn \
    numpy \
    scipy \
    matplotlib

COPY ./dist /house-price-prediction

RUN tar -xzvf house-price-prediction-0.1.tar.gz && \
    rm house-price-prediction-0.1.tar.gz && \
    pip install house_price_prediction-0.1-py3-none-any.whl

CMD python src/main.py