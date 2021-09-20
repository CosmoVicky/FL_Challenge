FROM fedora:34
LABEL maintainer="khoviktoriya@yandex.ru"
RUN dnf install python3-pip -y
RUN pip install tensorflow tensorflow_datasets numpy matplotlib
COPY ./cifar.py .
COPY ./main.py .
COPY ./run.sh .
COPY ./test.py .
CMD bash ./run.sh && echo OK || echo Failed
