FROM registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow:0.12.0

RUN 	mkdir -p /tf/mnist
WORKDIR	/tf/mnist
COPY	convert_to_records.py /tf/mnist
COPY	mnist_replica.py /tf/mnist
RUN	python /tf/mnist/convert_to_records.py

ENTRYPOINT ["/bin/bash"]