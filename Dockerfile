FROM registry.cn-hangzhou.aliyuncs.com/denverdino/tensorflow:0.12.0

RUN 	mkdir -p /tf/mnist;
WORKDIR	/tf/mnist
COPY	mnist_replica.py .

ENTRYPOINT ["python"]
