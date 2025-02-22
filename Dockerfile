FROM python:3.8
COPY . ./app
WORKDIR /app
RUN apt-get update
RUN apt-get install  libgl1-mesa-glx -y
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade setuptools
RUN git clone --depth 1 https://github.com/tensorflow/models
WORKDIR /app/models/research/
RUN chmod -R 777 ./
RUN apt install -y protobuf-compiler
RUN protoc object_detection/protos/*.proto --python_out=.
RUN pip3 install -r /app/requirements.txt
RUN cp /app/models/research/object_detection /usr/local/lib/python3.8/site-packages/object_detection -r
WORKDIR /app
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
