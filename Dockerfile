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
#RUN chmod -R 777 /opt/conda/envs/env38/lib/python3.8/site-packages

#RUN cp /app/models/research/object_detection /opt/conda/envs/env38/lib/python3.8/site-packages/object_detection -r

RUN useradd -m myuser
#RUN chmod -R a+rwX /opt/conda/lib/python3.8/
USER myuser
WORKDIR /app
#CMD gunicorn --bind 0.0.0.0:$PORT wsgi


ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env38", "gunicorn", "--bind", "0.0.0.0:$PORT", "wsgi2"]
#CMD gunicorn --bind 0.0.0.0:$PORT wsgi2
#CMD python main.py