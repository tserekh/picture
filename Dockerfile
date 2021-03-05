FROM continuumio/anaconda3
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
#RUN pip3 install object_detection
RUN pip3 install -r /app/requirements.txt
#RUN rm /opt/conda/lib/python3.8/site-packages/object_detection/protos -r
RUN cp /app/models/research/object_detection /opt/conda/lib/python3.8/site-packages/object_detection -r
WORKDIR /opt
RUN useradd -m myuser
RUN chmod -R a+rwX /opt
USER myuser
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
#CMD python main.py