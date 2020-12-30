FROM ubuntu:20.10
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3 -y
RUN apt install curl -y
RUN apt-get install python3-pip -y

RUN apt install nginx -y
RUN service nginx start

RUN openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/cert.key -out /etc/nginx/cert.crt

COPY . ./app
COPY devserver.conf /etc/nginx/conf.d/devserver.conf
RUN service nginx restart
RUN pip install -r app/requirements.txt
EXPOSE 80
CMD ["python3", "app/main.py"]