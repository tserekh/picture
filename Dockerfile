FROM heroku/miniconda

RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3 -y
RUN apt install curl -y
RUN apt-get install python3-pip -y


#RUN apt install nginx -y
#RUN service nginx start
#
#RUN openssl req -batch -x509 -nodes -days 365 -newkey rsa:2048 -keyout /etc/nginx/cert.key -out /etc/nginx/cert.crt

COPY . ./app
WORKDIR /app

#RUN service nginx restart
RUN pip3 install -r /app/requirements.txt
CMD gunicorn --bind 0.0.0.0:$PORT wsgi
