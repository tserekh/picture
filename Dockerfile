FROM ubuntu:20.10
RUN apt update
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa -y
RUN apt install python3 -y
RUN apt install curl -y
RUN apt-get install python3-pip -y
COPY . ./app
RUN pip install -r app/requirements.txt
EXPOSE 5000
CMD ["python3", "app/main.py"]