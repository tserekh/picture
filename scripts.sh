sudo docker build -t image2 .
sudo docker run -i -t -d -p 5000:5000 image2
sudo docker run -i -t -d -P --network="host" image2