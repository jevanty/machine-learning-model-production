heroku login
sudo docker login

sudo heroku container:login

sudo docker build -t registry.heroku.com/titanictamere/web .
sudo docker push registry.heroku.com/titanictamere/web:latest
sudo heroku container:release web -a titanictamere