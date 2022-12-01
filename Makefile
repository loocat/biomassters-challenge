UID := $(shell id -u)
GID := $(shell id -g)

inference: stage1_inference

train-inference: stage1

build:
	docker build -t biomassters:0.1 --build-arg "USER_UID=$(UID)" .

start:
	docker-compose up -d

install:
	docker exec biomassters-dev pip install -r requirements.txt --user 

clean:
	docker-compose down

stage1: stage1_train stage1_inference
