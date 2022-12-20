UID := $(shell id -u)
GID := $(shell id -g)

include $(CURDIR)/common.mk

DOCKER_COMPOSE := ~/.docker/cli-plugins/docker-compose

infer: stage1_infer

train-infer: stage1

_device_:
	cp -r /usr/local/cuda/samples ./
	docker build $(DOCKER_BINFMT_MISC) -t biomassters:device -f ./Dockerfile.device .
	docker run -it --rm --runtime nvidia biomassters:device
	docker rmi  `docker images | grep device | head -n1 | awk '{print $$3;}'`
	rm -rf ./samples

device:
	docker exec -t biomassters-dev bash scripts/device.sh

ml:
	docker run -it --rm --runtime nvidia --name l4t-ml nvcr.io/nvidia/l4t-ml:r32.7.1-py3

bash:
	docker exec -it biomassters-dev bash

load_tiff:
	docker exec -it biomassters-dev conda run -n ml python src/load_test.py --dir data/raw

printenv:
	docker exec -t biomassters-dev "bash -c echo $$$$PATH"

build:
	docker build $(DOCKER_BINFMT_MISC) -t biomassters:0.1 --build-arg "USER_UID=$(UID)" .

start:
	$(DOCKER_COMPOSE) up -d

install:
	docker exec biomassters-dev conda run -n ml pip3 install --no-cache-dir --verbose -r requirements.txt --user 

clean:
	$(DOCKER_COMPOSE) down

stage1: stage1_ready stage1_train stage1_infer

stage1_ready:
	docker exec -t biomassters-dev conda run -n ml bash scripts/stage1_ready.sh

stage1_train:
	docker exec -t biomassters-dev conda run -n ml bash scripts/stage1_train.sh

stage1_infer:
	docker exec -t biomassters-dev conda run -n ml bash scripts/stage1_infer.sh