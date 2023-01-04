DIR := $(shell pwd)

DOCKERFILE := Dockerfile
DOCKER_REPOSITORY := shibui/recommendation_study

############ COMMON COMMANDS ############
SRC := $(DIR)/recommendation/src/

.PHONY: lint
lint:
	black --check --diff --line-length 120 $(SRC)

.PHONY: sort
sort:
	isort $(SRC)

.PHONY: fmt
fmt: sort
	black --line-length 120 $(SRC)

.PHONY: vet
vet:
	mypy $(SRC)

.PHONY: install_prettier
install_prettier:
	npm install

.PHONY: format_md
format_md: install_prettier
	npx prettier --write .

.PHONY: req
req:
	poetry export \
		--without-hashes \
		-f requirements.txt \
		--output requirements.txt

.PHONY: install_dep
install_dep:
	pip install -r requirements.txt


############ RECOMMENDATIONS COMMANDS ############
RECOMMENDATION_DIR := $(DIR)/recommendation
RECOMMENDATION_VERSION := 0.0.0
DOCKERFILE_RECOMMENDATION = $(RECOMMENDATION_DIR)/$(DOCKERFILE)
DOCKER_RECOMMENDATION_IMAGE_NAME = $(DOCKER_REPOSITORY):$(RECOMMENDATION_VERSION)

RATING := SmallRating

.PHONY: req_recommendation
req_recommendation:
	cd $(RECOMMENDATION_DIR) && \
	poetry export \
		--without-hashes \
		-f requirements.txt \
		--output requirements.txt

.PHONY: build_recommendation
build_recommendation:
	docker build \
		--platform x86_64 \
		-t $(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		-f $(DOCKERFILE_RECOMMENDATION) \
		.

.PHONY: push_recommendation
push_recommendation:
	docker push $(DOCKER_RECOMMENDATION_IMAGE_NAME)

.PHONY: pull_recommendation
pull_recommendation:
	docker pull $(DOCKER_RECOMMENDATION_IMAGE_NAME)

.PHONY: run_download
run_download:
	docker run \
		-it \
		--rm \
		--name=download \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			download-command

.PHONY: run_small_rating
run_small_rating:
	docker run \
		-it \
		--rm \
		--name=small_rating \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			small-rating-command \
			--rate 0.1

.PHONY: run_random_recommend
run_random_recommend:
	docker run \
		-it \
		--rm \
		--name=random_recommend \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		-e RATING=$(RATING) \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			recommend \
			--num_users 1000 \
			--num_test_items 5 \
			--top_k 10 \
			random-recommend

.PHONY: run_popularity_recommend
run_popularity_recommend:
	docker run \
		-it \
		--rm \
		--name=popularity_recommend \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		-e RATING=$(RATING) \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			recommend \
			--num_users 1000 \
			--num_test_items 5 \
			--top_k 10 \
			popularity-recommend \
			--minimum_num_rating 200

.PHONY: run_association_recommend
run_association_recommend:
	docker run \
		-it \
		--rm \
		--name=popularity_recommend \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			recommend \
			--num_users 1000 \
			--num_test_items 5 \
			--top_k 10 \
			association-recommend \
			--min_support 0.1 \
			--min_threshold 1.0

.PHONY: run_umcf_recommend
run_umcf_recommend:
	docker run \
		-it \
		--rm \
		--name=umcf_recommend \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		-e RATING=$(RATING) \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			recommend \
			--num_users 1000 \
			--num_test_items 5 \
			--top_k 10 \
			umcf-recommend

.PHONY: run_regression_recommend
run_regression_recommend:
	docker run \
		-it \
		--rm \
		--name=regression_recommend \
		--platform linux/x86_64 \
		-v $(RECOMMENDATION_DIR)/data:/opt/data \
		$(DOCKER_RECOMMENDATION_IMAGE_NAME) \
		python \
			-m src.main \
			recommend \
			--num_users 1000 \
			--num_test_items 5 \
			--top_k 10 \
			regression-recommend


############ ALL COMMANDS ############
.PHONY: req_all
req_all: \
	req \
	req_recommendation

.PHONY: build_all
build_all: \
	build_recommendation

.PHONY: push_all
push_all: \
	push_recommendation

.PHONY: pull_all
pull_all: \
	pull_recommendation

.PHONY: run_small
run_small: run_random_recommend \
	run_popularity_recommend \
	run_association_recommend \
	run_umcf_recommend \
	run_regression_recommend
