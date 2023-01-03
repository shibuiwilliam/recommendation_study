DIR := $(shell pwd)

DOCKERFILE := Dockerfile
DOCKER_REPOSITORY := shibui/recommendation_study

############ COMMON COMMANDS ############
.PHONY: lint
lint:
	black --check --diff --line-length 120 .

.PHONY: sort
sort:
	isort .

.PHONY: fmt
fmt: sort
	black --line-length 120 .

.PHONY: vet
vet:
	mypy .

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
