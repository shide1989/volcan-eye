-include .env
export

GAR_HOST := $(GCP_REGION)-docker.pkg.dev
REPO     := $(GAR_HOST)/$(GCP_PROJECT)/$(GAR_IMAGE)
IMAGE    := $(REPO)/$(GAR_IMAGE):latest
SERVICE  := $(GAR_IMAGE)

.DEFAULT_GOAL := help

help:
	@echo "run_api | pytest | test_volcano_db | setup | build | push | deploy | redeploy | print"

reinstall_package:
	@pip uninstall -y lewagon || :
	@pip install -e .

run_api:
	uvicorn api.app:app --reload

pytest:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes

test_volcano_db:
	PYTHONDONTWRITEBYTECODE=1 pytest -v --color=yes tests/test_volcano_db.py

check:
	@test -n "$(GCP_REGION)"  || (echo "GCP_REGION is not set"  && exit 1)
	@test -n "$(GCP_PROJECT)" || (echo "GCP_PROJECT is not set" && exit 1)
	@test -n "$(GAR_IMAGE)"   || (echo "GAR_IMAGE is not set"   && exit 1)
	@test -n "$(GAR_MEMORY)"  || (echo "GAR_MEMORY is not set"  && exit 1)
	@echo "image    = [$(IMAGE)]"
	@echo "service  = [$(SERVICE)]"

setup: check
	gcloud services enable artifactregistry.googleapis.com run.googleapis.com \
		--project $(GCP_PROJECT)
	@gcloud artifacts repositories describe $(GAR_IMAGE) \
		--location=$(GCP_REGION) --project=$(GCP_PROJECT) >/dev/null 2>&1 \
		|| gcloud artifacts repositories create $(GAR_IMAGE) \
			--repository-format=docker --location=$(GCP_REGION) \
			--project=$(GCP_PROJECT) --quiet
	gcloud auth configure-docker $(GAR_HOST) --quiet

build: check
	docker build --platform linux/amd64 -t $(IMAGE) .

push: build
	docker push $(IMAGE)

deploy: check
	gcloud run deploy $(SERVICE) \
		--image $(IMAGE) \
		--memory $(GAR_MEMORY) \
		--region $(GCP_REGION) \
		--project $(GCP_PROJECT) \
		--max-instances 1 \
		--update-env-vars DATABASE_URL=$(DATABASE_URL) \
		--allow-unauthenticated

redeploy: push deploy

print:
	@echo "region  = [$(GCP_REGION)]"
	@echo "project = [$(GCP_PROJECT)]"
	@echo "image   = [$(IMAGE)]"
	@echo "service = [$(SERVICE)]"
	@echo "memory  = [$(GAR_MEMORY)]"

run_train:
	python -c 'from ml_logic.main import train; train()'

run_evaluate:
	python -c 'from ml_logic.main import evaluate; evaluate()'
