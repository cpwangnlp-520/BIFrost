.PHONY: build shell run test lint install clean

IMAGE ?= bif:latest
GPU ?= all
DATA_DIR ?= $(PWD)/data
OUTPUT_DIR ?= $(PWD)/outputs
MODEL_DIR ?= $(PWD)/models

build:
	docker compose build

shell:
	docker compose run --rm \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		bif /bin/bash

run:  ## run a specific shell script, e.g. make run CMD=scripts/01_build_pool.sh
CMD ?= /bin/bash
	docker compose run --rm \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		bif $(CMD)

train:  ## e.g. make train GPU=0,1
	docker compose run --rm \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		bif bash scripts/03_train_stage2.sh

bif:  ## e.g. make bif GPU=0,1
	docker compose run --rm \
		-e CUDA_VISIBLE_DEVICES=$(GPU) \
		bif bash scripts/05_run_bif.sh

lint:
	ruff check src/bif/
	ruff format --check src/bif/

format:
	ruff check --fix src/bif/
	ruff format src/bif/

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

clean:
	rm -rf outputs/ __pycache__ .ruff_cache *.egg-info
	docker compose down

# --- local dev without docker ---
local-install:
	pip install -e ".[dev]"

local-lint:
	ruff check src/bif/

local-format:
	ruff check --fix src/bif/ && ruff format src/bif/
