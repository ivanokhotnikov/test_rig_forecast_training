timestamp := $(shell date +%Y%m%d%H%M%S)
include env.sh
export

compile:
	$$(mkdir -p ./compiled)
	python src/training.py --timestamp $(timestamp) --compile_only

clean_vertex:
	python src/utils/clean_vertex.py --all

clean_local:
	rm -rf logs compiled

clean_all:
	@ $(MAKE) clean_vertex
	@ $(MAKE) clean_local

default_run:
	$$(mkdir -p ./compiled)
	python src/training.py --timestamp $(timestamp) --enable_caching

forced_default_run:
	$$(mkdir -p ./compiled)
	python src/training.py --timestamp $(timestamp)

sync:
	gsutil rsync -m -r -d ./compiled $(PIPELINES_URI)/compiled_pipelines

custom_run:
	$$(mkdir -p ./compiled)
	python src/training.py $(params) --timestamp $(timestamp)

download_tensorboards:
	$$(mkdir -p ./logs/tensorboards)
	gsutil -m cp -r $(PIPELINES_URI)/tensorboards ./logs

local_env: configs/requirements-dev.txt
	python -m venv .venv
	source ./.venv/Scripts/activate && \
	python -m pip install --upgrade pip setuptools && \
	pip install -r configs/requirements-dev.txt