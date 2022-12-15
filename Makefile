timestamp := $(shell date +%Y%m%d%H%M%S)
include env.sh
export

compile:
	$$(mkdir -p ./logs/$@)
	$$(mkdir -p ./compiled)
	python src/training.py --timestamp $(timestamp) --compile_only &> ./logs/$@/$(timestamp).log
	gsutil cp ./compiled/training_$(timestamp).json $(PIPELINES_URI)/compiled_pipelines &>> ./logs/$@/$(timestamp).log

clean_vertex:
	$$(mkdir -p ./logs/$@)
	python src/utils/clean_vertex.py --all &> ./logs/$@/$(timestamp).log

clean_local:
	rm -rf logs compiled

clean_all:
	@ $(MAKE) clean_vertex
	@ $(MAKE) clean_local

default_run:
	$$(mkdir -p ./logs/$@)
	$$(mkdir -p ./compiled)
	python src/training.py --timestamp $(timestamp) --enable_caching &> ./logs/$@/$(timestamp).log
	gsutil cp ./compiled/training_$(timestamp).json $(PIPELINES_URI)/compiled_pipelines &>> ./logs/$@/$(timestamp).log

custom_run:
	$$(mkdir -p ./logs/$@)
	$$(mkdir -p ./compiled)
	python src/training.py $(params) --timestamp $(timestamp) &> ./logs/$@/$(timestamp).log
	printf 'Training parameters\n$(params) $(timestamp)' &>> ./logs/$@/$(timestamp).log
	gsutil cp ./compiled/training_$(timestamp).json $(PIPELINES_URI)/compiled_pipelines &>> ./logs/$@/$(timestamp).log

download_tensorboards:
	$$(mkdir -p ./logs/tensorboards)
	gsutil -m cp -r $(PIPELINES_URI)/tensorboards ./logs

local_env: configs/dev.txt
	python -m venv .venv
	( \
		source ./.venv/Scripts/activate; \
		python -m pip install --upgrade pip setuptools; \
		pip install -r configs/dev.txt; \
	)