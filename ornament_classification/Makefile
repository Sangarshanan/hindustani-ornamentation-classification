.PHONY: setup test transcribe evaluate ablation demo clean

CONDA_ENV = hcm-transcription
CONFIG = configs/default.yaml

setup:
	conda env create -f environment.yml || conda env update -f environment.yml
	pip install -e .

test:
	pytest tests/ -v

transcribe:
	python scripts/run_transcription.py --config $(CONFIG)

evaluate:
	python scripts/evaluate_notes.py --config $(CONFIG)
	python scripts/evaluate_ornaments.py --config $(CONFIG)

ablation:
	python scripts/ablation_window_size.py --config $(CONFIG)

demo:
	jupyter notebook notebooks/demo.ipynb

clean:
	rm -rf results/*.csv results/*.json results/*.png
	rm -rf outputs/humdrum/*.krn
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
