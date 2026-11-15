.PHONY: install bind demo test serve history strict clean

install:
	python -m pip install -e ".[test]"

bind:
	python -m tagconsensus bind-adjudication --config configs/demo.yaml

demo:
	python -m tagconsensus demo --config configs/demo.yaml

test:
	pytest -q

serve:
	python -m tagconsensus serve --config configs/demo.yaml

history:
	python -m tagconsensus history --db var/tagconsensus.db

strict:
	python -m tagconsensus demo --config configs/strict.yaml

clean:
	rm -rf output output_strict var .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
