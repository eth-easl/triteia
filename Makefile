benchmark:
	python ao/benchmarks/matmul.py
test:
	pytest
format:
	python -m black .