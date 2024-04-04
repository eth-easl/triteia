benchmark:
	python ao/benchmarks/matmul.py
	python ao/benchmarks/bmm.py
test:
	pytest
format:
	python -m black .