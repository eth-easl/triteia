benchmark:
	python3 ao/benchmarks/matmul.py
	python3 ao/benchmarks/bmm.py
test:
	pytest tests --cov=triteia --cov-report html --cov-report term --cache-clear
format:
	python3 -m black . --exclude 3rdparty
install:
	pip3 install -e .
build-whl:
	python -m build --no-isolation
build-whl-container:
	bash scripts/build_whl.sh
