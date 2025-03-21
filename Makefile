install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt
	#force install latest whisper
	# pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git
test:
	#python -m pytest -vv --cov=main --cov=mylib test_*.py # runs the files "test_*.py" that are expected to contain test functions
	#python -m pytest --nlval notebook.ipynb # module nlval test if notebook.ipynb has errors. 

format:	
	black *.py

lint:
	pylint --disable=R,C hello.py

container-lint:
	docker run --rm -i hadolint/hadolint < Dockerfile

checkgpu:
	echo "Checking GPU for PyTorch"
	python utils/verify_pytorch.py
	echo "Checking GPU for Tensorflow"
	python utils/verify_tf.py

refactor: format lint

deploy:
	#deploy goes here
		
all: install lint test format deploy
