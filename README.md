# Serving PyTorch Models in C++

* This repository contains various examples to perform inference using PyTorch C++ API.
* Run `git clone https://github.com/Wizaron/pytorch-cpp-inference` in order to clone this repository.

## Environment

1. Dockerfiles can be found at `docker` directory. There are two dockerfiles; one for cpu and the other for cuda10. In order to build docker image, you should go to `docker/cpu` or `docker/cuda10` directory and run `docker build -t <docker-image-name> .`.
2. After creation of the docker image, you should create a docker container via `docker run -v <directory-that-this-repository-resides>:<target-directory-in-docker-container> -p 8181:8181 -it <docker-image-name>` (We will use 8181 to serve our PyTorch C++ model).
3. Inside docker container, go to the directory that this repository resides.
4. Download `libtorch` from [PyTorch Website](https://pytorch.org/get-started/locally/) (CPU : `https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.3.1%2Bcpu.zip` - CUDA10 : `https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip`).
5. Unzip libtorch via `unzip`. This will create `libtorch` directory that contains torch shared libraries and headers.

## Code Structure

* `models` directory stores PyTorch models.
* `libtorch` directory stores C++ torch headers and shared libraries to link the model against PyTorch.
* `utils` directory stores various utility function to perform inference in C++.
* `inference-cpp` directory stores codes to perform inference.

## Exporting PyTorch ScriptModule

* In order to export `torch.jit.ScriptModule` of ResNet18 to perform C++ inference, go to `models/resnet` directory and run `python3 resnet.py`. It will download pretrained ResNet18 model on ImageNet and create `models/resnet_model_cpu.pth` and (optionally) `models/resnet_model_gpu.pth` which we will use in C++ inference.

## Serving the C++ Model

* We can either serve the model as a single executable or as a web server.

### Single Executable

* In order to build a single executable for inference:
	1. Go to `inference-cpp/cnn-classification` directory.
	2. Run `./build.sh` in order to build executable, named as `predict`.
	3. Run the executable via `./predict <path-to-image> <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}>`.
	4. Example: `./predict image.jpeg ../../models/resnet/resnet_model_cpu.pth ../../models/resnet/labels.txt false`

### Web Server

* In order to build a web server for production:
	1. Go to `inference-cpp/cnn-classification/server` directory.
	2. Run `./build.sh` in order to build web server, named as `predict`.
	3. Run the binary via `./predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}>` (It will serve the model on `http://localhost:8181/predict`).
	4. Example: `./predict ../../../models/resnet/resnet_model_cpu.pth ../../../models/resnet/labels.txt false`
	4. In order to make a request, open a new tab and run `python test_api.py` (It will make a request to `localhost:8181/predict`).

## Acknowledgement

1. [pytorch](https://pytorch.org)
2. [crow](https://github.com/ipkn/crow)
3. [tensorflow_cpp_object_detection_web_server](https://github.com/CasiaFan/tensorflow_cpp_object_detection_web_server)
