# PyTorch C++ Inference Examples

* This repository contains various examples to perform inference using PyTorch C++ API.
* Run `git clone https://github.com/Wizaron/pytorch-cpp-inference` in order to clone this repository.

## Environment

1. Dockerfile can be found at `docker` directory. In order to build docker image, you should go to `docker` directory and run `docker build -t <docker-image-name> .`.
2. After creation of the docker image, you should create a docker container via `docker run -v <directory-that-this-repository-resides>:<target-directory-in-docker-container> -it <docker-image-name>`.
3. Inside docker container, go to the directory that this repository resides.
4. Download `libtorch` from [PyTorch Website](https://pytorch.org/get-started/locally/) or using `wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip`.
5. Unzip it via `unzip libtorch-shared-with-deps-latest.zip`. This will create `libtorch` directory that contains torch shared libraries and headers.

## Export PyTorch ScriptModule

* In order to export `torch.jit.ScriptModule` of ResNet18 to perform C++ inference, go to `models/resnet` directory and run `python resnet.py`. It will download pretrained ResNet18 model on ImageNet and create `models/resnet_model.pth` which we will use in C++ inference.

## C++ Inference Example Build

* `libtorch` directory stores C++ torch headers and shared libraries to link the model against PyTorch.
* `utils` directory stores various utility function to perform inference in C++.
* `inference-cpp` directory stores codes to perform inference.

* In order to build executable:
	1. Go to `inference-cpp/cnn-classification` directory.
	2. Run `./build.sh` in order to build executable, named as `predict`.
	3. Run the executable via `./predict <path-to-image> <path-to-exported-script-module> <path-to-labels-file>`.
	4. Example: `./predict image.jpeg ../../models/resnet/resnet_model.pth labels.txt`
