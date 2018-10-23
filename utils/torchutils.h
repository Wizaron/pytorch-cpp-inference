#ifndef TORCHUTILS_H // To make sure you don't declare the function more than once by including the header multiple times.
#define TORCHUTILS_H

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

std::shared_ptr<torch::jit::script::Module> read_model(std::string);
std::vector<float> forward(std::vector<cv::Mat>,
  std::shared_ptr<torch::jit::script::Module>);
std::tuple<std::string, std::string> postprocess(std::vector<float>,
  std::vector<std::string>);

#endif
