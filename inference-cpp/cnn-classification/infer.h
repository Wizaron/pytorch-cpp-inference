#ifndef INFER_H // To make sure you don't declare the function more than once by including the header multiple times.
#define INFER_H

#include <iostream>
#include <vector>
#include <tuple>
#include <chrono>
#include <fstream>
#include <random>
#include <string>
#include <memory>

#include <torch/script.h>
#include <torch/serialize/tensor.h>
#include <torch/serialize.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "../../utils/torchutils.h"
#include "../../utils/opencvutils.h"

std::tuple<std::string, std::string> infer(
  cv::Mat,
  int, int,
  std::vector<double>, std::vector<double>,
  std::vector<std::string>,
  std::shared_ptr<torch::jit::script::Module>);

#endif
