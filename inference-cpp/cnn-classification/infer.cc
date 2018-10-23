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

#include "../../utils/torchutils.h"
#include "../../utils/opencvutils.h"

std::tuple<std::string, std::string> infer(
  cv::Mat image,
  int image_height, int image_width,
  std::vector<double> mean, std::vector<double> std,
  std::vector<std::string> labels,
  std::shared_ptr<torch::jit::script::Module> model) {

  if (image.empty()) {
    std::cout << "WARNING: Cannot read image!" << std::endl;
  }

  std::string pred = "";
  std::string prob = "0.0";

  // Predict if image is not empty
  if (!image.empty()) {

    // Preprocess image
    image = preprocess(image, image_height, image_width,
      mean, std);

    // Forward
    std::vector<float> probs = forward({image, }, model);

    // Postprocess
    tie(pred, prob) = postprocess(probs, labels);
  }

  return std::make_tuple(pred, prob);
}

int main(int argc, char **argv) {

  if (argc != 4) {
    std::cerr << "usage: predict <path-to-image> <path-to-exported-script-module> <path-to-labels-file> \n";
    return -1;
  }

  std::string image_path = argv[1];
  std::string model_path = argv[2];
  std::string labels_path = argv[3];

  int image_height = 224;
  int image_width = 224;

  // Read labels
  std::vector<std::string> labels;
  std::string label;
  std::ifstream labelsfile (labels_path);
  if (labelsfile.is_open())
  {
    while (getline(labelsfile, label))
    {
      labels.push_back(label);
    }
    labelsfile.close();
  }

  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};

  cv::Mat image = cv::imread(image_path);
  std::shared_ptr<torch::jit::script::Module> model = read_model(model_path);

  std::string pred, prob;
  tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model);

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "PROBABILITY : " << prob << std::endl;

  return 0;
}
