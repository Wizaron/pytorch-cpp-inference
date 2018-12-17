#include "infer.h"

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
