#include "infer.h"

int main(int argc, char **argv) {

  if (argc != 5) {
    std::cerr << "usage: predict <path-to-image> <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> \n";
    return -1;
  }

  std::string image_path = argv[1];
  std::string model_path = argv[2];
  std::string labels_path = argv[3];
  std::string usegpu_str = argv[4];
  bool usegpu;

  if (usegpu_str == "true") {
      usegpu = true;
  } else {
      usegpu = false;
  }

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
  torch::jit::script::Module model = read_model(model_path, usegpu);

  std::string pred, prob;
  tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);

  std::cout << "PREDICTION  : " << pred << std::endl;
  std::cout << "CONFIDENCE  : " << prob << std::endl;

  return 0;
}
