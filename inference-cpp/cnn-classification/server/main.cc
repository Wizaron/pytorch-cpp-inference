#include "../infer.h"
#include "crow_all.h"
#include "base64.h"

int PORT = 8181;

int main(int argc, char **argv) {

  if (argc != 4) {
    std::cerr << "usage: predict <path-to-exported-script-module> <path-to-labels-file> <gpu-flag{true/false}> \n";
    return -1;
  }

  std::string model_path = argv[1];
  std::string labels_path = argv[2];
  std::string usegpu_str = argv[3];
  bool usegpu;

  if (usegpu_str == "true") {
      usegpu = true;
  } else {
      usegpu = false;
  }

  // Set image height and width
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

  // Define mean and std
  std::vector<double> mean = {0.485, 0.456, 0.406};
  std::vector<double> std = {0.229, 0.224, 0.225};

  // Load Model
  torch::jit::script::Module model = read_model(model_path, usegpu);

  // App
  crow::SimpleApp app;
  CROW_ROUTE(app, "/predict").methods("POST"_method, "GET"_method)
  ([&image_height, &image_width,
    &mean, &std, &labels, &model, &usegpu](const crow::request& req){
    crow::json::wvalue result;
    result["Prediction"] = "";
    result["Confidence"] = "";
    result["Status"] = "Failed";
    std::ostringstream os;

    try {
      auto args = crow::json::load(req.body);

      // Get Image
      std::string base64_image = args["image"].s();
      std::string decoded_image = base64_decode(base64_image);
      std::vector<uchar> image_data(decoded_image.begin(), decoded_image.end());
      cv::Mat image = cv::imdecode(image_data, cv::IMREAD_UNCHANGED);

      // Predict
      std::string pred, prob;
      tie(pred, prob) = infer(image, image_height, image_width, mean, std, labels, model, usegpu);

      result["Prediction"] = pred;
      result["Confidence"] = prob;
      result["Status"] = "Success";

      os << crow::json::dump(result);
      return crow::response{os.str()};

    } catch (std::exception& e){
      os << crow::json::dump(result);
      return crow::response(os.str());
    }

  });

  app.port(PORT).run();
  return 0;
}
