#include "opencvutils.h"

// Resize an image to a given size to
cv::Mat __resize_to_a_size(cv::Mat image, int new_height, int new_width) {

  // get original image size
  int org_image_height = image.rows;
  int org_image_width = image.cols;

  // get image area and resized image area
  float img_area = float(org_image_height * org_image_width);
  float new_area = float(new_height * new_width);

  // resize
  cv::Mat image_scaled;
  cv::Size scale(new_width, new_height);

  if (new_area >= img_area) {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_LANCZOS4);
  } else {
    cv::resize(image, image_scaled, scale, 0, 0, cv::INTER_AREA);
  }

  return image_scaled;
}

// Normalize an image by subtracting mean and dividing by standard deviation
cv::Mat __normalize_mean_std(cv::Mat image, std::vector<double> mean, std::vector<double> std) {

  // clone
  cv::Mat image_normalized = image.clone();

  // convert to float
  image_normalized.convertTo(image_normalized, CV_32FC3);

  // subtract mean
  cv::subtract(image_normalized, mean, image_normalized);

  // divide by standard deviation
  std::vector<cv::Mat> img_channels(3);
  cv::split(image_normalized, img_channels);

  img_channels[0] = img_channels[0] / std[0];
  img_channels[1] = img_channels[1] / std[1];
  img_channels[2] = img_channels[2] / std[2];

  cv::merge(img_channels, image_normalized);

  return image_normalized;  
}

// 1. Preprocess
cv::Mat preprocess(cv::Mat image, int new_height, int new_width,
  std::vector<double> mean, std::vector<double> std) {

  // Clone
  cv::Mat image_proc = image.clone();

  // Convert from BGR to RGB
  cv::cvtColor(image_proc, image_proc, cv::COLOR_BGR2RGB);

  // Resize image
  image_proc = __resize_to_a_size(image_proc, new_height, new_width);

  // Convert image to float
  image_proc.convertTo(image_proc, CV_32FC3);

  // 3. Normalize to [0, 1]
  image_proc = image_proc / 255.0;

  // 4. Subtract mean and divide by std
  image_proc = __normalize_mean_std(image_proc, mean, std);

  return image_proc;
}
