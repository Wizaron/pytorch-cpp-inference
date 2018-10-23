#ifndef OPENCVUTILS_H // To make sure you don't declare the function more than once by including the header multiple times.
#define OPENCVUTILS_H

#include <iostream>
#include <vector>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>

cv::Mat preprocess(cv::Mat, int, int,
  std::vector<double>,
  std::vector<double>);

#endif
