#ifndef UTILS_H
#define UTILS_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

cv::Mat preprocessImage(const std::string &image_path);

void printPredictions(const std::vector<float> &output);

#endif
