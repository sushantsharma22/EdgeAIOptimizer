#include "utils.h"
#include <iostream>
#include <algorithm>

cv::Mat preprocessImage(const std::string &image_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to read image: " << image_path << "\n";
        return img;
    }
    // Resize to 28x28 for our model
    cv::resize(img, img, cv::Size(28,28));
    return img;
}

void printPredictions(const std::vector<float> &output) {
    // Our model predicts 10 classes (digits 0-9)
    if (output.size() != 10) {
        std::cerr << "Unexpected output size.\n";
        return;
    }

    int predicted_class = (int)(std::max_element(output.begin(), output.end()) - output.begin());
    std::cout << "Predicted Digit: " << predicted_class << "\n";
}
