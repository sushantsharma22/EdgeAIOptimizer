#include <iostream>
#include <chrono>
#include "inference_engine.h"
#include "optimizer.h"
#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: ./ai_inference_optimizer <model.onnx> <input_image>\n";
        return 1;
    }

    std::string model_path = argv[1];
    std::string image_path = argv[2];

    // Preprocess the image
    cv::Mat input_img = preprocessImage(image_path);
    if (input_img.empty()) {
        return 1;
    }

    // Baseline inference
    InferenceEngine engine(model_path);

    auto start = std::chrono::high_resolution_clock::now();
    auto baseline_output = engine.runInference(input_img);
    auto end = std::chrono::high_resolution_clock::now();
    double baseline_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Baseline Inference Time: " << baseline_ms << " ms\n";

    printPredictions(baseline_output);

    // Simulate optimization steps
    Optimizer::quantizeModel(model_path);
    Optimizer::fuseOperators(model_path);

    // Enable ONNX Runtime internal optimizations
    engine.setSessionOptions(true);

    start = std::chrono::high_resolution_clock::now();
    auto optimized_output = engine.runInference(input_img);
    end = std::chrono::high_resolution_clock::now();
    double optimized_ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "Optimized Inference Time: " << optimized_ms << " ms\n";
    std::cout << "Speedup: " << (baseline_ms / optimized_ms) << "x\n";

    printPredictions(optimized_output);

    return 0;
}
