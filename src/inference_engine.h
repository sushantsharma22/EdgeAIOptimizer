#include "inference_engine.h"
#include <iostream>

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "inference") {

    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    session_ = Ort::Session(env_, model_path.c_str(), session_options_);

    // Assume single input and output
    size_t num_input_nodes = session_.GetInputCount();
    size_t num_output_nodes = session_.GetOutputCount();

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    char* input_name = session_.GetInputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
    input_name_ = std::string(input_name);

    char* output_name = session_.GetOutputNameAllocated(0, Ort::AllocatorWithDefaultOptions()).get();
    output_name_ = std::string(output_name);

    // Get input shape
    Ort::TypeInfo input_type_info = session_.GetInputTypeInfo(0);
    auto tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
    input_shape_ = tensor_info.GetShape();
    // Typically: [N, C, H, W] = [1, 1, 28, 28]
}

void InferenceEngine::setSessionOptions(bool optimized) {
    if (optimized) {
        // Simulate optimization by enabling ORT optimizations
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        // Recreate session
        session_ = Ort::Session(env_, session_.GetSessionOptions().GetModelPath(), session_options_);
    } else {
        session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        session_ = Ort::Session(env_, session_.GetSessionOptions().GetModelPath(), session_options_);
    }
}

std::vector<float> InferenceEngine::runInference(const cv::Mat &input) {
    // Convert Mat to tensor
    std::vector<float> input_tensor_values = matToTensor(input);
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(),
                                                              input_tensor_values.size(),
                                                              input_shape_.data(), input_shape_.size());

    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };

    auto output_tensors = session_.Run(Ort::RunOptions{nullptr},
                                       input_names, &input_tensor, 1,
                                       output_names, 1);

    float* output_data = output_tensors.front().GetTensorMutableData<float>();
    size_t output_size = output_tensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
    std::vector<float> output_vector(output_data, output_data + output_size);
    return output_vector;
}

std::vector<float> InferenceEngine::matToTensor(const cv::Mat &img) {
    // Assume img is 28x28 grayscale [0..255]
    // Normalize to [0..1]
    cv::Mat float_img;
    img.convertTo(float_img, CV_32FC1, 1.0/255.0);

    // Flatten
    std::vector<float> tensor_values((float*)float_img.data, (float*)float_img.data + float_img.total());
    return tensor_values;
}
