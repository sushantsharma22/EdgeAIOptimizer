import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import onnx


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # x: [N, 1, 28, 28]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleNet()
model.eval()

# Dummy input: batch of 1 image, 1 channel, 28x28
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
onnx_model_path = "/Users/sushant-sharma/Documents/projects/EdgeAIOptimizer/src/model.onnx"
torch.onnx.export(model, dummy_input, onnx_model_path, input_names=["input"], output_names=["output"], opset_version=11)

print(f"Model exported to {onnx_model_path}")
