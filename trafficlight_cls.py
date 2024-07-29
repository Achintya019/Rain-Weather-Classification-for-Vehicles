from module.shufflenetv2 import ShuffleNetV2
import torch
import torch.nn as nn

class LightClassifier(nn.Module):
    def __init__(self, classes, load_param, debug=False):
        super(LightClassifier, self).__init__()
        
        self.stage_repeats = [4, 8, 4]
        self.stage_out_channels = [-1, 24, 48, 96, 192]
        self.base = ShuffleNetV2(self.stage_repeats, self.stage_out_channels, load_param)

        # Add a global average pooling layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Add a fully connected layer for classification
        self.fc = nn.Linear(self.stage_out_channels[-1], classes)
        
        self.debug = debug
        
    def forward(self, x):
        if self.debug :
            print("forward ", x.size())
        _, _, P3 = self.base(x)
        if self.debug :
            print("base output ", P3.size())
        x = self.global_pool(P3)
        if self.debug :
            print("after global pool ", x.size())
        features = x.view(x.size(0), -1)  # Flatten the tensor
        if self.debug :
            print("after tensor flattening ", x.size())
        logits = self.fc(features)
        if self.debug :
            print("final shape ", x.size())
        return features, logits

    
if __name__ == '__main__':
    model = LightClassifier(2, False, True)
    test_data = torch.rand(1, 3, 200, 200)
    features, logits = model(test_data)
    print("final shape ", features.size())
    print("logits ", logits.size())