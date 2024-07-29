# Rain-Weather-Classification-for-Vehicles

## Dataset description
* The training dataset was extracted from different youtube videos and github. It comprises 2 classes “RAIN” and “OTHER”.
* The class “RAIN” contains 8,811 images and class “OTHER” contains 6,877 images.

```
dataset  # train dataset
├── OTHER                
│   ├── 000001.jpg
│   ├── 000002.jpg
│   ├── 000003.jpg
│   └── .
│   └── .
│   └── .
├── RAIN                                
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   └── .
│   └── .
│   └── .
```
You can download the dataset from :- [Google Drive Link](https://drive.google.com/file/d/1abPK-cDwhRe_wovZk4wv7j5uxUPrQRo4/view?usp=sharing)


## Model Architecture
### LightClassifier Model Architecture:
* The LightClassifier class is a deep learning model designed for classifying traffic lights using the ShuffleNetV2 architecture.
* ShuffleNetV2 Base:
* The model uses ShuffleNetV2 as its backbone. ShuffleNetV2 is a lightweight and efficient neural network architecture designed for mobile and embedded vision applications. The architecture involves grouped convolutions, channel shuffling, and pointwise convolutions, which reduce the computational cost while maintaining accuracy.
* Global Pooling and Fully Connected Layer: After the backbone, the model applies global average pooling to reduce the spatial dimensions to 1x1.A fully connected (fc) layer maps the pooled features to the number of classes (in this case, traffic light classes).
### The training script splits the dataset into training and validation sets, applies data augmentation, and trains the model for a specified number of epochs. Here's a detailed breakdown:
* Number of Epochs: The training process is set to run for 201 epochs.
* Training and Validation Size:
* The dataset is split into training and validation sets with a 85% to 15% ratio.
* Data Augmentation and Transformations:
* Data augmentation is applied to the training set to improve generalization. This includes resizing, horizontal flipping, rotation, scaling, and normalization.
* The validation set only undergoes resizing and normalization.
* Training Batch Size:-  128




### The training loop handles the training process, including loss calculation, backpropagation, and validation. Here's a breakdown of the key steps:
* Forward Pass: The model processes the input images to produce feature vectors and logits.
* Loss Calculation: The combined loss (cross-entropy and center loss) is calculated.
* Backpropagation and Optimization: Gradients are computed and applied to update the model's weights.The Stochastic Gradient Descent (SGD) optimizer is used to update the model's parameters during training. In the provided code, SGD is applied to both the model parameters and the parameters of the center loss component.
* Validation: The model is evaluated on the validation set, and performance metrics are recorded.
* Saving Model Weights: Model weights are periodically saved during training.
* Input Size of the images : 200x200 pixels
* Output Size of the model : 2  [ rain, other]
### Results :- 
* Epoch 200: Minimum Loss : weighted loss :-6.685408,  Accuracy:- 99.62% (Final model used)
* Epoch 30: Maximum Accuracy: Accuracy :- 99.70% , weighted loss:- 38.536098


