# Tiny YOLOv1 from scratch in PyTorch

## Purpose and description

This is a recreation of the Tiny YOLOv1 model in PyTorch, created with the aim of gaining a better understanding of its model structure, loss function used in the original paper, and the methods required for preparing data and evaluating object detection models.

As the computational requirements to train the original model are high, I decided to work on the Tiny version and reduce the number of neurons in the fully connected layer from 4096 to 496. 

Given that this is primarily a learning project, the definition of success is to create a theoretically sound model rather than training it to completion and testing it on a test dataset. Thus, I did not implement most of the measures against overfitting, and the model was trained for a limited number of epochs. 

The model exhibited a desirable trend in minimizing loss and increasing mean average precision.

---
## Structure
### Model
* 6 Blocks of (Conv2d, LeakyReLU, MaxPool2d)
* 3 Blocks of (Conv2d, LeakyReLU)
* Fully connected layer

### Data
* Dataset - https://www.kaggle.com/datasets/andrewmvd/face-mask-detection
