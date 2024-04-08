# vintage-models

Hey lucky people, you have probably lost yourself to end up
here.

Anyway, welcome to my vintage-models repository. Here you will find some personal implementations
of few (I do not yet how many) of the most famous neural network models allowing to process images.

These implementations are made from my personal understanding of the scientific papers introducing
these models. Thus, some implementation/architecture choices can be (are probably) wrong. Even though
the likelihood one of you will really use code from here, please accept my deep apologies for that.

Beside better understanding these models behavior, I am also using this repository
as a road to better learn how to use PyTorch. Thus, in addition of questionable model architecture,
these implementations are probably weak version of "more official" ones.

## Implemented models

The list as been created from prompting OpenAI ChatGpt regarding the most famous models.

### Convolutional Neural Networks

- LeNet5: Developed by Yann LeCun and his collaborators, LeNet-5 was one of the earliest
convolutional neural networks (CNNs) and played a crucial role in the development
of image recognition techniques.

- Generative Adversarial Network (GAN): GANs, introduced by Ian Goodfellow and his colleagues
in 2014, have become one of the most influential and popular models for image generation.
GANs consist of a generator network and a discriminator network that are trained adversarially,
leading to the generation of realistic images.

### Transformers

- ViT (Vision Transformer) (2020): The original Vision Transformer introduced by Alexey Dosovitskiy
et al. demonstrated the effectiveness of using transformer architectures for image classification
tasks, challenging the dominance of convolutional neural networks.

### Autoencoder

- Variational Autoencoder (VAE): VAEs, introduced for image generation by Kingma and Welling in 2013,
are probabilistic generative models. VAEs learn a probabilistic mapping from the input space to a
latent space, enabling the generation of new samples by sampling from the latent space.

## How to run the code

### Install Python

```bash
# Install pyenv and pyenv-virtualenv
curl https://pyenv.run | bash

# Configure bash
# see https://github.com/pyenv/pyenv/issues/1906
# tested and working on yueh
echo >> ~/.bashrc 'export PATH="$HOME/.pyenv/bin:$PATH"'
echo >> ~/.bashrc 'eval "$(pyenv init --path)"'
echo >> ~/.bashrc 'eval "$(pyenv init -)"'
echo >> ~/.bashrc 'eval "$(pyenv virtualenv-init -)"'

# Reload bash's configuration
source ~/.bashrc

# Prepare your Ubuntu for building Python
sudo apt install build-essential libz-dev libssl-dev libbz2-dev libreadline-dev libsqlite3-dev libffi-dev

# Install Python
pyenv install 3.10.9
```

### Install the requirements

```bash
pyenv virtualenv 3.10.9 vintage-models
pyenv activate vintage-models
pip install -r requirements.txt
```

### Activate Pre-commits validation

Then activate the `pre-commit` hooks:

```
pre-commit install --allow-missing-config
```

> Warning: if you have aliases for `python` or `pip`, it will break pyenv. Please
verify that you are running the version of Python controlled by pyenv by
running `which python`. If the output does not contain the string `pyenv`,
something is wrong with your install


## Not exhaustive list of could be implemented models

The list as been created from prompting OpenAI ChatGpt regarding the most famous models.

### Convolutional Neural Networks

- AlexNet (2012): AlexNet, designed by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton,
was a milestone in the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) in 2012.
It popularized the use of deep convolutional neural networks for image classification tasks.

- GoogleNet (Inception) (2014): The Inception architecture, also known as GoogleNet, introduced the
concept of inception modules and significantly improved the efficiency and performance
of deep neural networks. It won the ILSVRC 2014 competition.

- DeepDream (2015): While not a traditional image classification model, DeepDream, developed by Google,
gained popularity for its ability to generate artistic and visually intriguing images through
neural network visualization techniques.

- YOLO (You Only Look Once) (2016): YOLO is an object detection algorithm that divides an
image into a grid and predicts bounding boxes and class probabilities directly.
YOLO models are known for their real-time object detection capabilities.

- Mask R-CNN (2017): An extension of Faster R-CNN, Mask R-CNN is widely used for instance
segmentation tasks, allowing for the detection and segmentation of objects within an image.


### Transformers

- DeiT (Data-efficient Image Transformer) (2021): DeiT, proposed by researchers at Facebook AI,
focused on improving the data efficiency of Vision Transformers, making them perform
well even with smaller datasets.

- Swin Transformer (2021): The Swin Transformer, introduced by researchers from Microsoft Research Asia,
featured a hierarchical design that allowed for better capturing of long-range dependencies in images,
leading to improved performance on various computer vision tasks.

- T2T-ViT (Tokens-to-Token ViT) (2021): T2T-ViT is an extension of the Vision Transformer architecture
that aims to enhance the tokenization strategy for better capturing spatial information in images.

- PVT (Pyramid Vision Transformer) (2021): PVT incorporated a pyramid structure to the Vision Transformer,
enabling the model to capture information at different spatial resolutions efficiently.

- TNT (Token-wise self-attention) (2021): TNT proposed a token-wise self-attention mechanism
to improve the Vision Transformer's performance in handling fine-grained details in images.

### Vision Language Models

- ViLBERT (2019): ViLBERT, or Vision-and-Language BERT, is designed to jointly reason about
both vision and language in a unified model. It can perform tasks such as image-text co-grounding,
where it localizes elements in an image corresponding to specific parts of a given text.

- DALL-E (2021): Developed by OpenAI, DALL-E is a generative model that can create images from textual descriptions.
It demonstrates the ability to generate diverse and creative visual outputs.

- CLIP (Contrastive Language-Image Pre-training) (2021): Developed by OpenAI, CLIP is a
vision-language model that can understand images and text in a unified manner.
It is trained to learn a joint representation space for images and their associated textual descriptions,
enabling tasks such as zero-shot image classification.
