# Task 05 – Neural Style Transfer

## Objective
The objective of this task is to apply the artistic style of one image to the content of another image using Neural Style Transfer.

## Approach
Neural Style Transfer was implemented using a pre-trained VGG19 convolutional neural network. The model extracts content and style features and optimizes a target image to combine both.

## Tools & Technologies
- Python
- PyTorch
- Torchvision
- VGG19 (Pre-trained Model)

## Implementation Steps
1. Loaded content and style images
2. Extracted feature representations using VGG19
3. Computed Gram matrices for style representation
4. Optimized a target image using gradient descent

## Result
The final output image preserves the content structure while adopting the artistic style of the style image.

## Learning Outcome
- Understanding CNN feature representations
- Applying optimization-based generative techniques
- Hands-on experience with Neural Style Transfer
