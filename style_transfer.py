import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import VGG19_Weights
from PIL import Image

# -------------------------------
# Device (CPU safe)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Image Loader
# -------------------------------
def load_image(image_path, max_size=400):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize(max_size),
        transforms.ToTensor()
    ])
    image = transform(image).unsqueeze(0)
    return image.to(device)

# Load images
content_image = load_image("images/content.jpg")
style_image = load_image("images/style.jpg")

# -------------------------------
# Load Pretrained VGG19
# -------------------------------
vgg = models.vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

# Freeze VGG parameters
for param in vgg.parameters():
    param.requires_grad = False

# -------------------------------
# Layers to extract features
# -------------------------------
content_layers = ['21']
style_layers = ['0', '5', '10', '19', '28']

# -------------------------------
# Gram Matrix
# -------------------------------
def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    return torch.mm(tensor, tensor.t())

# -------------------------------
# Feature Extraction
# -------------------------------
def get_features(image, model):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in content_layers:
            features[f"content_{name}"] = x
        if name in style_layers:
            features[f"style_{name}"] = x
    return features

# Extract features
content_features = get_features(content_image, vgg)
style_features = get_features(style_image, vgg)

# Detach style Gram matrices (IMPORTANT FIX)
style_grams = {}
for layer in style_features:
    style_grams[layer] = gram_matrix(style_features[layer]).detach()

# -------------------------------
# Target Image (Start from Content)
# -------------------------------
target = content_image.clone().requires_grad_(True)

# -------------------------------
# Optimizer
# -------------------------------
optimizer = optim.Adam([target], lr=0.003)

# -------------------------------
# Weights
# -------------------------------
content_weight = 1
style_weight = 1e6

# -------------------------------
# Training Loop
# -------------------------------
steps = 300

for step in range(steps):
    target_features = get_features(target, vgg)

    # Content loss
    content_loss = torch.mean(
        (target_features["content_21"] - content_features["content_21"]) ** 2
    )

    # Style loss
    style_loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        style_loss += torch.mean((target_gram - style_gram) ** 2)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Step {step}, Total Loss: {total_loss.item()}")

# -------------------------------
# Save Output Image
# -------------------------------
output = target.squeeze().detach().cpu()
output = transforms.ToPILImage()(output)
output.save("stylized_output.png")

print("Neural Style Transfer complete. Image saved as stylized_output.png")
