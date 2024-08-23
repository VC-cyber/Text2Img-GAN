#rendering image with the help of the trained model

import torch
from GAN_model.embeddings import get_clip_model, get_caption_embedding

from GAN_model.generator import Generator
from PIL import Image


def save_image(tensor, path):
    # Convert tensor to numpy array and normalize to [0, 255]
    image = tensor.permute(1, 2, 0).detach().numpy()  # Convert to HWC
    image = (image * 255).astype('uint8')  # Scale to [0, 255]
    Image.fromarray(image).save(path)


text = "dog"

# Load the trained model
gen = Generator(256)

# Load the trained weights
gen.load_state_dict(torch.load("GAN_model/weights/generator.pth"))

# Generate an image from the text
gen.eval()

# Convert the text to a tensor
clip_model, clip_processor = get_clip_model()
inputs = clip_processor(text=text, return_tensors="pt", padding=True)
   
with torch.no_grad():
    text_embedding = clip_model.get_text_features(**inputs)

# Generate an image from the text
image = gen(torch.randn(1, 256), text_embedding).squeeze(0)

#shape
#print(image.shape)

# Save the generated image
save_image(image, "rendered_images/output9.png")
