import transformers
from torchvision import transforms
import torch

def get_clip_model():
    #model used to get the embeddings
    clip_model = transformers.CLIPModel.from_pretrained("openai/clip-vit-base-patch32")

    #processor used to tokenize the text and images
    clip_processor = transformers.CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return clip_model, clip_processor
# Load your dataset with COCO captions and images
# For each image, get the associated captions and convert them to embeddings using CLIP
# COCO dataset provides 5 captions per image, you can randomly choose one per epoch
def get_caption_embedding(caption, clip_model, clip_processor):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        #print("MPS (Apple GPU) is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU. MPS not available.")

    inputs = clip_processor(text=caption, return_tensors="pt", padding=True).to(device)
   
    with torch.no_grad():
        text_embedding = clip_model.get_text_features(**inputs)
    return text_embedding

#getting similarity scores
def get_similarity_score(image, caption, clip_model, clip_processor):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        #print("MPS (Apple GPU) is available.")
    else:
        device = torch.device("cpu")
        print("Using CPU. MPS not available.")
    #print(image.shape)
    image = clip_processor(images=image, return_tensors="pt", padding=True).to(device)
    caption = clip_processor(text=caption, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_features = clip_model.get_image_features(**image)
        caption_features = clip_model.get_text_features(**caption)
        similarity = torch.cosine_similarity(image_features.to(device), caption_features.to(device)).to(device)
    return similarity.item()