from PIL import Image
import requests
from skimage import io
import os
import numpy as np
from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"

image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a dog", "a photo of a house"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
embeddings = outputs['image_embeds']
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

##  get emneddings
path_img = "./coco_data/train2017"
list_imgs = os.listdir(path_img)
embeddings = []
for i in range(len(list_imgs)):
    if i<500:
        image = Image.open(os.path.join(path_img, list_imgs[i]))
        inputs = processor(text=["a photo of a dog", "a photo of a house"], images=image, return_tensors="pt", padding=True)
        outputs = model(**inputs)
        embedding = outputs['image_embeds']
        embeddings.append(embedding.detach().cpu().numpy()[0])
                

embeddings = np.array(embeddings)
print(embeddings.shape, len(embeddings))
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)
from sklearn.manifold import TSNE
import numpy as np

# Reduce dimensions to 2D
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)
