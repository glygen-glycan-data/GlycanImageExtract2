import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import os

from BKGlycanExtractor.image_manager import Image_Manager

class ImageEmbeddingExtractor:
    """
    Extracts embeddings from images using pre-trained models --> resnet

    Summary:
    - Uses resnet pre-trained model along with weights (IMAGENET1K_V2). 
    Weights are automatically downloaded by pytorch and cached when the code runs.

    - Remove the final classification layer (keep feature extraction)
    - Process image: resize to 224x224, normalize
    - Forward pass: produces a spatial feature map [1, 2048, 7, 7]
    - Global average pooling: mean(dim=[2, 3]) → [1, 2048]
    - Result: 2048-dimensional embedding vector
    - The embedding captures visual features (edges, textures, shapes, objects) learned from ImageNet.

    - Scoring mechanism for embeddings - Cosine distance
        Cosine distance measures how similar the directions of two vectors are, regardless of their magnitude (length).
        Benefits:
        Robust to cropping: Cropped images can have similar feature patterns but different magnitudes. 
        Cosine distance focuses on direction, so it's less affected by magnitude differences.
        Normalization: Works well even if embeddings aren't L2-normalized (though normalization helps).

        (Magnitude based calculations are used in eucledian distance - so avoiding that metric.)

    - With cosine distance, typical thresholds for glycan based images from what I have observed:
        <=0.07: Very similar (same image, cropped)
        <0.07: Items in the image have some similarities (but I would not consider this as a match)
        >0.1: Different images

    - The model can be fine-tuned as well if needed - the base/early layers can be frozen
    """

       
    def __init__(self, model_name = 'resnet50', device = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Load pre-trained model
        self.model = self._load_model(model_name)
        self.model.eval()
        self.model.to(self.device)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_name: str) -> nn.Module:
        """Load and modify model to output embeddings."""
        model = models.resnet50(weights='IMAGENET1K_V2')
        
        # Remove final classification layer, keep features
        model = nn.Sequential(*list(model.children())[:-1])
        return model

    def extract_embedding(self, image_input):
        """        
        Args:
            image_input: Can be:
                - Path to an image file (str)
                - PIL Image object
                - numpy array (H, W, C) with values in [0, 255]
        """
        # Handle different input types
        if isinstance(image_input, str):
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            image = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            if image_input.dtype != np.uint8:
                if image_input.max() <= 1.0:
                    image_input = (image_input * 255).astype(np.uint8)
            image = Image.fromarray(image_input).convert('RGB')
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
            
            # ResNet outputs [batch, 2048, 7, 7] - need to pool spatial dimensions
            embedding = embedding.mean(dim=[2, 3])  # Global average pooling -> [batch, 2048]
            
            # Convert to numpy
            embedding = embedding.squeeze(0).cpu().numpy()  # Remove batch dim -> [embedding_dim]
            
            # Ensure 1D array
            if embedding.ndim > 1:
                embedding = embedding.flatten()
        
        return embedding

def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray):
    """
    Cosine similarity (higher = more similar, range: -1 to 1)
    """
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)


def cosine_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Cosine distance = 1 - cosine similarity
    Cosine distance (lower = more similar, range: 0 to 2)
    """
    return 1 - cosine_similarity(embedding1, embedding2)

def compare_images(image1, image2, model_name: str = 'resnet50', metric = 'cosine_distance'):
    """
    Compare two images and return similarity score.
    """
    extractor = ImageEmbeddingExtractor(model_name=model_name)
    
    emb1 = extractor.extract_embedding(image1)
    emb2 = extractor.extract_embedding(image2)

    score = cosine_distance(emb1, emb2)
    
    return score


if __name__ == '__main__':
    output_tsv = '/home/nmathias/GlycanImageExtract2/image_embeddings.tsv'

    fitz_images = Image_Manager(['/home/nmathias/GlycanImageExtract2/extract_fitz'], pattern='*.png')
    figcap_images = Image_Manager(['/home/nmathias/GlycanImageExtract2/extract_figcap'], pattern='*.png')

    with open(output_tsv, 'w', encoding='utf-8') as f:
        # header
        f.write("fitz_image\tfigcap_image\tscore\n")
        for fitz_image in fitz_images.images:
            for figcap_image in figcap_images.images:
                if os.path.basename(fitz_image).split('.')[0] == os.path.basename(figcap_image).split('.')[0]:
                    fitz_name = os.path.basename(fitz_image)
                    figcap_name = os.path.basename(figcap_image)

                    score = compare_images(fitz_image, figcap_image, metric='cosine_distance')
                    if score <= 0.45:
                        f.write(f"{fitz_name}\t{figcap_name}\t{score}\n")
