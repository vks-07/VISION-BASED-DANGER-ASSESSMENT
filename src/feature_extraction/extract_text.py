import os
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from typing import Tuple

def init_bert_model(model_name: str = "bert-base-uncased") -> Tuple[BertModel, BertTokenizer, str]:
    """
    Initializes the HuggingFace BERT model and tokenizer.
    
    Parameters:
        model_name (str): The specific BERT model to load.
    Returns:
        tuple: (model, tokenizer, device string)
    """
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

def process_text_through_bert(text: str, model: BertModel, tokenizer: BertTokenizer, device: str) -> np.ndarray:
    """
    Passes a text string through BERT to get its semantic embedding.
    
    Parameters:
        text (str): The text description.
        model (BertModel): Loaded BERT model.
        tokenizer (BertTokenizer): Loaded BERT tokenizer.
        device (str): Compute device.
    Returns:
        np.ndarray: The resulting text embedding.
    """
    # Tokenizer converts words into numbers and pads short sentences to a max length
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        # pooler_output represents the [CLS] token, which BERT uses as a summary of the entire sentence.
        # This extracts a single vector (shape: 1x768) that represents the semantic meaning of the whole text.
        embedding = outputs.pooler_output.cpu().numpy()
        
    return embedding

def extract_text_features(text: str, output_path: str) -> None:
    """
    Main function coordinating the text extraction pipeline.
    
    Parameters:
        text (str): The input text description.
        output_path (str): Filepath to save the .npy embedding.
    Returns:
        None
    """
    model, tokenizer, device = init_bert_model()
    embedding = process_text_through_bert(text, model, tokenizer, device)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, embedding)
    print(f"Saved text embedding to {output_path}")

if __name__ == "__main__":
    # Test block
    sample_description = "A person is recklessly skateboarding down a steep hill in heavy traffic."
    extract_text_features(sample_description, "../../data/embeddings/sample_text.npy")
