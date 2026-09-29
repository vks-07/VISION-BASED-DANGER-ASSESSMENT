import os
from datasets import load_dataset
from typing import Dict, Any

def inspect_dataset(dataset_name: str = "pranked03/ViDAS") -> None:
    """
    Loads a dataset from the Hugging Face Hub and prints its structure.
    
    Parameters:
        dataset_name (str): The Hugging Face dataset identifier.
    Returns:
        None
    """
    print(f"Downloading/Loading dataset '{dataset_name}'...")
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset(dataset_name)
        
        print("\n--- Dataset Structure ---")
        print(dataset)
        
        # Print the first row of the train split to understand the columns
        if 'train' in dataset:
            print("\n--- First Row of Train Split ---")
            sample = dataset['train'][0]
            for key, value in sample.items():
                print(f"{key}: {value}")
        else:
            print("\nNo 'train' split found. Keys available:", dataset.keys())
            
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    inspect_dataset()
