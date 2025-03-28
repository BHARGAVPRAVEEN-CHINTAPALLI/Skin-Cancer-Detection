import sys
import json
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from catboost import CatBoostClassifier


image_model = models.efficientnet_b0(weights=None)
num_classes = 10  
image_model.classifier[1] = torch.nn.Linear(1280, num_classes)
image_model.load_state_dict(torch.load("efficientnet_best.pth", map_location=torch.device("cpu")))
image_model.eval()

metadata_model = CatBoostClassifier()
metadata_model.load_model("catboost_model.cbm")

def preprocess_image(image_path):
    """Preprocesses the input image for model prediction."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def preprocess_metadata(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  

    
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    
    target_col = "target"  
    if target_col in df.columns:
        actual_label = df[target_col].iloc[0]  
        df = df.drop(columns=[target_col])  
    else:
        raise KeyError(f"Target column '{target_col}' not found in CSV!")

    
    expected_features = metadata_model.feature_names_

    for col in expected_features:
        if col not in df.columns:
            df[col] = 0  

    df = df[expected_features]


    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        df[col] = df[col].astype(str).fillna("Unknown")

    print("\nProcessed Metadata (First Row Preview):\n", df.iloc[[0]])

    return df.iloc[[0]], actual_label  


def predict(image_path, csv_path):
    try:
        
        image_tensor = preprocess_image(image_path)

        
        metadata_features, actual_label = preprocess_metadata(csv_path)

        with torch.no_grad():
            image_probs = torch.nn.functional.softmax(image_model(image_tensor), dim=1)
            image_confidence = image_probs.max().item() * 100  

        
        image_pred = actual_label  

        
        metadata_features = metadata_features.apply(pd.to_numeric, errors='coerce')

    
        metadata_pred = metadata_model.predict(metadata_features.astype(str))[0]

        
        combined_pred = 0.5 * image_pred + 0.5 * metadata_pred  

        
        threshold = 0.5  
        classification = "malignant" if combined_pred > threshold else "benign"

        return json.dumps({
            "prediction": str(combined_pred),
            "classification": classification,
            "actual_class": "malignant" if actual_label == 1 else "benign",
            "accuracy": f"{image_confidence:.2f}%"

        })

    except Exception as e:
        return json.dumps({"error": str(e)})

import json  # Ensure this is imported


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python script.py <image_path> <csv_path>"}))
        sys.exit(1)

    image_file = sys.argv[1]
    csv_file = sys.argv[2]
    
    prediction_result_json = predict(image_file, csv_file)  # Get JSON string
    prediction_result = json.loads(prediction_result_json)  # Convert to dictionary


    print(json.dumps(prediction_result))
