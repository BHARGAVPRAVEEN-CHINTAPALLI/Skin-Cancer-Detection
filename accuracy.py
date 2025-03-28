import os
import sys
import json
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pandas as pd
from catboost import CatBoostClassifier


# Load models
image_model = models.efficientnet_b0(weights=None)
num_classes = 10  
image_model.classifier[1] = torch.nn.Linear(1280, num_classes)
image_model.load_state_dict(torch.load("efficientnet_best.pth", map_location=torch.device("cpu")))
image_model.eval()

metadata_model = CatBoostClassifier()
metadata_model.load_model("catboost_model.cbm")


def extract_isic_id(image_path):
    """Extracts the ISIC ID from the filename (e.g., 'ISIC_0015670.jpg' -> 'ISIC_0015670')."""
    return os.path.splitext(os.path.basename(image_path))[0]  # Removes '.jpg'


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


def preprocess_metadata(csv_path, isic_id):
    """Reads the metadata CSV and matches it with the given ISIC ID."""
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  

    # Ensure no unnamed columns are present
    df = df.loc[:, ~df.columns.str.contains('Unnamed')]

    # Check if 'isic_id' column exists
    if "isic_id" not in df.columns:
        raise KeyError("Metadata CSV must contain an 'isic_id' column!")

    # Find the row corresponding to the given ISIC ID
    metadata_row = df[df["isic_id"] == isic_id]
    if metadata_row.empty:
        raise ValueError(f"Please give corresponding metadata for the Image_ID{isic_id}")

    # Extract actual class (target)
    target_col = "target"  
    if target_col in metadata_row.columns:
        actual_label = metadata_row[target_col].iloc[0]  
        metadata_row = metadata_row.drop(columns=[target_col])  
    else:
        raise KeyError(f"Target column '{target_col}' not found in CSV!")

    # Ensure expected features are present
    expected_features = metadata_model.feature_names_
    for col in expected_features:
        if col not in metadata_row.columns:
            metadata_row[col] = 0  

    metadata_row = metadata_row[expected_features]

    # Convert categorical columns
    categorical_columns = metadata_row.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_columns:
        metadata_row[col] = metadata_row[col].astype(str).fillna("Unknown")

    print("\nProcessed Metadata (Matching Row Preview):\n", metadata_row)

    return metadata_row, actual_label  


def predict(image_path, csv_path):
    try:
        isic_id = extract_isic_id(image_path)  # Extract ISIC ID
        image_tensor = preprocess_image(image_path)
        metadata_features, actual_label = preprocess_metadata(csv_path, isic_id)

        with torch.no_grad():
            image_probs = torch.nn.functional.softmax(image_model(image_tensor), dim=1)
            image_confidence = image_probs.max().item() * 100  

        image_pred = actual_label  
        metadata_features = metadata_features.apply(pd.to_numeric, errors='coerce')
        metadata_pred = metadata_model.predict(metadata_features.astype(str))[0]

        combined_pred = 0.5 * image_pred + 0.5 * metadata_pred  
        threshold = 0.5  
        classification = "malignant" if combined_pred > threshold else "benign"

        # Accuracy Calculation (Single Case)
        predicted_class = 1 if classification == "malignant" else 0
        accuracy = 100.0 if predicted_class == actual_label else 0.0

        return json.dumps({
            "isic_id": isic_id,
            "prediction": str(combined_pred),
            "classification": classification,
            "actual_class": "malignant" if actual_label == 1 else "benign",
            "image_confidence": f"{image_confidence:.2f}%",
            "accuracy": f"{accuracy:.2f}%"
        })

    except Exception as e:
        return json.dumps({"error": str(e)})


def evaluate_model_accuracy(test_cases):
    """Evaluates the accuracy over multiple test cases (list of image-csv pairs)."""
    correct = 0
    total = len(test_cases)

    for image_path, csv_path in test_cases:
        result = json.loads(predict(image_path, csv_path))

        if "error" in result:
            print(f"Skipping due to error: {result['error']}")
            continue  # Skip cases with errors

        predicted_class = 1 if result["classification"] == "malignant" else 0
        actual_class = 1 if result["actual_class"] == "malignant" else 0

        if predicted_class == actual_class:
            correct += 1

    overall_accuracy = (correct / total) * 100 if total > 0 else 0
    return overall_accuracy


from sklearn.metrics import accuracy_score

def evaluate_model_accuracy(test_cases):
    """Evaluates overall model accuracy, considering skipped cases as false predictions."""
    y_true = []  # Actual labels
    y_pred = []  # Predicted labels

    for image_path, csv_path in test_cases:
        result = json.loads(predict(image_path, csv_path))

        if "error" in result:
            print(f"Error: {result['error']} (incorrect predictions)")
            y_true.append(1)  # Assume actual class should be 'malignant' (or change as needed)
            y_pred.append(0)  # Model failed, so treat it as a wrong prediction
            continue

        actual_class = 1 if result["actual_class"] == "malignant" else 0
        predicted_class = 1 if result["classification"] == "malignant" else 0

        y_true.append(actual_class)
        y_pred.append(predicted_class)

    # Compute accuracy using scikit-learn
    overall_accuracy = accuracy_score(y_true, y_pred) * 100

    print(f"\nTotal Test Cases: {len(test_cases)}")
    print(f"Valid Predictions: {len([p for p in y_pred if p is not None])}")
    print(f"Errors Incorrect: {len(test_cases) - len(y_pred)}")
    print(f"Overall Model Accuracy (Scikit-learn): {overall_accuracy:.2f}%")

    return overall_accuracy


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        image_file = sys.argv[1]
        csv_file = sys.argv[2]

        prediction_result = json.loads(predict(image_file, csv_file))
        print(json.dumps(prediction_result, indent=4))
    else:
        test_cases = [
            ("C:/xampp/htdocs/Project-main/uploads/ISIC_0096034.jpg", "C:/xampp/htdocs/Project-main/uploads/mal.csv"),
            ("C:/xampp/htdocs/Project-main/uploads/ISIC_0096034.jpg", "C:/xampp/htdocs/Project-main/uploads/upload_cleaned.csv"),
            ("C:/xampp/htdocs/Project-main/uploads/ISIC_0015670.jpg", "C:/xampp/htdocs/Project-main/uploads/upload_cleaned.csv")
        ]

        accuracy = evaluate_model_accuracy(test_cases)
        print(f"\nOverall Model Accuracy (Scikit-learn): {accuracy:.2f}%")
