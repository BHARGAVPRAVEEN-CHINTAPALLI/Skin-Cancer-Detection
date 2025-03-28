<h1>Skin Cancer Classification Using Deep Learning     
<h2>Overview   </br>
The system incorporates a Gradient Boosting Algorithm for processing patient metadata and a Convolutional Neural Network (CNN) for assessing skin lesion photos. A multi-modal approach to classification is made possible by the dataset, which comes from the ISIC 2024 repository and includes over 400,000 tagged skin lesion photos with comprehensive metadata.

Advanced data preprocessing methods, such as feature engineering and image augmentation, were used to improve performance. Users can submit skin lesion images along with corresponding metadata for real-time analysis. The system then provides a classification result to assist individuals and medical professionals in determining the probability of malignancy.

This project offers a fast, scalable, and accurate method for early skin cancer detection by integrating deep learning and structured data analysis, which may enhance clinical outcomes and reduce diagnostic uncertainty.

Skin cancer is one of the most prevalent and potentially fatal diseases worldwide, with millions of new cases annually. Early detection is crucial for improving survival rates. However, even experienced dermatologists struggle to distinguish between benign and malignant skin tumors. Traditional diagnostic techniques rely on subjective visual inspection, leading to misdiagnoses.

Advancements in Machine Learning (ML) and Artificial Intelligence (AI) have enabled more accurate and automated skin cancer detection. This project leverages deep learning to develop an intelligent skin cancer detection system that analyzes both patient metadata and medical images.

The Gradient Boosting Algorithm processes structured patient metadata, while the CNN evaluates skin lesion images. By integrating information from both sources, the model significantly improves prediction accuracy, increasing the likelihood of an early and reliable diagnosis.

The dataset for this study is obtained from the ISIC 2024 archive, a widely recognized dermatology dataset. It includes over 400,000 annotated skin lesion images, each labeled as benign or malignant, along with metadata containing patient details such as lesion location and characteristics.

Proposed System  
The proposed skin cancer detection system leverages deep learning and machine learning to improve diagnostic accuracy, accessibility, and speed. It combines:

A Convolutional Neural Network (CNN) to analyze skin lesion images.

A Gradient Boosting Algorithm (CatBoost) to process patient metadata, including age, sex, and lesion location.

By merging image-based and metadata-driven predictions, the system reduces false positives and false negatives, enhancing classification accuracy.

Key Features  
✅ Advanced Preprocessing: Image augmentation, feature engineering, missing value handling, and one-hot encoding.  
✅ Hybrid Learning Model: Combines CNN-based image classification with metadata-driven predictions.  
✅ Scalable & Non-Invasive: Provides automated, AI-driven early skin cancer detection without requiring invasive biopsies.  
✅ Future Enhancements:  

Explainable AI (Grad-CAM) for model interpretability.  

Mobile App Integration for real-time diagnosis.  

Federated Learning for privacy-preserving model improvements.  

Multi-Class Classification to detect various skin lesion types.  

This system can revolutionize early skin cancer detection by reducing diagnostic delays and improving patient outcomes.  

Dataset  
The model is trained using the ISIC 2024 dataset, containing 401,059 annotated skin lesion images and detailed patient metadata.  

Each image is categorized as either:  

Non-cancerous (Benign) → Label: 0  

Cancerous (Malignant) → Label: 1  

The metadata enhances the diagnostic process by including:  
📌 Patient Age  
📌 Sex  
📌 Lesion Location  
📌 Lesion Size & Color Characteristics  

Methodology  
1. Data Collection & Preprocessing  
🔹 Image Data Processing  
📌 Dataset: ISIC 2024 archive with 401,059 labeled skin lesion images.  
📌 Image Augmentation: Flipping, brightness adjustments, noise addition, and distortion for better generalization.  
📌 Image Resizing: All images resized to 224x224 pixels for CNN input.  

🔹 Metadata Processing  
📌 Handling Missing Values using median imputation.  
📌 Feature Engineering: Creating new features like lesion size ratio, color contrast, and border irregularity scores.  
📌 One-Hot Encoding for categorical variables (e.g., lesion location, gender).  

2. Model Training  
🔹 CNN for Image Processing  
The CNN extracts visual features such as color, texture, and shape, distinguishing between benign and malignant lesions.  

🔹 Gradient Boosting (CatBoost) for Metadata  
📌 Uses structured patient metadata (age, sex, lesion size, etc.) to enhance classification accuracy.  
📌 Handles imbalanced data using Stratified Group K-Fold cross-validation.  

🔹 Hybrid Model Fusion  
📌 The CNN output (image-based probability) and CatBoost output (metadata-based probability) are combined using weighted averaging.  

3. Performance & Evaluation  
🔹 Evaluation Metrics Used   
✅ Accuracy   
✅ Precision  
✅ Recall  
✅ F1-score  
✅ AUC (Area Under Curve)  

🔹 Results Analysis  
📌 CNN alone achieved 82% accuracy.  
📌 Metadata alone achieved 75% accuracy.  
📌 Hybrid Model (CNN + Metadata) achieved 88% accuracy, proving the effectiveness of combining both methods.  

Output
![Image](https://github.com/user-attachments/assets/b5a01d53-803a-4618-a806-b55ead68f729)

Conclusion  
This project presents a highly accurate, AI-driven approach to early skin cancer detection. By integrating deep learning for image processing and machine learning for metadata analysis, it provides enhanced diagnostic capabilities.  

Key Contributions:  
✅ Multi-Modal Approach: Fuses image and metadata processing for superior accuracy.  
✅ Scalability: Can be deployed in hospitals and mobile apps for real-time diagnosis.  
✅ Potential Impact: Can significantly reduce diagnostic errors and improve patient outcomes.  

With future advancements, such as explainable AI, mobile integration, and federated learning, this project has the potential to revolutionize skin cancer detection in clinical and remote healthcare settings.
