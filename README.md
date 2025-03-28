Skin Cancer Classification Using Deep Learning
Overview
  The system incorporates a Gradient Boosting Algorithm for processing patient metadata and a Convolutional Neural Network (CNN) for assessing skin lesion photos. A multi-modal approach to classification is made possible by the dataset, which comes from the ISIC 2024 repository and includes over 400,000 tagged skin lesion photos with comprehensive information.  Advanced data preprocessing methods, such as feature engineering and picture augmentation, were used to improve performance. Users can submit their image and corresponding metadata of that particular image, where they can submit photographs of skin lesions and enter metadata for real-time analysis. After that, the system gives a classification result, which aids both individuals and medical professionals in determining the probability of malignancy. This project provides a quick, scalable, and precise method for early skin cancer detection by fusing deep learning and structured data analysis, which may enhance clinical results and lower diagnostic uncertainty. 
  With millions of new cases reported annually, skin cancer is one of the most prevalent and potentially fatal diseases in the world. Improving treatment results and survival rates requires early identification. Even seasoned dermatologists find it difficult to differentiate between benign and malignant skin tumors. Conventional diagnostic techniques depend on subjective eye inspection and dermoscopic analysis, which might result in incorrect diagnoses. Developments in machine learning (ML) and artificial intelligence (AI) have opened up new avenues for automated and more precise skin cancer detection in order to meet this problem. In order to create an intelligent skin cancer detection system that examines patient metadata and medical photographs, this research makes use of machine learning. Building a strong model that can help medical practitioners by making quick, accurate, and data-driven predictions is the main goal. The system analyzes structured patient metadata using a Gradient Boosting Algorithm and processes skin lesion photos using a Convolutional Neural Network (CNN). The model improves prediction accuracy by integrating information from both sources, increasing the likelihood of an accurate and timely diagnosis. The ISIC 2024 archive, a well-known collection of dermatology photos, provided the dataset for this study. In addition to metadata containing patient details, lesion location, and lesion characteristics, it includes more than 400,000 annotated skin lesion photos, each of which is categorized as either benign or malignant. Combining textual and image-based data increases the system's capacity for prediction.

  Proposed System 
  
Deep learning and machine learning are used in the suggested skin cancer detection method to improve diagnostic speed, accessibility, and accuracy. It combines a Gradient Boosting Algorithm (CatBoost) to process patient metadata, including age, sex, and lesion location, with a Convolutional Neural Network (CNN) to assess skin lesion photos. The approach lowers false positives and negatives and increases classification accuracy by merging image-based and metadata-driven predictions. To improve model robustness, the preprocessing pipeline consists of image augmentation, feature engineering, missing value handling, and one-hot encoding. This AI-driven methodology offers a non-invasive, automated, and scalable option for early skin cancer identification, in contrast to conventional biopsy-dependent procedures. As such, it is accessible to both healthcare professionals and people living in distant places. To guarantee privacy-preserving model improvements, future developments will incorporate explainable AI (Grad-CAM), mobile app integration, multi-class categorization, and federated learning. By cutting down on diagnostic delays and enhancing patient outcomes, this technology has the potential to completely transform the early diagnosis of skin cancer.

DATASET 
The ISIC 2024 dataset, which includes 401,059 annotated skin lesion images and comprehensive patient metadata, is used to train the model. 
Every picture is categorized as either: 
 Non-cancerous (benign) - Label: 0  
 Malignant (Cancerous) - Label: 1 
Beyond picture analysis, the metadata offers additional diagnostic features by containing patient specific data like age, sex, lesion location, lesion size, and colour characteristics. 

1. Data Collection & Preprocessing

Image Data Processing: 
 Dataset: ISIC 2024 archive containing 401,059 labeled skin lesion images. Image 
Augmentation: Flipping, brightness adjustment, noise addition, distortion to improve generalization; Resizing all images to 224x224 pixels for CNN processing. Metadata 
Processing: Handling missing values using median imputation. Feature engineering: Creating new features like lesion size ratio, color contrast, border irregularity scores; One-hot encoding for categorical data (e.g., lesion location, gender). 
2. Model Training: 
 Convolutional Neural Network (CNN) for Image Processing: The model extracts visual features like colour, texture, and shape to differentiate benign and malignant lesions. 
Gradient Boosting Model (CatBoost) for Metadata Processing: Uses patient metadata (age, sex, lesion size, etc.) to improve classification accuracy ; Handles imbalanced data using Stratified Group K-Fold cross-validation.  
 Combining Predictions (Hybrid Model Fusion): The CNN output (image-based probability) and CatBoost output (metadata-based probability) are combined using weighted averaging. 
3. Performance & Evaluation: 
 Evaluation Metrics Used: Accuracy, Precision, Recall, F1-score, and AUC (Area 
Under Curve). 
 Results Analysis: CNN alone achieved 82% accuracy, while metadata alone achieved 75% accuracy. Hybrid model (CNN + Metadata) achieved 88% accuracy, proving the effectiveness of the integrated approach.

OUTPUT: 
![Image](https://github.com/user-attachments/assets/b5a01d53-803a-4618-a806-b55ead68f729)
