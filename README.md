<h1>Skin Cancer Classification Using Deep Learning</h1>

<h2>Overview</h2>
<p>The system incorporates a Gradient Boosting Algorithm for processing patient metadata and a Convolutional Neural Network (CNN) for assessing skin lesion photos. A multi-modal approach to classification is made possible by the dataset, which comes from the ISIC 2024 repository and includes over 400,000 tagged skin lesion photos with comprehensive metadata.</p>

<p>Advanced data preprocessing methods, such as feature engineering and image augmentation, were used to improve performance. Users can submit skin lesion images along with corresponding metadata for real-time analysis. The system then provides a classification result to assist individuals and medical professionals in determining the probability of malignancy.</p>

<p>This project offers a fast, scalable, and accurate method for early skin cancer detection by integrating deep learning and structured data analysis, which may enhance clinical outcomes and reduce diagnostic uncertainty.</p>

<h2>Skin Cancer and AI-Based Detection</h2>
<p>Skin cancer is one of the most prevalent and potentially fatal diseases worldwide, with millions of new cases annually. Early detection is crucial for improving survival rates. However, even experienced dermatologists struggle to distinguish between benign and malignant skin tumors. Traditional diagnostic techniques rely on subjective visual inspection, leading to misdiagnoses.</p>

<p>Advancements in Machine Learning (ML) and Artificial Intelligence (AI) have enabled more accurate and automated skin cancer detection. This project leverages deep learning to develop an intelligent skin cancer detection system that analyzes both patient metadata and medical images.</p>

<p>The Gradient Boosting Algorithm processes structured patient metadata, while the CNN evaluates skin lesion images. By integrating information from both sources, the model significantly improves prediction accuracy, increasing the likelihood of an early and reliable diagnosis.</p>

<p>The dataset for this study is obtained from the ISIC 2024 archive, a widely recognized dermatology dataset. It includes over 400,000 annotated skin lesion images, each labeled as benign or malignant, along with metadata containing patient details such as lesion location and characteristics.</p>

<h2>Proposed System</h2>
<p>The proposed skin cancer detection system leverages deep learning and machine learning to improve diagnostic accuracy, accessibility, and speed. It combines:</p>

<ul>
<li>A Convolutional Neural Network (CNN) to analyze skin lesion images.</li>
<li>A Gradient Boosting Algorithm (CatBoost) to process patient metadata, including age, sex, and lesion location.</li>
</ul>

<p>By merging image-based and metadata-driven predictions, the system reduces false positives and false negatives, enhancing classification accuracy.</p>

<h2>Key Features</h2>
<ul>
<li>✅ <b>Advanced Preprocessing:</b> Image augmentation, feature engineering, missing value handling, and one-hot encoding.</li>
<li>✅ <b>Hybrid Learning Model:</b> Combines CNN-based image classification with metadata-driven predictions.</li>
<li>✅ <b>Scalable & Non-Invasive:</b> Provides automated, AI-driven early skin cancer detection without requiring invasive biopsies.</li>
<li>✅ <b>Future Enhancements:</b>
  <ul>
    <li>Explainable AI (Grad-CAM) for model interpretability.</li>
    <li>Mobile App Integration for real-time diagnosis.</li>
    <li>Federated Learning for privacy-preserving model improvements.</li>
    <li>Multi-Class Classification to detect various skin lesion types.</li>
  </ul>
</li>
</ul>

<p>This system can revolutionize early skin cancer detection by reducing diagnostic delays and improving patient outcomes.</p>

<h2>Dataset</h2>
<p>The model is trained using the ISIC 2024 dataset, containing 401,059 annotated skin lesion images and detailed patient metadata.</p>

<p>Each image is categorized as either:</p>

<ul>
<li>📌 Non-cancerous (Benign) → Label: 0</li>
<li>📌 Cancerous (Malignant) → Label: 1</li>
</ul>

<p>The metadata enhances the diagnostic process by including:</p>
<ul>
<li>📌 Patient Age</li>
<li>📌 Sex</li>
<li>📌 Lesion Location</li>
<li>📌 Lesion Size & Color Characteristics</li>
</ul>

<h2>Methodology</h2>

<h3>1. Data Collection & Preprocessing</h3>

<h4>🔹 Image Data Processing</h4>
<ul>
<li>📌 Dataset: ISIC 2024 archive with 401,059 labeled skin lesion images.</li>
<li>📌 Image Augmentation: Flipping, brightness adjustments, noise addition, and distortion for better generalization.</li>
<li>📌 Image Resizing: All images resized to 224x224 pixels for CNN input.</li>
</ul>

<h4>🔹 Metadata Processing</h4>
<ul>
<li>📌 Handling Missing Values using median imputation.</li>
<li>📌 Feature Engineering: Creating new features like lesion size ratio, color contrast, and border irregularity scores.</li>
<li>📌 One-Hot Encoding for categorical variables (e.g., lesion location, gender).</li>
</ul>

<h3>2. Model Training</h3>

<h4>🔹 CNN for Image Processing</h4>
<p>The CNN extracts visual features such as color, texture, and shape, distinguishing between benign and malignant lesions.</p>

<h4>🔹 Gradient Boosting (CatBoost) for Metadata</h4>
<ul>
<li>📌 Uses structured patient metadata (age, sex, lesion size, etc.) to enhance classification accuracy.</li>
<li>📌 Handles imbalanced data using Stratified Group K-Fold cross-validation.</li>
</ul>

<h4>🔹 Hybrid Model Fusion</h4>
<ul>
<li>📌 The CNN output (image-based probability) and CatBoost output (metadata-based probability) are combined using weighted averaging.</li>
</ul>

<h3>3. Performance & Evaluation</h3>

<h4>🔹 Evaluation Metrics Used</h4>
<ul>
<li>✅ Accuracy</li>
<li>✅ Precision</li>
<li>✅ Recall</li>
<li>✅ F1-score</li>
<li>✅ AUC (Area Under Curve)</li>
</ul>

<h4>🔹 Results Analysis</h4>
<ul>
<li>📌 CNN alone achieved 82% accuracy.</li>
<li>📌 Metadata alone achieved 75% accuracy.</li>
<li>📌 Hybrid Model (CNN + Metadata) achieved 88% accuracy, proving the effectiveness of combining both methods.</li>
</ul>

<h2>Output</h2>
<img src="https://github.com/user-attachments/assets/b5a01d53-803a-4618-a806-b55ead68f729" alt="Model Output"/>

<h2>Conclusion</h2>
<p>This project presents a highly accurate, AI-driven approach to early skin cancer detection. By integrating deep learning for image processing and machine learning for metadata analysis, it provides enhanced diagnostic capabilities.</p>

<h3>Key Contributions:</h3>
<ul>
<li>✅ <b>Multi-Modal Approach:</b> Fuses image and metadata processing for superior accuracy.</li>
<li>✅ <b>Scalability:</b> Can be deployed in hospitals and mobile apps for real-time diagnosis.</li>
<li>✅ <b>Potential Impact:</b> Can significantly reduce diagnostic errors and improve patient outcomes.</li>
</ul>

<p>With future advancements, such as explainable AI, mobile integration, and federated learning, this project has the potential to revolutionize skin cancer detection in clinical and remote healthcare settings.</p>
