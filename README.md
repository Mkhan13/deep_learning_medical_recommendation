# Medical Treatment Recommendation System
## Problem
People often struggle to interpret their symptoms and determine the appropriate next step for care. Searching online can lead to incorrect guidance, unnecessary fear or alarm, or delays in seeking treatment for something major. The goal of this project is to to build an AI symptom recommendation system that converts a user’s symptom description into a plausible condition and a recommended action to take. The goal is not to provide medical diagnosis in any capacity, but to identify whether symptoms align with common conditions and identify when immediate medical intervention may be necessary.

---

## Data Source
[Disease Diagnosis Dataset](https://www.kaggle.com/datasets/s3programmer/disease-diagnosis-dataset/data) on Kaggle
- 2000+ rows
- Symptom, diagnosis, and treatment columns

This dataset provides a foundation for learning relationships between symptoms and potential conditions or therapeutic actions, which helps with the development of this symptom to recommendation model

---

## Review of Relevant Previous Efforts and Literature  
There have been previous efforts to make an AI bot that can make medical diagnoses or treatment recommendations. [This bot](https://hms.harvard.edu/news/ai-system-detailed-diagnostic-reasoning-makes-its-case), created by Harvard researchers, is able to diagnose challenging medical cases and explain its reasoning behind a diagnosis. [This literature review](https://www.sciencedirect.com/science/article/pii/S2772632024000035) reviews different AI algorithms as recommender systems with a focus on integrating AI into medical settings. 

**My Contribution:**  
My project takes these diagnostic tools a step further by including treatment recommendations

---

## Model Evaluation Process & Metric Selection   
- **Metrics:**
  - Accuracy  
  - Precision, Recall
  - F1-score

- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same split. The results are compared directly against the naive baseline.

---

## Modeling Approach  
1. **Naive Baseline:** Predicts the most common medical condition in the dataset
2. **Classical ML Approach:** Uses TF-IDF features of the symptoms text to train a logistic regression classifier
3. **Deep Learning Approach:** Fine-tunes a BERT text classification model to map symptoms to conditions


### Data Processing Pipeline  

The raw dataset consists of symptom descriptions and corresponding medical conditions. Entries are split into train (80%), validation (10%), and test (10%) sets using a stratified split to preserve class distribution. The text is cleaned by lowercasing, removing punctuation, and normalizing spacing.

The processed CSVs are stored under data/processed/:
```
data/processed/
├── train.csv
├── val.csv
└── test.csv
```

### Models Evaluated and Model Selected  
- **Evaluated Models:**
## Model Performance Comparison

| Approach                  | Accuracy | Precision | Recall | F1-score | Notes |
|---------------------------|----------|-----------|--------|----------|-------|
| **Naive Baseline**        | 0.585    | 0.3422    | 0.585  | 0.4318   | Predicts the most common class |
| **Classical ML**          | 0.655    | 0.5722    | 0.655  | 0.607    | Best performance among models; strong for short text |
| **Deep Learning**         | 0.615    | 0.5234    | 0.615  | 0.5540   | Limited by short, sparse symptom text |


- **Model Selected:**  Classical ML

### Comparison to Naive Approach  

The naive approach predicts the most common diagnosis for all patients and has an accuracy of 58.5%. The classical ML model uses TF-IDF and logistic regression to learn the patterns between symptoms and different diagnoses. This method improves the accuracy to 65.5% with better precision, recall, and f1-scores. 

---

## Visual Interface Demo
<img width="887" height="627" alt="Screenshot 2025-12-01 at 6 20 20 PM" src="https://github.com/user-attachments/assets/25be6452-eaac-4be8-8043-26f9a5a4e0b1" />


Video demo of the project can be found [here](https://drive.google.com/file/d/1kBkCDm5SrdklU8F2y9WzUqkJPTxgMEis/view?usp=sharing)  
Streamlit site can be found [here](https://medical-app-963698787646.us-central1.run.app/)

---

## Results and Conclusions  
Across the three evaluated approaches, the classical machine learning model achieved the strongest overall performance, with an accuracy of 65.5% and has the highest precision, recall, and F1-score. The naive baseline always predicts the most common diagnosis and achieved 58.5% accuracy, while the deep learning model reached an accuracy of 61.5% but it was not good at generalizing because of the short symptom descriptions. The classical ML model benefited slightly from TF-IDF’s ability to extract informative patterns from limited text, which allowed it to capture distinctions the deep learning model could not learn effectively with the available data.

The best perfoming model, the classical ml approach, still has a relatively low accuracy due to limitations in the dataset. Many symptom descriptions lacked detail or variation which constrained the amount of meaningful signal the models could learn. Despite this, the classical ML approach demonstrated the most reliable and interpretable performance, making it the preferred choice for this task. These results suggest that model performance could improve significantly with more descriptive symptom descriptions or additional features.

---
 
## Ethics Statement  
This project is intended for educational and research purposes only and should not be used to assess or diagnose illnesses. The model's outputs are not a professional diagnosis and should not replace professional care or clinical advice. Please seek medical attention if you are feeling sick. Any real-world application of this tool should undergo ethical review and validation to ensure accuracy, user safety, and data privacy.

---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone https://github.com/Mkhan13/medical_recommendation.git`  
`cd medical_recommendation`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. Write sickness symptoms in the provided area as a comma separated list

