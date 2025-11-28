# Deep Learning Medical Recommendation System
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

- **Model Selected:**  

### Comparison to Naive Approach  

---

## Visual Interface Demo


Video demo of the project can be found here  
Streamlit site can be found here

---

## Results and Conclusions  

---

## Ethics Statement  


---

## Instructions on How to Run the Code

1. Clone the Repository  
`git clone URL`  
`cd FOLDER`

3. Install Dependencies  
`pip install -r requirements.txt`

4. Run the Streamlit App  
`streamlit run main.py`  
The app will open in your browser  

6. INSTRUCTIONS

