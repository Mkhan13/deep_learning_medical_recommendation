# Deep Learning Medical Recommendation System
## Problem
People often struggle to interpret their symptoms and determine the appropriate next step for care. Searching online can lead to incorrect guidance, unnecessary fear or alarm, or delays in seeking treatment for something major. The goal of this project is to to build an AI symptom recommendation system that converts a user’s symptom description into a plausible condition and a recommended action to take. The goal is not to provide medical diagnosis in any capacity, but to identify whether symptoms align with common conditions and identify when immediate medical intervention may be necessary.

---

## Data Source
[Medical Chatbot Dataset](https://www.kaggle.com/datasets/redflame03/medical-chatbot-dataset-gen-ai) on Kaggle
- 463 rows and 11 columns
- Structured clinical information linking symptoms, conditions, treatments, and demographic context

This dataset provides a foundation for learning relationships between symptoms and potential conditions or therapeutic actions, which helps with the development of this symptom to recommendation model

---

## Review of Relevant Previous Efforts and Literature  


**My Contribution:**  


---

## Model Evaluation Process & Metric Selection   
- **Metrics:**  

- **Data Splits:** Stratified 80%/10%/10% split for train/validation/test 

All three approaches (naive, classical ML, and deep learning) are trained and evaluated on the same split. The results are compared directly against the naive baseline.

---

## Modeling Approach  
1. **Naive Baseline:** 
2. **Classical ML Approach:**  
3. **Deep Learning Approach:**  
### Data Processing Pipeline  



The CSVs are saved under the following folder structure under `data/processed/`:
```
data/processed/

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

