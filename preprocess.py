import pandas as pd
from sklearn.model_selection import train_test_split
import re
import os

RAW_PATH = './data/raw.csv'
PROCESSED_DIR = './data/processed/'
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(RAW_PATH, encoding='latin1')  # Load dataset
df = df[['symptoms','medical_condition','drug_name','side_effects','rx_otc','alcohol','age_group','gender','body_part']]
df = df.dropna(subset=['symptoms', 'medical_condition']).reset_index(drop=True)

counts = df['medical_condition'].value_counts()
df = df[df['medical_condition'].isin(counts[counts >= 2].index)] # Remove classes with only 1 sample

def clean_text(text):
    text = str(text).lower()
    text = text.replace('\n', ' ').replace('\r', ' ')  # newlines
    text = re.sub(r'[^a-z0-9, ]+', '', text)  # punctuation except commas
    text = re.sub(r'\s+', ' ', text).strip()  # normalize whitespace
    return text

text_cols = ['symptoms', 'drug_name', 'side_effects', 'body_part']
for col in text_cols:
    df[col] = df[col].apply(clean_text)

# Stratified split
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['medical_condition'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Save processed files
train_df.to_csv(f'{PROCESSED_DIR}train.csv', index=False)
val_df.to_csv(f'{PROCESSED_DIR}val.csv', index=False)
test_df.to_csv(f'{PROCESSED_DIR}test.csv', index=False)