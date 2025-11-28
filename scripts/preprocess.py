import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/raw.csv')
df = df[['Symptom_1', 'Symptom_2', 'Symptom_3', 'Diagnosis', 'Treatment_Plan']]

def merge_symptoms(row):
    '''
    Merge symptom columns into one string
    '''
    symptoms = [str(row['Symptom_1']), str(row['Symptom_2']), str(row['Symptom_3'])]
    symptoms = [s.strip().lower() for s in symptoms if s.strip().lower() not in ['', 'nan', 'none']]
    return ", ".join(symptoms)

df['symptoms'] = df.apply(merge_symptoms, axis=1) # Merge symptom columns

df = df[df['symptoms'].str.strip() != ""] # Drop rows with empty symptoms
df = df.dropna(subset=['Diagnosis']) # Drop rows with missing diagnosis
df = df.reset_index(drop=True)

diagnoses = sorted(df['Diagnosis'].unique())
label_to_id = {diag: i for i, diag in enumerate(diagnoses)} # Diagnosis to id mapping
id_to_label = {i: diag for diag, i in label_to_id.items()} # id to diagnosis mapping

df['id'] = df['Diagnosis'].map(label_to_id) # Map diagnosis to label ids

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['id'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['id'])

train_df.to_csv('./data/processed/train.csv', index=False)
val_df.to_csv('./data/processed/val.csv', index=False)
test_df.to_csv('./data/processed/test.csv', index=False)