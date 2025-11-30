import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from huggingface_hub import login, HfApi
from dotenv import load_dotenv
import os
import warnings

warnings.filterwarnings('ignore')

class SymptomDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.tolist()
        self.labels = labels.tolist()

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = tokenizer(
            self.texts[idx],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )

        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def compute_metrics(pred):
    logits = pred.predictions
    preds = logits.argmax(axis=1)
    labels = pred.label_ids

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def push_model_to_hf(model, tokenizer, repo_id: str, output_dir: str = 'bert_model'):
    '''
    Push the trained model and tokenizer to the Hugging Face Hub.
    '''
    load_dotenv()
    login(token=os.getenv('HUGGINGFACE_TOKEN'))

    api = HfApi()
    api.create_repo(repo_id, repo_type='model', exist_ok=True)

    # Save locally
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Upload folder
    api.upload_folder(folder_path=output_dir, repo_id=repo_id)

    print(f'Model pushed to: https://huggingface.co/{repo_id}')

def main():
    global tokenizer

    train_df = pd.read_csv('./data/processed/train.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')

    num_labels = train_df['id'].nunique()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = SymptomDataset(train_df['symptoms'], train_df['id'])
    val_dataset = SymptomDataset(val_df['symptoms'], val_df['id'])
    test_dataset = SymptomDataset(test_df['symptoms'], test_df['id'])

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_labels)

    training_args = TrainingArguments(
        output_dir='./model_output',
        eval_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=50,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        learning_rate=2e-5,
        weight_decay=0.01,
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    test_results = trainer.evaluate(test_dataset)
    print(test_results)

    push_model_to_hf(model=model, tokenizer=tokenizer, repo_id='moosejuice13/bert_medical_diagnosis_classifier', output_dir='bert_model')

if __name__ == '__main__':
    main()