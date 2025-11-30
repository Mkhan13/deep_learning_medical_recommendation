import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

def main():
    train_df = pd.read_csv('./data/processed/train.csv')
    test_df = pd.read_csv('./data/processed/test.csv')

    y_train = train_df['id']  # Extract target variable
    y_test = test_df['id']

    most_common_class = y_train.value_counts().idxmax()  # Find most common class
    y_pred = [most_common_class] * len(y_test)  # Predict most common for all samples

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print('Accuracy:', round(accuracy, 4))
    print('Precision:', round(precision, 4))
    print('Recall:', round(recall, 4))
    print('F1 Score:', round(f1, 4))

if __name__ == '__main__':
    main()