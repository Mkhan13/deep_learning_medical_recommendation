import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

def main():
    train_df = pd.read_csv('./data/processed/train.csv')
    val_df = pd.read_csv('./data/processed/val.csv')
    test_df = pd.read_csv('./data/processed/test.csv')

    # Features and Labels
    X_train = train_df['symptoms']
    y_train = train_df['id']

    X_test = test_df['symptoms']
    y_test = test_df['id']

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)

    X_train = vectorizer.fit_transform(train_df['symptoms'])
    X_test = vectorizer.transform(test_df['symptoms'])

    model = LogisticRegression(max_iter=400, multi_class='ovr')
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    print(f'Accuracy: {round(accuracy, 4)}')
    print(f'Precision: {round(precision, 4)}')
    print(f'Recall: {round(recall, 4)}')
    print(f'F1-score: {round(f1, 4)}')

if __name__ == '__main__':
    main()
