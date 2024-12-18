import sys
import pandas as pd
import joblib
from sqlalchemy import create_engine
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

def load_data(database_filepath):
    """
    Load data from SQLite database.

    Input:
        database_filepath: filepath to SQLite database containing cleaned data.

    Output:
        X: a dataframe; Features dataset.
        Y: a dataframe; Labels dataset.
        category_names: List of label names.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('df', engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = Y.columns.tolist()
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize text data.

    Input:
        text: Text data to be tokenized.

    Output:
        lemmatized_tokens: List of cleaned and lemmatized tokens.
    """
    lemmatizer = WordNetLemmatizer()

    # Normalize to lowercase and remove punctuation
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text and remove stop words
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]

    return lemmatized_tokens

def build_model():
    """
    Build a machine learning pipeline and perform grid search.

    Input:
    None

    Output:
        cv: GridSearchCV object. Grid search model object.
    """
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Define parameters for GridSearchCV
    parameters = {
        'vect__max_df': [0.75],
        'vect__ngram_range': [(1, 1), (1, 2)],
        'clf__estimator__n_estimators': [50],
        'clf__estimator__min_samples_split': [2, 4]
    }

    # Create GridSearchCV object
    cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3, n_jobs=1)
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model on a test set.

    Input:
        model: Trained model
        X_test: dataframe. Test features
        Y_test: dataframe. True labels for test set
        category_names: list. List of category names

    Output:
        None
    """
    Y_pred = model.predict(X_test)
    for i, column in enumerate(category_names):
        print(f"Metrics for {column}:")
        accuracy = accuracy_score(Y_test[column], Y_pred[:, i])
        precision = precision_score(Y_test[column], Y_pred[:, i], average='weighted', zero_division=0)
        recall = recall_score(Y_test[column], Y_pred[:, i], average='weighted', zero_division=0)
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(classification_report(Y_test[column], Y_pred[:, i]))

def save_model(model, model_filepath):
    """
    Save the model as a pickle file.

    Input:
        model: Trained model to be saved
        model_filepath: Filepath to save the model

    Output:
        None
    """
    joblib.dump(model, model_filepath)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()