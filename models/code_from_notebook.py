# import libraries
import pandas as pd
from sqlalchemy import create_engine

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt_tab')
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import joblib


from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# load data from database
engine = create_engine('sqlite:///merged_data.db')
df = pd.read_sql_table('df', engine)
X = df['message']
Y = df.iloc[:, 4:]

lemmatizer = WordNetLemmatizer()

def tokenize(text):
    # Normalize to lowercase
    text = text.lower()

    # Remove punctuation and non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)

    # Tokenize text
    tokens = word_tokenize(text)

    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stopwords.words('english')]

    return lemmatized_tokens

pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),  # Tokenize and count word occurrences
    ('tfidf', TfidfTransformer()),  # Transform word counts to TF-IDF features
    ('clf', MultiOutputClassifier(RandomForestClassifier()))  # Classify with RandomForest within MultiOutputClassifier
])

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
pipeline.fit(X_train, Y_train)

Y_pred = pipeline.predict(X_test)


#test your model
for i, column in enumerate(Y.columns):
    print(f"Category: {column}")
    print(classification_report(Y_test.iloc[:, i], Y_pred[:, i]))
    print("\n")


#improve your model
parameters = {
    'vect__max_df': [0.75],
    'vect__ngram_range': [(1, 1), (1, 2)],
    'clf__estimator__n_estimators': [50],
    'clf__estimator__min_samples_split': [2, 4]
}

X_sample, _, Y_sample, _ = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, verbose=3, n_jobs=1)
cv.fit(X_sample, Y_sample)

print(f"Best parameters found: {cv.best_params_}")

Y_pred = cv.predict(X_test)

#test your improved model
for i, column in enumerate(Y_test.columns):
    accuracy = accuracy_score(Y_test[column], Y_pred[:, i])
    precision = precision_score(Y_test[column], Y_pred[:, i], average='weighted')
    recall = recall_score(Y_test[column], Y_pred[:, i], average='weighted')
    print(f"Metrics for {column}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(classification_report(Y_test[column], Y_pred[:, i]))

#save your model
joblib.dump(cv, 'optimized_model.pkl')
