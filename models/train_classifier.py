# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import re
import numpy as np
import pandas as pd
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sqlalchemy import create_engine

def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)

    for column in df.columns:
        if (column != 'id') and (column != 'message') and (column != 'original') and (column != 'genre'):
            df = df[(df[column] == 0) | (df[column] == 1)]

    X = df['message'].values
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    
    return X, Y, Y.columns

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('count_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
            ('hashing_pipeline', Pipeline([
                ('hash', HashingVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
        ('clf', MultiOutputClassifier(ExtraTreesClassifier()))
    ])

    parameters = {
            'features__count_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
            'features__count_pipeline__vect__max_df': (0.5, 0.75, 1.0)
        }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for category in category_names:
        print(category + ":")
        print(classification_report(Y_test[category], Y_pred[category]))

def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))

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
