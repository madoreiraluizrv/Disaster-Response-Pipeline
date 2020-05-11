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
    '''
    INPUT
    datanbase_filepath - filepath for the database

    OUTPUT
    X - input for the model (messages)
    Y - output for the model (categories)
    Y.columns - list of categories names
    '''
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
    '''
    INPUT
    text - text to be tokenized

    OUTPUT
    clean_tokens - list of tokens lemmatized and normalized
    '''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

def build_model():
    '''
    OUTPUT
    cv - machine learning model to be used
    '''
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
            'features__count_pipeline__tfidf__use_idf': (True, False),
            'features__transformer_weights': (
                {'count_pipeline': 1, 'hashing_pipeline': 0.5},
                {'count_pipeline': 0.5, 'hashing_pipeline': 1},
            )
        }

    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT
    model - machine learning model
    X_test - input array for test
    Y_test - true output for test
    category_names - list of category names
    '''
    Y_pred = model.predict(X_test)
    Y_pred = pd.DataFrame(Y_pred, columns=Y_test.columns)

    for category in category_names:
        print(category + ":")
        print(classification_report(Y_test[category], Y_pred[category]))

def save_model(model, model_filepath):
    '''
    INPUT
    model - machine learning model
    model_filepath - filepath for where to save the model
    '''
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
