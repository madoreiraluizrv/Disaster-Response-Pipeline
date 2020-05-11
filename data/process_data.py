# import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    INPUT
    messages_filepath - filepath for the messages database
    categories_filepath - filepath for the categories database

    OUTPUT
    df - dataframe with merged data from messages and categories databases
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # merge datasets
    df = messages.merge(categories, on=('id'))
    
    return df

def clean_data(df):
    '''
    INPUT
    df - dataframe to be cleaned

    OUTPUT
    df - cleaned dataframe (column names for categories, numeric values in categories columns and duplicates dropped)
    '''
    # create a dataframe of the individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';').values.tolist())
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: str(x)[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df = df.drop(columns=['categories'])
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)
    
    # drop duplicates
    df.drop_duplicates(keep=False, inplace=True)
    
    return df

def save_data(df, database_filename):
    '''
    INPUT
    df - dataframe to be saved
    database_filename - filename to be used to save the database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)  

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()
