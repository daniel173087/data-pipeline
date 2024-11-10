import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load messages and categories datasets and merge them."""
    # Load messages dataset
    messages = pd.read_csv(messages_filepath)
    
    # Load categories dataset
    categories = pd.read_csv(categories_filepath)
    
    # Merge datasets on 'id'
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """Clean the merged dataframe by splitting categories and removing duplicates."""
    # Split 'categories' into separate columns
    categories = df['categories'].str.split(';', expand=True)
    
    # Extract category names from the first row
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    
    # Convert category values to binary (0 or 1)
    for column in categories:
        # Set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        
        # Convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original 'categories' column from `df`
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate original `df` with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    """Save the clean dataset into an SQLite database."""
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('df', engine, index=False, if_exists='replace')


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