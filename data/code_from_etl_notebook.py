import pandas as pd
from sqlalchemy import create_engine

# load messages dataset
messages = pd.read_csv('messages.csv')
messages.head()

# load categories dataset
categories = pd.read_csv('categories.csv')
categories_id = categories['id']

categories.head()

# merge datasets
df = pd.merge(messages, categories, on='id')
df.head()

# create a dataframe of the 36 individual category columns
categories_new = categories['categories'].str.split(';', expand=True)
categories_new.head()

# select the first row of the categories dataframe
row = categories_new.iloc[0]


# use this row to extract a list of new column names for categories.
# one way is to apply a lambda function that takes everything 
# up to the second to last character of each string with slicing
category_colnames = row.apply(lambda x: x.split('-')[0])
print(category_colnames)

# rename the columns of `categories`
categories_new.columns = category_colnames
categories_new.head()

for column in categories_new:
    # set each value to be the last character of the string
    categories_new[column] = categories_new[column].astype(str).str[-1]
    
    # convert column from string to numeric
    categories_new[column] = pd.to_numeric(categories_new[column])
categories_new.head()

# drop the original categories column from `df`
df.drop('categories', axis=1, inplace=True)

df.head()

# concatenate the original dataframe with the new `categories` dataframe
categories_final = pd.concat([categories_id, categories_new], axis=1)
df = pd.merge(messages, categories_final, on='id')
df.head()

# check number of duplicates
duplicates_before = df.duplicated().sum()
print(f'duplicates before removing {duplicates_before}')

# drop duplicates
df = df.drop_duplicates()

# check number of duplicates
duplicates_after = df.duplicated().sum()
print(f'duplicates after removing {duplicates_after}')

engine = create_engine('sqlite:///merged_data.db')
df.to_sql('df', engine, index=False)