import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    extract messages and categories dataset from csv file and merge them into one dataset
    :param
    messages_filepath: the file path for messages dataset
    categories_filepath: the file path for categories dataset
    :return:
    df: merged dataset containing messages and categories
    """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='right')
    return df


def clean_data(df):
    """
    Apply transformation to the categories values in the dataframe

    :param
    df: merged messages and categories raw dataframe
    :return:
    df: cleaned data frame
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    category_colnames = row.str.split('-', expand=True)[0].tolist()

    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)

    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], join='inner', axis=1)

    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df


def save_data(df, database_filename):
    """
    Load the transformed data frame into a database table

    :param
    df: cleaned dataframe
    database_filename: the path to the SQLite database

    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('tbl_disaster_response', engine, index=False, if_exists='replace')


def main():
    """
    Main functions that process the dataset. The three primary actions performed by the function:
    1. extract the messages and categories dataset and merge them into one dataset
    2. apply transformation and cleaning the merged dataset
    3. load the cleaned dataset into SQLite database

    """
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
        print('Please provide the filepaths of the messages and categories ' \
              'datasets as the first and second argument respectively, as ' \
              'well as the filepath of the database to save the cleaned data ' \
              'to as the third argument. \n\nExample: python process_data.py ' \
              'disaster_messages.csv disaster_categories.csv ' \
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
