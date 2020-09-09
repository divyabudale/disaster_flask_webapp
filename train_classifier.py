import sys
import pandas as pd
import re
from sqlalchemy import create_engine
import pickle
import numpy as np
import nltk

nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

from utils.custom_tokenize import tokenize

def load_data(database_filepath):
    """
    load data from the database

    :param
    database_filepath: the filepath to the database
    :return:
    X: dataframe containing features
    Y: dataframe containing labels

    """
    database_filepath = ''.join(database_filepath)
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('tbl_disaster_response', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    category_names = list(df.columns[4:])
    return X, Y, category_names



# custom transformer for extracting the starting verb of the sentence
class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """Extract the starting verb of a sentence

        """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model(grid_search_cv=False):
    """
    Build the pipeline function

    :param
    grid_search_cv: if True, performs grid search to optimize the hyper parameters of a model
    :return:
    pipeline: pipeline that process the text data
    """
    # pipeline = Pipeline([
    #     ('features', FeatureUnion([
    #         ('text_pipeline', Pipeline([
    #             ('vect', CountVectorizer(tokenizer=tokenize)),
    #             ('tfidf', TfidfTransformer())])),
    #         ('starting_verb', StartingVerbExtractor())])),
    #     ('clf', MultiOutputClassifier(AdaBoostClassifier()))])

    pipeline = Pipeline([
        ('countvectorizer', CountVectorizer(tokenizer=tokenize)),
        ('tfidftransformer', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier(), n_jobs=1))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100, 150]
                  }
    if grid_search_cv == True:
        pipeline = GridSearchCV(pipeline, parameters)

    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the model performance by f1 score

    :param
    model: a machine learning model
    X_test: features from the test dataframe
    Y_test: labels from the test dataframe
    category_names: label names
    :return:
    None
    """
    Y_pred = model.predict(X_test)
    for i in range(len(category_names)):
        print('Category: {} '.format(category_names[i]))
        print(classification_report(Y_test.iloc[:, i].values, Y_pred[:, i]))
        print('Accuracy {}\n\n'.format(accuracy_score(Y_test.iloc[:, i].values, Y_pred[:, i])))


def save_model(model, model_filepath):
    """save the model in the model_filepath
    :param
    model: a machine learning model
    model_filepath: the filepath where the model will be saved
    :return
    None

    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    """
    Main function that train the classifier
    Parameters:
    arg1: the file path of the database
    arg2: the file path that the trained model will be saved

    :return:
    """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model(grid_search_cv=False)

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database ' \
              'as the first argument and the filepath of the pickle file to ' \
              'save the model to as the second argument. \n\nExample: python ' \
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
