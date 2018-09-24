import csv
import pickle
import codecs
import random
from time import time
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

random.seed(1)

MODEL = 'lr'

POS_REVIEW_FILE = './data/rt-polarity_pos.csv'
NEG_REVIEW_FILE = './data/rt-polarity_neg.csv'
ENGLISH_STOPWORDS = set(stopwords.words('english')) 
TEST_SIZE = 0.3

MAX_DF_RANGE = (0.2, 0.3, 0.4, 0.5, 1.0)
MIN_DF_RANGE = (1, 0.0001, 0.001)
NGRAM_RANGE = ((1, 1), (1, 2))
STOPWORDS_RANGE = (ENGLISH_STOPWORDS, None)
C_RANGE = (0.25, 0.5, 1., 2, 3, 5)

SVM_PARAMETERS = {
    'vect__max_df': MAX_DF_RANGE,
    'vect__min_df': MIN_DF_RANGE,
    'vect__ngram_range': NGRAM_RANGE,  
    'vect__stop_words': STOPWORDS_RANGE,
    'clf__C': C_RANGE, 
    'clf__loss': ('squared_hinge',)
}

NB_PARAMETERS = {
    'vect__max_df': MAX_DF_RANGE,
    'vect__min_df': MIN_DF_RANGE,
    'vect__ngram_range': NGRAM_RANGE,  
    'vect__stop_words': STOPWORDS_RANGE,
    'clf__alpha': (0.5, 0.75, 1., 1.5, 2),
}

LR_PARAMETERS = {
    'vect__max_df': MAX_DF_RANGE,
    'vect__min_df': MIN_DF_RANGE,
    'vect__ngram_range': NGRAM_RANGE,  
    'vect__stop_words': STOPWORDS_RANGE,
    'clf__C': C_RANGE,
}

class Utils:
    @staticmethod
    def load_csv(path):
        out_data = []
        with open(path, 'r', encoding='cp1252') as csvfile:
            in_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
            for row in in_data:
                out_data.append(' '.join(row))
        return out_data

    @staticmethod
    def split_data(inputs, targets, test_pct):
        train_pct = 1 - test_pct
        data_size = len(targets)
        train_size = int(round(train_pct*data_size)) 
        
        examples = range(data_size)

        train_index = random.sample(examples, train_size)
        test_index = [example for example in examples if example not in train_index]

        train_x = [inputs[i] for i in train_index]
        train_y = [targets[i] for i in train_index]

        test_x = [inputs[i] for i in test_index]
        test_y = [targets[i] for i in test_index]

        data = {
            'train_x': train_x,
            'train_y': train_y,
            'test_x': test_x,
            'test_y': test_y
        }

        return data

def evaluate(pred, test_y):
    test_size = len(test_y)

    correct = 0
    for i in range(test_size):
        if pred[i] == test_y[i]:
            correct += 1
    return round(correct/test_size, 4)

def run_experiment(inputs, targets, test_pct, model):
    """Run 
    
    """

    data = Utils.split_data(inputs, targets, test_pct)
    train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']

    if model == 'svm':
        clf = LinearSVC()
        parameters = SVM_PARAMETERS
    elif model == 'nb':
        clf = BernoulliNB()
        parameters = NB_PARAMETERS
    elif model == 'lr':
        clf = LogisticRegression()
        parameters = LR_PARAMETERS
    
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', clf),
    ])

    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3, refit=True, scoring='accuracy')
    t0 = time()
    grid_search.fit(train_x, train_y)
    print("done in %0.3fs" % (time() - t0))
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    train_pred = grid_search.predict(train_x)
    train_acc = evaluate(train_pred, train_y) 
    
    test_pred = grid_search.predict(test_x)
    test_acc = evaluate(test_pred, test_y)

    print(' Best train accuracy:', train_acc)
    print(' Best test accuracy:', test_acc)

    cm = confusion_matrix(test_y, test_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(cm)

    return 

if __name__ == '__main__':

    pos_review = Utils.load_csv(POS_REVIEW_FILE)
    neg_review = Utils.load_csv(NEG_REVIEW_FILE)

    reviews = pos_review + neg_review
    polarities = [1]*len(pos_review) + [0]*len(neg_review)

    corpus = [w for x in reviews for w in x.split()]

    run_experiment(reviews, polarities, TEST_SIZE, MODEL)
    

    

