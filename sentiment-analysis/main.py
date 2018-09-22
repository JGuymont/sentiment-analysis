from time import time
from utils import load_pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_PATH = './data/data.pkl'

SVM_PARAMETERS = {
    # 'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'vect__stop_words': (None, 'english'),
    'clf__penalty': ('l2',),
    'clf__tol': (0.0001,),
    # 'clf__C': (0.05, 0.10, 0.15, 0.25,),
    # 'clf__loss': ('squared_hinge', 'hinge')
}

if __name__ == '__main__':

    data = load_pickle(DATA_PATH)
    train_x, train_y, test_x, test_y = data['train_x'], data['train_y'], data['test_x'], data['test_y']

    clf = LinearSVC()

    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('clf', clf),
    ])

    parameters = SVM_PARAMETERS
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    print(parameters) 
    t0 = time()
    grid_search.fit(train_x, train_y)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print(grid_search.cv_results_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))


    