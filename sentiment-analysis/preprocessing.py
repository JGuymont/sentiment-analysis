import csv
import random
from sklearn.feature_extraction.text import CountVectorizer

POS_REVIEW_FILE = './data/rt-polarity_pos.csv'
NEG_REVIEW_FILE = './data/rt-polarity_neg.csv'

def load_csv(path):
    
    out_data = []
    with open(path, 'rb') as csvfile:
        in_data = csv.reader(csvfile, delimiter=' ', quotechar='|')
        for row in in_data:
            out_data.append(' '.join(row))
    return out_data

def split_data(inputs, targets, test_pct):
    train_pct = 1 - test_pct
    data_size = len(targets)
    train_size = round(train_pct*data_size) 
    examples = range(data_size)

    train_index = random.sample(examples, train_size)
    

def vectorizer(ngram_range, stop_words, max_df, min_df):
    """Term frequency vectorizer.

    ngram_range : tuple (min_n, max_n)
        The lower and upper boundary of the range of n-values for different
        n-grams to be extracted. All values of n such that min_n <= n <= max_n
        will be used.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.

        If a list, that list is assumed to contain stop words, all of which
        will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value
        in the range [0.7, 1.0) to automatically detect and filter stop
        words based on intra corpus document frequency of terms.

    max_df : float in range [0.0, 1.0] or int, default=1.0
        When building the vocabulary ignore terms that have a document
        frequency strictly higher than the given threshold (corpus-specific
        stop words).
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int, default=1
        When building the vocabulary ignore terms that have a document
        frequency strictly lower than the given threshold. This value is also
        called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer
        absolute counts.
        This parameter is ignored if vocabulary is not None.
    """

    tf_vectorizer = CountVectorizer(ngram_range, stop_words, max_df, min_df)

    return tf_vectorizer

if __name__ == '__main__':

    POS_REVIEW = load_csv(POS_REVIEW_FILE)
    NEG_REVIEW = load_csv(NEG_REVIEW_FILE)

