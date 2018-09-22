import random
from utils import load_csv, load_pickle, save_data

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

if __name__ == '__main__':

    POS_REVIEW_FILE = './data/rt-polarity_pos.csv'
    NEG_REVIEW_FILE = './data/rt-polarity_neg.csv'

    POS_REVIEW = load_csv(POS_REVIEW_FILE)
    NEG_REVIEW = load_csv(NEG_REVIEW_FILE)

    REVIEWS = POS_REVIEW + NEG_REVIEW
    POLARITIES = [1]*len(POS_REVIEW) + [0]*len(NEG_REVIEW)

    DATA = split_data(REVIEWS, POLARITIES, test_pct=0.15)
    save_data(DATA, path='./data/data.pkl')