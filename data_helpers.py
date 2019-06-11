import re
import numpy as np
from config import cfg
from tensorflow.contrib import learn
from keras.datasets import imdb
from keras.preprocessing import sequence
from mlxtend.preprocessing import one_hot


def load_data(dataset, root="./datasets"):
    if dataset == 'IMDB':
        return load_imdb()
    elif dataset == 'ProcCons':
        return load_pr(root + '/ProcCons/ProcCons/IntegratedPros.txt', root + '/ProcCons/ProcCons/IntegratedCons.txt')
    elif dataset == 'MR':
        return load_mr(root + '/MR/MR/rt-polarity.pos', root + '/MR/MR/rt-polarity.neg')
    elif dataset == 'SST-1':
        return load_sst1(root + '/SST-1/train.csv', root + '/SST-1/dev.csv', root + '/SST-1/test.csv')
    elif dataset == 'SST-2':
        return load_sst2(root + '/SST-2/train.csv', root + '/SST-2/dev.csv', root + '/SST-2/test.csv')
    elif dataset == 'SUBJ':
        return load_subj(root + '/SUBJ/Subj/plot.tok.gt9.5000', root + '/SUBJ/Subj/quote.tok.gt9.5000')
    elif dataset == 'TREC':
        return load_trec(root + '/TREC/TREC/train_5500.label.txt', root + '/TREC/TREC/TREC_10.label.txt')
    else:
        raise Exception('Invalid dataset, please check the name of dataset:', dataset)


def load_imdb():
    (x_train, y_train), (x_test, y_test) = imdb.load_data(path='imdb.npz',num_words=cfg.max_features)

    x_train = sequence.pad_sequences(x_train, maxlen=cfg.max_len, padding='post')
    x_test = sequence.pad_sequences(x_test, maxlen=cfg.max_len, padding='post')

    y_train = [[1, 0] if y == 0 else [0, 1] for y in y_train]
    y_test = [[1, 0] if y == 0 else [0, 1] for y in y_train]

    X = np.concatenate((np.array(x_train), np.array(x_test)))
    Y = np.concatenate((np.array(y_train), np.array(y_test)))

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    vocab_size = cfg.max_features
    max_len = cfg.max_len

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), vocab_size, max_len


def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string.strip('\"')
    string.strip('\'')
    return string.strip().lower()


def load_pr(pos, neg):
    positive_examples = list(open(pos).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(x_text)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length


def load_mr(pos, neg):
    positive_examples = list(open(pos).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(x_text)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length


def load_sst1(train, dev, test):
    x_train = list()
    y_train = list()

    for line in [line.split(",", 1) for line in open(train).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    for line in [line.split(",", 1) for line in open(dev).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    for line in [line.split(",", 1) for line in open(test).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    # Generate labels
    X = x_train
    Y = one_hot(y_train, dtype='int')

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in X])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(X)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length


def load_sst2(train, dev, test):
    x_train = list()
    y_train = list()

    for line in [line.split(",", 1) for line in open(train).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    for line in [line.split(",", 1) for line in open(dev).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    for line in [line.split(",", 1) for line in open(test).readlines()]:
        y_train.append(int(line[0])-1)
        x_train.append(clean_str(line[1]))

    # Generate labels
    X = x_train
    Y = one_hot(y_train, dtype='int')

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in X])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(X)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length


def load_subj(pos, neg):
    positive_examples = list(open(pos).readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(neg).readlines())
    negative_examples = [s.strip() for s in negative_examples]

    # Split by words
    x_text = negative_examples + positive_examples
    x_text = [clean_str(sent) for sent in x_text]

    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    negative_labels = [[1, 0] for _ in negative_examples]
    Y = np.concatenate([negative_labels, positive_labels], 0)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(x_text)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length


def load_trec(dev, test):
    categories = {"ABBR":0, "ENTY":1, "DESC":2, "HUM":3, "LOC":4, "NUM":5}

    x_train = list()
    y_train = list()

    for line in [line.split(" ", 1) for line in open(dev).readlines()]:
        i = line[0].split(":")
        y_train.append(categories[i[0]])
        x_train.append(clean_str(line[1]))

    for line in [line.split(" ", 1) for line in open(test).readlines()]:
        i = line[0].split(":")
        y_train.append(categories[i[0]])
        x_train.append(clean_str(line[1]))

    # Generate labels
    X = x_train
    Y = one_hot(y_train, dtype='int')

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in X])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X = np.array(list(vocab_processor.fit_transform(X)))
    X = sequence.pad_sequences(X, maxlen=max_document_length, padding='post')

    X_TRAIN = X[:len(X)*9/10]
    Y_TRAIN = Y[:len(Y)*9/10]

    X_DEV = X[len(X)*9/10:len(X)*95/100]
    Y_DEV = Y[len(Y)*9/10:len(Y)*95/100]

    X_TEST = X[len(X)*95/100:]
    Y_TEST = Y[len(Y)*95/100:]

    return (X_TRAIN, Y_TRAIN), (X_DEV, Y_DEV), (X_TEST, Y_TEST), len(vocab_processor.vocabulary_) + 1, max_document_length
