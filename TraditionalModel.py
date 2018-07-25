import re, sys
import pandas as pd
import numpy as np
import sklearn_crfsuite
from sklearn import linear_model
from sklearn.feature_extraction import DictVectorizer
from nltk.stem import WordNetLemmatizer
import nltk
import copy
nltk.download('wordnet')
LEMMATIZER = WordNetLemmatizer()
START1 = 'START1'
START2 = 'START2'
END1 = 'END1'
END2 = 'END2'


def data_loader(file, isTrain=False):
    f = open(file, 'r')
    sentences = []
    sentence = []
    labels = []
    label = []
    for line in f:
        if len(line) is 1:
            sentences.append(sentence)
            sentence = []
            labels.append(label)
            label = []
        else:
            sentence.append(line.split()[0])

            # get the label if is train/dev
            if isTrain:
                label.append(line.split()[1])

    # extract features
    train_X = extract_feature_crf(sentences)

    return train_X, labels

def get_wordClass(word):
    res = re.match(r'[a-z]+', word)
    if res is not None and res.span()[0] is 0 and res.span()[1] is len(word):
        return 'LOWERCASE'

    res = re.match(r'[A-Z]+', word)
    if res is not None and res.span()[0] is 0 and res.span()[1] is len(word):
        return 'allCaps'

    res = re.match(r'[A-Z][a-z]*', word)
    if res is not None and res.span()[0] is 0 and res.span()[1] is len(word):
        return 'initCap'

    res = re.match(r'[a-z]*[A-Z]+[a-z]*', word)
    if res is not None and res.span()[0] is 0 and res.span()[1] is len(word):
        return 'containCap'

    word = word.lower()
    res = re.match(r'[a-z]*[0-9]+[a-z]*', word)
    if res is not None and res.span()[0] is 0 and res.span()[1] is len(word):
        return 'containsDigit'

    return 'UNK'


def extract_feature_crf(sentences):
    train_features = []
    for sentence in sentences:
        tag_list = [START1, START2] + list(map(lambda x: x[1], nltk.pos_tag(sentence))) + [END1, END2]
        sentence = [START1, START2] + sentence + [END1, END2]

        sen_dict = []
        for i in range(2, len(sentence) - 2):
            feature_dict = {}
            cur_info = word_process(sentence[i], tag_list[i], 'cur', feature_dict)
            prev_info = word_process(sentence[i - 1], tag_list[i - 1], 'prev', feature_dict)
            #             prev_prev_info = word_process(sentence[i-2], tag_list[i-2], 'prev-prev', feature_dict)
            next_info = word_process(sentence[i + 1], tag_list[i + 1], 'next', feature_dict)
            #             next_next_info = word_process(sentence[i+2], tag_list[i+2], 'next-next', feature_dict)

            # more customized feature
            word = sentence[i]
            feature_dict['pre-1'] = word[:1]
            feature_dict['pre-2'] = word[:2]
            feature_dict['pre-3'] = word[:3]
            feature_dict['pre-4'] = word[:4]

            feature_dict['post-1'] = word[-1:]
            feature_dict['post-2'] = word[-2:]
            feature_dict['post-3'] = word[-3:]
            feature_dict['post-4'] = word[-4:]

            sen_dict.append(feature_dict)
        train_features.append(sen_dict)

    return train_features


def extract_feature_log(train_X_crf, labels=None):
    features = []
    for sen in train_X_crf:
        for word in sen:
            temp = []
            for (key, item) in word.items():
                temp.append(item)
            features.append(temp)

    Y = []
    if labels is not None:
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                for k in range(len(labels[i][j])):
                    if labels[i][j][k] == 'O':
                        Y.append(0)
                    else:
                        Y.append(1)

    return features, np.array(Y)


def word_process(word, pos_tag, prefix, feature_dict):
    lemma = LEMMATIZER.lemmatize(word)
    lemma_key = prefix + '-lemma'
    pos_tag_key = prefix + '-pos-tag'
    word_class_key = prefix + '-word-class'

    _dict = {}
    #     _dict[lemma_key] = lemma
    _dict[pos_tag_key] = pos_tag
    _dict[word_class_key] = get_wordClass(word)
    _dict[prefix + '-word'] = word


    # Adding features to prev_word next_word
            #     _dict[prefix+'-len'] = len(word)

            #     _dict[prefix+'-pre-1'] = word[:1]
            #     _dict[prefix+'-pre-2'] = word[:2]
            #     _dict[prefix+'-pre-3'] = word[:3]
            #     _dict[prefix+'-pre-4'] = word[:4]

            #     _dict[prefix+'-post-1'] = word[-1:]
            #     _dict[prefix+'-post-2'] = word[-2:]
            #     _dict[prefix+'-post-3'] = word[-3:]
            #     _dict[prefix+'-post-4'] = word[-4:]
    feature_dict.update(_dict)
    return _dict


def write2file(file_path, predict):
    fh = open(file_path, "w", encoding='utf-8')

    for sen in predict:
        for tag in sen:
            fh.write(tag + '\n')
        fh.write('\n')

    fh.close()


def convert_t0_obi(res):
    arr = ['O', 'B', 'I']
    new_res = []
    for i in range(len(res)):
        if i == 0:
            new_res.append(arr[res[i]])
        else:
            if (new_res[-1] == 'B' or new_res[-1] == 'I') and res[i] == 1:
                new_res.append('I')
            else:
                new_res.append(arr[res[i]])
    return new_res


def convertLog2Crf(predict_log, train_crf):
    output = []
    index = 0

    for i in range(len(train_crf)):
        temp = []
        for j in range(len(train_crf[i])):
            temp.append(predict_log[index])
            index += 1
        output.append(temp)

    return output


def ensemble_union(results):
    result = results[0]
    _len = len(result)

    output = copy.deepcopy(result)

    for r in results:
        assert len(r) == _len

    for k in range(len(results)):
        for i in range(_len):
            for j in range(len(result[i])):
                if results[k][i][j] is not "O" and output[i][j] is "O":
                    output[i][j] = results[k][i][j]

    return output

def logistic_train():
    X_dataframe = pd.DataFrame(np.array(train_X_log + dev_X_log + test_X_log))
    vectorizer = DictVectorizer(sparse=True)
    one_hot_x_training = vectorizer.fit_transform(X_dataframe.to_dict("records"))
    logistic = linear_model.LogisticRegression(C=1e7)
    logistic.fit(one_hot_x_training[0:len(train_X_log)], train_y_log)
    return logistic

def crf_train():
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.05,
        c2=0.005,
        max_iterations=100,
        all_possible_transitions=True,
    )
    crf.fit(train_X_crf, train_y_crf)
    return crf

if __name__=='__main__':
    train_X_crf, train_y_crf = data_loader('data/train/train.txt', isTrain=True)
    dev_X_crf, dev_y_crf = data_loader('data/dev/dev.txt', isTrain=True)
    test_X_crf, _ = data_loader('data/test/test.nolabels.txt', isTrain=False)
    train_X_log, train_y_log = extract_feature_log(train_X_crf, labels=train_y_crf)
    dev_X_log, dev_y_log = extract_feature_log(dev_X_crf, labels=dev_y_crf)
    test_X_log, _ = extract_feature_log(test_X_crf)

