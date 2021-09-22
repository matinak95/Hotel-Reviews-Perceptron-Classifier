import sys
import os
import re
import numpy as np
import random
from copy import deepcopy
import time

file_counter = 0
pos_tru_num = 0
pos_dec_num = 0
neg_tru_num = 0
neg_dec_num = 0
root_path = sys.argv[1]

epoch_number = 100

train_paths = []
test_paths = []

reviews = {}
test_reviews = {}

pos_tru_fold1 = os.path.join(root_path, "positive_polarity", "truthful_from_TripAdvisor", "fold1")
pos_tru_fold2 = os.path.join(root_path, "positive_polarity", "truthful_from_TripAdvisor", "fold2")
pos_tru_fold3 = os.path.join(root_path, "positive_polarity", "truthful_from_TripAdvisor", "fold3")
pos_tru_fold4 = os.path.join(root_path, "positive_polarity", "truthful_from_TripAdvisor", "fold4")

pos_dec_fold1 = os.path.join(root_path, "positive_polarity", "deceptive_from_MTurk", "fold1")
pos_dec_fold2 = os.path.join(root_path, "positive_polarity", "deceptive_from_MTurk", "fold2")
pos_dec_fold3 = os.path.join(root_path, "positive_polarity", "deceptive_from_MTurk", "fold3")
pos_dec_fold4 = os.path.join(root_path, "positive_polarity", "deceptive_from_MTurk", "fold4")

neg_tru_fold1 = os.path.join(root_path, "negative_polarity", "truthful_from_Web", "fold1")
neg_tru_fold2 = os.path.join(root_path, "negative_polarity", "truthful_from_Web", "fold2")
neg_tru_fold3 = os.path.join(root_path, "negative_polarity", "truthful_from_Web", "fold3")
neg_tru_fold4 = os.path.join(root_path, "negative_polarity", "truthful_from_Web", "fold4")

neg_dec_fold1 = os.path.join(root_path, "negative_polarity", "deceptive_from_MTurk", "fold1")
neg_dec_fold2 = os.path.join(root_path, "negative_polarity", "deceptive_from_MTurk", "fold2")
neg_dec_fold3 = os.path.join(root_path, "negative_polarity", "deceptive_from_MTurk", "fold3")
neg_dec_fold4 = os.path.join(root_path, "negative_polarity", "deceptive_from_MTurk", "fold4")

train_pos_tru = [pos_tru_fold2, pos_tru_fold3, pos_tru_fold4]
train_pos_dec = [pos_dec_fold2, pos_dec_fold3, pos_dec_fold4]
train_neg_tru = [neg_tru_fold2, neg_tru_fold3, neg_tru_fold4]
train_neg_dec = [neg_dec_fold2, neg_dec_fold3, neg_dec_fold4]

attributes = {}
attr_prob = {}


def file_reader():
    global file_counter
    global pos_tru_num
    global pos_dec_num
    global neg_tru_num
    global neg_dec_num
    for fold in train_pos_tru:
        for file in os.listdir(fold):
            with open(os.path.join(fold, file), 'r') as f:
                tokens = tokenizer(f)
                f.close()
                reviews[str(os.path.join(fold, file))] = deepcopy(tokens)
                pos_tru_num += 1
                indexer(tokens, 0)
                train_paths.append(str(os.path.join(fold, file)))
    for fold in train_pos_dec:
        for file in os.listdir(fold):
            with open(os.path.join(fold, file), 'r') as f:
                tokens = tokenizer(f)
                f.close()
                reviews[str(os.path.join(fold, file))] = deepcopy(tokens)
                pos_dec_num += 1
                indexer(tokens, 1)
                train_paths.append(str(os.path.join(fold, file)))
    for fold in train_neg_tru:
        for file in os.listdir(fold):
            with open(os.path.join(fold, file), 'r') as f:
                tokens = tokenizer(f)
                f.close()
                reviews[str(os.path.join(fold, file))] = deepcopy(tokens)
                neg_tru_num += 1
                indexer(tokens, 2)
                train_paths.append(str(os.path.join(fold, file)))
    for fold in train_neg_dec:
        for file in os.listdir(fold):
            with open(os.path.join(fold, file), 'r') as f:
                tokens = tokenizer(f)
                f.close()
                reviews[str(os.path.join(fold, file))] = deepcopy(tokens)
                neg_dec_num += 1
                indexer(tokens, 3)
                train_paths.append(str(os.path.join(fold, file)))

    file_counter = pos_tru_num + pos_dec_num + neg_tru_num + neg_dec_num


def tokenizer(review):
    extra1 = ['\n', '\t', '-', '_', '  ', '~', '&', '%', '$', '#', '@', ':', '/', '*']
    extra2 = ['.', ',', ';', '(', ')', '!', '?', '\"', '\'']

    text = review.read()
    text = text.lower()

    for item in extra1:
        text = text.replace(item, ' ')

    for item in extra2:
        text = text.replace(item, '')

    numbers = '[0-9]'

    tokens = text.split(' ')

    try:
        tokens = tokens.remove(' ')
    except ValueError:
        pass

    tokens = [re.sub(numbers, '', i) for i in tokens]
    tokens = [i for i in tokens if len(i) > 2]

    return tokens


def cleaner():

    stop_words = ['were', 'have', 'would', 'each', 'doing', 'travel', 'travelling', 'someone', 'guy', 'room', 'girl',
                  'daughter', 'wont', 'did', 'from', 'without', 'your', 'when', 'where', 'what', 'why', 'was',
                  'one', 'two', 'three', 'who', 'how', 'for', 'using', 'want', 'remind', 'share',
                  'seeing', 'ahead', 'indeed', 'cannot', 'bring', 'anyone',
                  'yourself', 'truly', 'heard', 'mention', 'behind', 'house', 'everywhere', 'waiting',
                  'guest', 'almost', 'throughout', 'family', 'saying', 'above', 'taking',
                  'normal', 'sitting', 'instead', 'somewhere', 'below', 'inside', 'saturday', 'bottom', 'internet',
                  'another', 'either', 'boyfriend', 'anyway', 'thought', 'themselves', 'myself',
                  'across', 'enough', 'along', 'weekend', 'morning', 'watching', 'something', 'bathroom',
                  'traveling', 'getting', 'since', 'opinion', 'taken', 'itself', 'thing', 'staying', 'first', 'again',
                  'through', 'could', 'between', 'everyone', 'everything', 'going', 'because', 'which',
                  'anywhere', 'place', 'being', 'recent', 'bedroom']
    save_short = ['bad', 'worst', 'worse', 'rat', 'not', 'shit', 'damn', 'good', 'well', 'poor', 'cheap', 'worth', 'buy', 'bath', 'dog', 'kind', 'fuck', 'hate', 'fake', 'cost', 'safe', 'warm', 'cool', 'love', 'low', 'high', 'bitch']

    removal = set()
    for i in stop_words:
        removal.add(i)

    for token in attributes:

        if np.sum(attributes[token]) < 3 or np.sum(attributes[token]) > 1 * file_counter:
            removal.add(token)
        if len(token) > 1 and token[-1] == 's':
            try:
                attributes[token[0:-1]] += attributes[token]
                removal.add(token)
            except KeyError:
                pass

        if len(token) > 2 and (token[-2:] == 've' or 'nt' or 'ly' or 'ed'):
            try:
                attributes[token[0:-2]] += attributes[token]
                removal.add(token)
            except KeyError:
                pass

        if len(token) < 5:
            if token not in save_short:
                removal.add(token)

    for item in removal:
        attributes.pop(item)


def indexer(tokens, num):
    for token in tokens:
        try:
            attributes[token][num] += 1
        except KeyError:
            attributes[token] = np.array([0, 0, 0, 0])
            attributes[token][num] += 1


def review_indexer():

    for path in train_paths:
        for i in range(len(reviews[path])):
            if len(reviews[path][i]) > 1 and reviews[path][i][-1] == 's':
                try:
                    attributes[reviews[path][i][0:-1]]
                    reviews[path][i] = reviews[path][i][0:-1]
                except KeyError:
                    pass

            if len(reviews[path][i]) > 2 and (reviews[path][i][-2:] == 've' or 'nt' or 'ly' or 'ed'):
                try:
                    attributes[reviews[path][i][0:-2]]
                    reviews[path][i] = reviews[path][i][0:-2]
                except KeyError:
                    pass


def modeler(w1, b1, w2, b2, w3, b3, w4, b4):
    file = open("vanillamodel.txt", "w+")
    for item in w1:
        file.write(str(item) + '\t' + str(w1[item]) + '\t' + str(w2[item]) + "\n")
    file.write("B1" + '\t' + str(b1) + "\n")
    file.write("B2" + '\t' + str(b2))

    file.close()

    file = open("averagedmodel.txt", "w+")
    for item in w3:
        file.write(str(item) + '\t' + str(w3[item]) + '\t' + str(w4[item]) + "\n")
    file.write("B1" + '\t' + str(b3) + "\n")
    file.write("B2" + '\t' + str(b4))

    file.close()



def class_finder(path):
    chunk = path.split('/')

    tru_dec_large = chunk[-3]
    tru_dec = tru_dec_large.split('_')[0]

    pos_neg_large = chunk[-4]
    pos_neg = pos_neg_large.split('_')[0]

    return pos_neg, tru_dec


def perceptron():
    w1 = {}
    w2 = {}
    u3 = {}
    w3 = {}
    u4 = {}
    w4 = {}

    for item in attributes:
        w1[item] = 0
        w2[item] = 0
        u3[item] = 0
        w3[item] = 0
        u4[item] = 0
        w4[item] = 0



    b1 = np.random.rand()/10000000000
    b2 = np.random.rand()/10000000000
    b3 = np.random.rand()/10000000000
    b4 = np.random.rand() / 10000000000
    beta3 = 0
    beta4 = 0
    sample_count = 1

    for i in range(epoch_number):

        random.shuffle(train_paths)

        for path in train_paths:

            pos_neg, tru_dec = class_finder(path)
            a1 = b1
            a2 = b2
            a3 = b3
            a4 = b4

            for token in reviews[path]:
                try:
                    a1 += w1[token]
                    a2 += w2[token]
                    a3 += w3[token]
                    a4 += w4[token]
                except KeyError:
                    pass

            if pos_neg == 'positive':
                if a1 < 0:
                    for token in reviews[path]:
                        try:
                            w1[token] += 1
                        except KeyError:
                            pass
                    b1 += 1

                if a3 < 0:
                    for token in reviews[path]:
                        try:
                            w3[token] += 1
                            u3[token] += sample_count
                        except KeyError:
                            pass
                    b3 += 1
                    beta3 += sample_count

            else:
                if a1 > 0:
                    for token in reviews[path]:
                        try:
                            w1[token] -= 1
                        except KeyError:
                            pass
                    b1 -= 1

                if a3 > 0:
                    for token in reviews[path]:
                        try:
                            w3[token] -= 1
                            u3[token] -= sample_count
                        except KeyError:
                            pass
                    b3 -= 1
                    beta3 -= sample_count

            if tru_dec == 'truthful':
                if a2 < 0:
                    for token in reviews[path]:
                        try:
                            w2[token] += 1
                        except KeyError:
                            pass
                    b2 += 1

                if a4 < 0:
                    for token in reviews[path]:
                        try:
                            w4[token] += 1
                            u4[token] += sample_count
                        except KeyError:
                            pass
                    b4 += 1
                    beta4 += sample_count

            else:
                if a2 > 0:
                    for token in reviews[path]:
                        try:
                            w2[token] -= 1
                        except KeyError:
                            pass
                    b2 -= 1

                if a4 > 0:
                    for token in reviews[path]:
                        try:
                            w4[token] -= 1
                            u4[token] -= sample_count
                        except KeyError:
                            pass
                    b4 -= 1
                    beta4 -= sample_count
            sample_count += 1

    for token in w3:
        w3[token] = deepcopy(w3[token] - u3[token]/sample_count)
        w4[token] = deepcopy(w4[token] - u4[token] / sample_count)

    b3 -= beta3/sample_count
    b4 -= beta4/sample_count

    return w1, b1, w2, b2, w3, b3, w4, b4


def learner():
    file_reader()
    cleaner()
    review_indexer()
    w1, b1, w2, b2, w3, b3, w4, b4 = perceptron()
    modeler(w1, b1, w2, b2, w3, b3, w4, b4)


# def test(w1, b1, w2, b2):
#     for run in range(10):
#         pos_result = 0
#         truth_result = 0
#
#         for file in os.listdir(neg_dec_fold1):
#             with open(os.path.join(neg_dec_fold1, file), 'r') as f:
#                 tokens = tokenizer(f)
#                 f.close()
#                 test_reviews[str(os.path.join(neg_dec_fold1, file))] = deepcopy(tokens)
#                 test_paths.append(str(os.path.join(neg_dec_fold1, file)))
#         test_indexer(w1)
#
#
#         for path in test_paths:
#
#             a1 = b1
#             a2 = b2
#
#             for token in test_reviews[path]:
#                 try:
#                     a1 += w1[token]
#                     a2 += w2[token]
#                 except KeyError:
#                     pass
#
#             if a1 > 0:
#                 pos_result += 1
#
#             if a2 > 0:
#                 truth_result += 1
#
#     print("Positive:\t" + str(pos_result/10))
#     print("Truthful:\t" + str(truth_result/10))
#
#
#
# def test_indexer(w1):
#
#     for path in test_paths:
#         for i in range(len(test_reviews[path])):
#             if len(test_reviews[path][i]) > 1 and test_reviews[path][i][-1] == 's':
#                 try:
#                     w1[test_reviews[path][i][0:-1]]
#                     test_reviews[path][i] = test_reviews[path][i][0:-1]
#                 except KeyError:
#                     pass
#
#             if len(test_reviews[path][i]) > 2 and (test_reviews[path][i][-2:] == 've' or 'nt' or 'ly' or 'ed'):
#                 try:
#                     w1[test_reviews[path][i][0:-2]]
#                     test_reviews[path][i] = test_reviews[path][i][0:-2]
#                 except KeyError:
#                     pass


if __name__ == "__main__":

    learner()

