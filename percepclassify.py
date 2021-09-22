import sys
import os
import re
import numpy as np
import random
from copy import deepcopy

w1 = {}
w2 = {}
b1 = 0
b2 = 0
test_reviews = {}
test_paths = []
output_lines = []


def model_reader(modeler_path):

    global b1
    global b2

    model_params = open(modeler_path, 'r')
    for line in model_params:
        if re.search('\t', line):
            field = line.strip().split('\t')

            token = field[0]

            if token != "B1" and token != "B2":
                w1[token] = float(field[1])
                w2[token] = float(field[2])

            elif token == "B1":
                b1 = float(field[1])
            else:
                b2 = float(field[1])

    model_params.close()


def classifier():
    model = sys.argv[1]
    path = sys.argv[2]
    model_reader(model)
    test_perceptron(path)


def test_perceptron(path):
    for (root, dirs, files) in os.walk(path):
        for file in files:
            if len(str(file)) > 4 and str(file)[-4:] == '.txt':

                review_path = str(root) + "/" + str(file)

                with open(review_path, 'r') as f:
                    tokens = tokenizer(f)
                    f.close()
                    test_reviews[review_path] = deepcopy(tokens)
                    test_paths.append(review_path)

    test_indexer()

    for path in test_paths:

        a1 = b1
        a2 = b2

        for token in test_reviews[path]:
            try:
                a1 += w1[token]
                a2 += w2[token]
            except KeyError:
                pass

        if a1 > 0:
            label_b = "positive"
        else:
            label_b = "negative"

        if a2 > 0:
            label_a = "truthful"
        else:
            label_a = "deceptive"

        output_lines.append(str(label_a) + " " + str(label_b) + " " + path + "\n")

    with open("percepoutput.txt", "w+") as output:
        for lines in output_lines:
            output.write(str(lines))
        output.close()


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


def test_indexer():

    for path in test_paths:
        for i in range(len(test_reviews[path])):
            if len(test_reviews[path][i]) > 1 and test_reviews[path][i][-1] == 's':
                try:
                    w1[test_reviews[path][i][0:-1]]
                    test_reviews[path][i] = test_reviews[path][i][0:-1]
                except KeyError:
                    pass

            if len(test_reviews[path][i]) > 2 and (test_reviews[path][i][-2:] == 've' or 'nt' or 'ly' or 'ed'):
                try:
                    w1[test_reviews[path][i][0:-2]]
                    test_reviews[path][i] = test_reviews[path][i][0:-2]
                except KeyError:
                    pass


if __name__ == "__main__":

    classifier()
