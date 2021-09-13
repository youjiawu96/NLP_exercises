import os
import re
import numpy as np
# For Programming Problem 2, we will implement a naive Bayesian sentiment classifier which learns to classify movie reviews as positive or negative.
#
# Please implement the following functions below: train(), predict(), evaluate(). Feel free to use any Python libraries such as sklearn, numpy, etc. 
# DO NOT modify any function definitions or return types, as we will use these to grade your work. However, feel free to add new functions to the file to avoid redundant code (e.g., for preprocessing data).
#
# *** Don't forget to additionally submit a README_2 file as described in the assignment. ***


# Description: Trains the naive Bayes classifier.
# Inputs: String for the file location of the training data (the "training" directory).
# Outputs: An object representing the trained model.
def train(training_path):
    # count the frequency of words in both pos and neg training sets
    vocab_counts_pos = {}
    vocab_counts_neg = {}
    neg_sub_path = '/neg/'
    pos_sub_path = '/pos/'
    # total number of words
    num_words_neg = 0
    num_words_pos = 0
    # deal with negative training set
    for path in os.listdir(training_path+neg_sub_path):
        with open(training_path+neg_sub_path+path, 'rb') as f:
            content = f.read()
        content = str(content)
        # split by all non-letter characters, labelled by r'\W+' as regular expression
        cont_words = re.split(r'\W+', content)

        for word in cont_words:
            num_words_neg += 1
            if word not in vocab_counts_neg:
                vocab_counts_neg[word] = 1
            else:
                vocab_counts_neg[word] += 1
    # sort the dictionary by frequency of words, from most frequent to least
    vocab_counts_neg = dict(sorted(vocab_counts_neg.items(), key=lambda x:x[1], reverse=True))

    # deal with positive training set
    for path in os.listdir(training_path+pos_sub_path):
        with open(training_path+pos_sub_path+path, 'rb') as f:
            content = f.read()
        content = str(content)
        cont_words = re.split(r'\W+', content)
        
        for word in cont_words:
            num_words_pos += 1
            if word not in vocab_counts_pos:
                vocab_counts_pos[word] = 1
            else:
                vocab_counts_pos[word] += 1
    # same as above
    vocab_counts_pos = dict(sorted(vocab_counts_pos.items(), key=lambda x:x[1], reverse=True))
    
    # normalize the counts with Laplace normalization: add one count to each appearance of the word
    for w in vocab_counts_neg:
        vocab_counts_neg[w] = (vocab_counts_neg[w]+1)/(num_words_neg + len(vocab_counts_neg))
    for w in vocab_counts_pos:
        vocab_counts_pos[w] = (vocab_counts_pos[w]+1)/(num_words_pos + len(vocab_counts_pos))
    
    trained_model = (vocab_counts_neg, vocab_counts_pos, num_words_neg, num_words_pos)
    return trained_model


# Description: Runs prediction of the trained naive Bayes classifier on the test set, and returns these predictions.
# Inputs: An object representing the trained model (whatever is returned by the above function), and a string for the file location of the test data (the "testing" directory).
# Outputs: An object representing the predictions of the trained model on the testing data, and an object representing the ground truth labels of the testing data.
def predict(trained_model, testing_path):
    neg_probs, pos_probs, neg_num, pos_num = trained_model
    neg_sub_path = '/neg/'
    pos_sub_path = '/pos/'
    # arrays to save predictions and ground truth, 1 for positive, -1 for negative
    model_predictions = []
    ground_truth = []

    for path in os.listdir(testing_path+pos_sub_path):
        with open(testing_path+pos_sub_path+path, 'rb') as f:
            content = f.read()
        content = str(content)
        cont_words = re.split(r'\W+', content)
        # log of positive ang negative probabilities
        log_P_pos = 0.0
        log_P_neg = 0.0
        for word in cont_words:
            # calculate log of probability that this content is positive
            if word in pos_probs:
                log_P_pos += np.log10(pos_probs[word])
            else:
                # if the word doesn't appear in the training set, use the default value for probability
                log_P_pos += np.log10(1.0/(pos_num+len(pos_probs)))

            # calculate log of probability that this content is negative
            if word in neg_probs:
                log_P_neg += np.log10(neg_probs[word])
            else:
                log_P_neg += np.log10(1.0/(neg_num+len(neg_probs)))
        # determine whether it's positive or negative according to relative size of two probabilities
        if log_P_pos > log_P_neg:
            model_predictions.append(1)
        else:
            model_predictions.append(-1)
        
        ground_truth.append(1)

    for path in os.listdir(testing_path+neg_sub_path):
        with open(testing_path+neg_sub_path+path, 'rb') as f:
            content = f.read()
        content = str(content)
        cont_words = re.split(r'\W+', content)
        # log of positive ang negative probabilities
        log_P_pos = 0.0
        log_P_neg = 0.0
        for word in cont_words:
            # calculate log of probability that this content is positive
            if word in pos_probs:
                log_P_pos += np.log10(pos_probs[word])
            else:
                log_P_pos += np.log10(1.0/(pos_num+len(pos_probs)))

            # calculate log of probability that this content is negative
            if word in neg_probs:
                log_P_neg += np.log10(neg_probs[word])
            else:
                log_P_neg += np.log10(1.0/(neg_num+len(neg_probs)))
        # determine whether it's positive or negative according to relative size of two probabilities
        if log_P_pos > log_P_neg:
            model_predictions.append(1)
        else:
            model_predictions.append(-1)
        
        ground_truth.append(-1)

    return model_predictions, ground_truth


# Description: Evaluates the accuracy of model predictions using the ground truth labels.
# Inputs: An object representing the predictions of the trained model, and an object representing the ground truth labels for the testing data.
# Outputs: Floating-point accuracy of the trained model on the test set.
def evaluate(model_predictions, ground_truth):
    correct_count = 0
    for i in range(len(model_predictions)):
        if model_predictions[i] == ground_truth[i]:
            correct_count += 1
    
    accuracy = correct_count/len(ground_truth)
    return accuracy


# GRADING: We will be using lines like these to run your functions (from a separate file). You can run the file naivebayes.py in the command line (e.g., "python naivebayes.py") to verify that your code works as expected for grading.
TRAINING_PATH='./HW1_data/training' # TODO: replace with your path
TESTING_PATH='./HW1_data/testing' # TODO: replace with your path

trained_model = train(TRAINING_PATH)
model_predictions, ground_truth = predict(trained_model, TESTING_PATH)
accuracy = evaluate(model_predictions, ground_truth)
print('Accuracy: %s' % str(accuracy))
