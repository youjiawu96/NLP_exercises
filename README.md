# NLP_exercises
A record of python prgramming exercises in NLP class of 2021

**namedentity.py**: Identify DATE, DOLLAR_AMOUNT, TIME, EMAIL_ADDRESS, WEB_ADDRESS with regular expressions.

**naivebayes.py**: Apply naive Bayes method to classify film comments into "positive" or "negative" catogories. Use the "bag of words" as features. Apply Laplace smoothing ('add 1') to count for words which don't appear in the training set. This method achieves 80.7% accuracy on the test dataset provided.

**rnnpos.py**: Apply recurrent neural networks (RNN) to do Parts of Speech (POS) tagging. LSTM model is implemented in this script, and the training is done with PyTorch. Both the training and testing data are the Wall Street Journal data from Penn Treebank. The model trained with 10 epochs, learning rate 0.05 and 100 hidden LSTM units gives out the best accuracy: 90.3%. The other hyperparameters that are not tuned including: batch size 1, SGD optimizer, and 50 dimensional pre-trained GloVe word embedding.
