from keras.preprocessing.text import Tokenizer
import numpy as np

f = open("original_data.csv","r")
data = []
for line in f:
    line.strip('"').strip("\n")
    data.append(line.strip().strip('"').strip())

# create tokenizer object
tokenizer = Tokenizer(num_words=None, # max numbers to keep based on freq. Most common num_words-1 words will be kept
                      # filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', # removes all punctuations -- may not learn properly
                      filters='#$%&()*+-<=>@[\\]^_`{|}~\t\n', # keeps punctuations
                      lower=True,
                      split=' ',
                      char_level=False, #every character will be treated as a token if true
                      oov_token=None)

# train the tokenizer to the texts
print("tokenizing %s sentences..." %len(data))
tokenizer.fit_on_texts(data)

# convert list of string into list of lists of integers
seq = tokenizer.texts_to_sequences(data)

# mapping of indexes to words and words to indexes
idx_to_word = tokenizer.index_word #starts from 1
word_to_idx = tokenizer.word_index

features = []
labels = []
training_length = 50
# iterate through the sequences of tokens
print("generting features and labels...")
for s in seq:
    # create multiple training examples from each sequence
    for i in range(training_length, len(s)):
        # extract features and label
        extract = s[i-training_length : i + 1] # len = 51

        # set the features and label
        features.append(extract[:-1])
        labels.append(extract[-1])
features = np.array(features)
print("features shape:", features.shape) #(354403, 50)

# NOTE: neaural network can train most effectively when the labels are one-hot encoded
num_words = len(word_to_idx)+1 #19092 # +1 because idx_to_word starts from 1

# empty array to hold labels
onehot_labels = np.zeros((len(features), num_words), dtype=np.int8) #(354403, 19092)

# one hot encode the labels
for idx, word_indexed in enumerate(labels):
    onehot_labels[idx][word_indexed] = 1
# to find corresponding owrd: idx_to_word[np.argmax(onehot_labels[idx])]



