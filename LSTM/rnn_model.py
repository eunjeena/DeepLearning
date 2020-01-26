'''
Recurrent Neural Network:
Give the network a sequence of words and train it to predict the next word;
(1) Clean data using regular expressions
(2) Prepare data for neural network
    - convert text to integers (tokenization)
    - encode labels using one-hot encoding
    - build training and validation set
(3) Build a recurrent neural network using LSTM cells
(4) Use pre-trained word embedding (we can train our own embeddings)
(5) Adjust model paramaeters to improve performance
(6) Inspect model results
'''
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Bidirectional
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from keras.utils import plot_model

from keras.optimizers import Adam


RANDOM_STATE = 50
EPOCHS = 150
BATCH_SIZE = 2048
TRAINING_LENGTH = 50
TRAIN_FRACTION = 0.7
LSTM_CELLS = 64
VERBOSE = 0
SAVE_MODEL = True

# 1. DATA PREPARATION
'''
Raw data is availble on https://www.patentsview.org/query/
by searching 'neural network' and selecting 'Patent Title'
'''
print("\n[STEP1] Data Preparation")
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
print("  tokenizing %s sentences..." %len(data))
tokenizer.fit_on_texts(data)

# convert list of string into list of lists of integers
seq = tokenizer.texts_to_sequences(data)

# mapping of indexes to words and words to indexes
idx_to_word = tokenizer.index_word #starts from index 1
word_to_idx = tokenizer.word_index
print("  >> there are %s unique words" %len(idx_to_word))

word_counts = tokenizer.word_counts
print("  The top 15 most common words are:")
print(sorted(word_counts.items(), key=lambda x:x[1], reverse=True)[:15])

# 2. FEATURES AND LABELS
'''
Features(x): List of 50 sequence numbers(index)
Labels(y): List of 1 following number
'''
print("\n[STEP2] Features and Labels")
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
#print("features shape:", features.shape) #(354403, 50)
print("  >> there are %s training sequences" %len(features))

# NOTE: neaural network can train most effectively (better+faster) when the labels are one-hot encoded
num_words = len(word_to_idx)+1 #19092 # +1 because indexed_word starts from 1 and we need one more extra column for index=0

# randomly shuffle data
features, labels= shuffle(features, labels, random_state=RANDOM_STATE)
print("one hot encoing of labels...")
# empty array to hold labels
onehot_labels = np.zeros((len(features), num_words), dtype=np.int8) #(354403, 19093)

# one hot encode the labels
for idx, word_indexed in enumerate(labels):
    onehot_labels[idx][word_indexed] = 1
# to find corresponding owrd: idx_to_word[np.argmax(onehot_labels[idx])]

print("split into training and validation data..")
train_end = int(TRAIN_FRACTION * len(labels))

X_train = np.array(features[:train_end])
X_valid = np.array(features[train_end:])

y_train = onehot_labels[:train_end]
y_valid = onehot_labels[train_end:]
print("  >> training data X:", X_train.shape)
print("  >> training data y:", y_train.shape)

# Memory management
import gc
gc.enable()
del features, labels, onehot_labels
gc.collect()
''' NOTE: can check with
import sys
sys.getsizeof({variable})/1e9
'''
import sys
def check_sizes(gb_min=1):
    for x in globals():
        size = sys.getsizeof(eval(x)) / 1e9
        if size > gb_min:
            print(f'Object: {x:10}\tSize: {size} GB.')


check_sizes(gb_min=1)

# 3. PRE-TRAINED EMBEDDINGS
'''
Among many different kinds of pre-trained embedding, we will use glove embedding (400K trained words in 100-dim)
source: https://nlp.stanford.edu/projects/glove/
'''
print("\n[STEP3] Pre-trained Embeddings")
# load in embeddings
glove_vectors = "./glove.6B.100d.txt"
glove = np.loadtxt(glove_vectors, dtype='str', comments=None) #400k,101 (word+100-d)

# extract the vectors and words
vectors = glove[:, 1:].astype('float') #400k 100
words = glove[:, 0] #400k 1

# create lookup of words to vectors
word_lookup = {word: vector for word, vector in zip(words, vectors)}

# new matrix to hold word embeddings
print("building our embedding matrix based on the pre-trained one...")
embedding_matrix = np.zeros((num_words, vectors.shape[1]))#19093, 100

not_found = 0
for i, word in enumerate(word_to_idx.keys()):
    vector = word_lookup.get(word, None)
    if vector is not None:
        embedding_matrix[i+1, :] = vector
    else:
        not_found += 1
print("  >>There were {0}/{1} words not found in pre-trained embedding".format(not_found, num_words))

gc.enable()
del vectors, words
gc.collect()

# NOTE: Each word is represented by 100 numbers with a numbner of words that can't be found. We can find the closet words to a given word in embedding space using the cosine distance. It requires first normalization the vectors to have a magnitude 1.
def find_closest(query, em=embedding_matrix,
                wti=word_to_idx, itw=idx_to_word, n=10):
    '''find closet n words to a query word in embeddins'''
    idx = wti.get(query, None)
    if idx is None:
        print(f"{query} not found in vocab.")
        return
    else:
        vec = em[idx]
        # handle case where word doesn't have an embedding
        if np.all(vec == 0):
            print(f"{query} has no pre-trained embedding.")
            return
        else:
            # calculate distance between vector and all others
            dists = np.dot(em, vec)
            print(dists)
            # sort indexes in reverse order
            idxs = np.argsort(dists)[::-1][:n]
            sorted_dists = dists[idxs]
            closest = [itw[i] for i in idxs]
    print(f"Query:{query}\n")
    max_len = max([len(i) for i in closest])
    for word, dist in zip(closest, sorted_dists):
        print(f"Word: {word:15} Cosine Similarity: {round(dist,4)}")

print("embedding matrix is ready:", embedding_matrix.shape)
# 4. BUILDING A RECURRENT NEURAL NETWORK
'''
using Keras Sequential API,
which menas that we build the network up one layer at a time
'''
print("\n[STEP4] Building a RNN model")
def make_word_level_model(num_words,
                          em = embedding_matrix,
                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          bi_direc=False):
    model = Sequential()
    # map words to an embedding (embedding layer)
    if not trainable:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=em.shape[1], #each input word to 100-dim vec
                #input_length=training_length,
                weights=[em], #pre-trained weights
                trainable=False, #we don't want to update the embeddings
                mask_zero=True))

        # masking layer for pre-trained embeddings
        # mask any words that do not have a pre-trained embedding which will be represented as all zeros
        model.add(Masking(mask_value=0.0))
        #<=> model.add(Masking())
    else:
        model.add(
            Embedding(
                input_dim=num_words,
                output_dim=em.shape[1],
                weights=[em],
                trainable=True))

    # recurrent layer (LSTM cell)
    ## if want to add multiple LSTM layers
    if lstm_layers > 1:
        for i in range(lstm_layers - 1):
            model.add(
                LSTM(
                    lstm_ceslls,
                    return_sequences=True,
                    # for multiple layers, return_sequence should be True
                    dropout=0.1,
                    # dropout is a regularization technique for NN models where randomly selected neurons are ignored during training. By setting dropout, we can prevent overfitting
                    recurrent_dropout=0.1))
    ## add final LSTM cell layer
    if bi_direc:
        model.add(
            Bidirectional(
                LSTM(
                    lstm_ceslls,
                    return_sequences=False,
                    # return_sequence=F as we're using only one LSTM layer
                    dropout=0.1,
                    recurrent_dropout=0.1)))
    else:
        model.add(
            LSTM(
                lstm_cells,
                return_sequences=False,
                dropout=0.1,
                recurrent_dropout=0.1))

    # fully connected layer
    # dense layer to prevent overfitting to the trainign data
    model.add(Dense(128, activation='relu'))

    # dropout for regularization
    model.add(Dropout(0.5))

    # output layer
    # it produces a probability for every word in the vocab using softmax
    model.add(Dense(num_words, activation='softmax'))

    # compile the model
    model.compile(
        optimizer='adam', # a variant on stochastic gradient descent
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    return model

#NOTE: the model needs a loss to minimize(categorical_crossentropy) as well as a method for updating the weights using the gradients(Adam)
# Also we will monitor accuracy which is an interpretable measure of the model performance.
model = make_word_level_model(
    num_words,
    embedding_matrix,
    LSTM_CELLS,
    trainable=False,
    lstm_layers=1)
model.summary()
print("Summary Output Shape Interpretation:")
print("  >> (batch_size, timesteps(input), features)")
# NOTE: (batch_size, timesteps(input), features) shape
# Using pre-trained embeddings lose many words to train --> not good
model_name = 'pre-trained-rnn'
model_dir = './models/'
try:
    from IPython.display import Image
    plot_model(model,
            to_file=f'{model_dir}{model_name}.png',
            show_shapes=True)
    Image(f'{model_dir}{model_name}.png')
except:
    print("Failed to save IPython.display image")

# 5. TRAINING THE MODEL
print("\n[STEP5] Training the model")
def make_callbacks(model_name, save=SAVE_MODEL):
    '''create callbacks'''
    callbacks = [EarlyStopping(monitor='val_loss', patience=5)]
    # we won't overfit to the training data and waste time training for extra epochs that don't improve performance. It will halt training when validation loss is no longer decreasing.
    if save:
#       # saves the best model (measured by validation loss) on disk for using best model
        callbacks.append(
            ModelCheckpoint(
                f'{model_dir}{model_name}.h5',
                save_best_only=True,
                save_weights_only=False))
    return callbacks

callbacks = make_callbacks(model_name)

history = model.fit(X_train,
                    y_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=VERBOSE,
                    callbacks=callbacks,
                    validation_data=(X_valid, y_valid))

def load_and_evaluate(model_name, return_model=False):
    '''Load the model and evaluate with log loss and accuracy'''
    model = load_model(f'{model_dir}{model_name}.h5')
    r = model.evaluate(X_valid, y_valid, batch_size=2048, verbose=1)
    print(r)
    valid_crossentropy = r[0]
    valid_accuracy = r[1]
    print(f"Cross Entropy: {round(valid_crossentropy,4)}")
    print(f"Accuracy: {round(100*valid_accuracy,2)}%")
    if return_model:
        return model

model = load_and_evaluate(model_name, return_model=True)

#NOTE: To check how the model compares to just using the word frequencies to make predictions, we can compute the accuracy if we were to use the most frequent word for every quess. We can also choose from a multinomial distribution using the word frequencies as probabilities.
print('Checking how the model compares to just using the word frequencies to make predictions, we can compute the accuracy if we were to use the most frequent word for every quess...')

np.random.seed(40)
# number of all words
total_words = sum(word_counts.values())

# compute frequency of each word in vocab
frequencies = [word_counts[word]/total_words for word in word_to_idx.keys()]
frequencies.insert(0,0)
frequencies[1:10]
list(word_to_idx.keys())[0:9]
print("The most common word is `the`")
print("See the accuracy of guessing `the` for every validation example...")
print(" >> The accuracy is", round(100 * np.mean(np.argmax(y_valid, axis=1), 4)), " %")
print("Cosine similarity test available with find_closest('query')")

print("")
print("""
We can change LSTM Layers #, Bi-directional, training length, pre-trained...
to see validation log loss, validation accuracy and # words in vocab and
select the best model
""")
