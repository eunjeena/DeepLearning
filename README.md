# DeepLearning
### What is Deep Learning?
Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled.

### Traiditional Neural Network
- Learn without Persistence.
- It is like calssifying what kind of event is happening at every point in a movie.

### Recurrent Neural Network (RNN)
- Networks with loops in them, allowing information to persist.
- Intimately related to sequences and lists. (Multiple copies of NNs)
- Usage: Speech recognition, Language modeling, Translation, Image captioning...
- Repeating module in a standard RNN contins a single (tahn) layer.
- shortcoming: cannot learn long-term dependency.
- e.g: I grew up in Korea.. I speak fluent `Korean`
  RNNs can know it's language, but it may be hard to know which language.

### Long Short Term Memory (LSTM) networks
- A very special kind of RNN, which can learn long-term dependency as default.
- Better in cases where we need more context because RNNs cannot learn to connect the information in practice as the gap grows.
- Repeating module in 4 layers (standard).
    1. Forget gate layer (sigmoid layer)
        - disregard irregular information.
        - how: decide how much to keep C(t-1) (previous cell state)
    2. Input gate layer (sigmoid + tanh layers)
        - handle current input
        - how: sigmoid layer decides which value to update from h(t-1)
              tanh layer decides which to add new information
    3. Ouput gate layer (sigmoid layer + tanh)
        - produce predictions h(t)
        - how: sigmoid layer decides what parts of the cell state it will output
              and put the cell state through tanh to have value -1 - 1
- There are many variances of LSTM networks other than above.


