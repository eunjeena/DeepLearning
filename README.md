# DeepLearning
### What is Deep Learning?
Deep learning is a subset of machine learning in artificial intelligence that has networks capable of learning unsupervised from data that is unstructured or unlabeled.

Simply speaking, it is training a neural network!

A neaural network: any function that fits some data.
A single neurons has no advantage over a traditional ML algorithm.
Therfore, a neural network combines multiple neurons.

- *Sequential* defines a sequence of layers in the neural network.
    The first layer in your network should be the same shape as your data.
- *Flatten* remembers earlier and turns that into a 1 dimentional set.
- *Dense* adds a layer of connected neurons.
- *Loss function* measures how good the current guess is.
- *Optimizer* generates a new improved guess based on calculated loss.
- *Activation function* tells each layer of neurons what to do.
    - *Relu* if X>0 return X, else return 0, so it only passes 0 or greater to the next layers in the network.
    - *Softmax* takes a set of values and picks the biggest one to save coding.


### Traiditional Neural Network
- Learn without Persistence.
- It is like calssifying what kind of event is happening at every point in a movie.
- Usage: online advertising purposes

### Recurrent Neural Network (RNN)
- Networks with loops in them, allowing information to persist.
- Intimately related to sequences and lists. (Multiple copies of NNs)
- Usage: Speech recognition, Language modeling, Translation, Image captioning(Convolutional NN)...
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

### Optimization Methods in Neural Networks
#### mini-batch gradient descent
Traditional gradient descent needs to process all of the training examples before making the first update to the parameters, which is obviously inefficient.
- Instead, we break up the dataset into smaller set (mini-batch).
    1. sgd (stochastic gradient descent)
        - mini-batch = 1
        - each step is taken after training on only 1 data point
        - thus, it is not very good as it often takes steps in the wrong direction and it will not converge to the global minimum
    2. batch gradient descent
        - when the dataset is small (< 2K)
    3. typically 64, 128, 256, 512 based on CPU/GPU memory
        - when the dataset is large
- Mini-batch gradient descent is a bit harder to implement (not sgd), because the trainign size is likely not divisible by the mini-batch size.
- Therefore, we need to address the last batch to accommodate that.

#### Gradient descent with momentum
- It involves applying exponential smoothing to the computed gradient.
- It will speed up training because the algorithm will oscillate less towards the minmum and it will take more steps towards the minimum.
- hyperparmaneters:
    - alpha: learning rate
    - beta : smoothing parameter

#### Adam optimization algorithm (ADAptive Moment estimation)
- momenntum + RMSprop (root mean squared prop)
- hyperparmaneters:
    - alpha: learning rate
    - beta : momentum. usually 0.9
    - beta2: RMSprop. usualyy 0.999
    - epsilon : usually 1e-8

### Improve a Neural Network with Regularization
Two common ways to address overfitting:
- Get more data (sometimes impossible, very expensive)
- User regularization

#### Regularization
1. L2 regularization
    - adds penalizing large weights
    - decreases the effect of the activation function
    - Thus, less complex function will be fit to the data (reducing overfitting)
2. Dropout regularization
    - goes over all the layers in NN and sets probability of keeping a certain nodes or not
    - threshold: 0.7 (probability of 30% that a node will be removed from the network)
    - results in smlaler, simpler neural networks
    - neural network cannot rely on any input node as each have a random probability of being removed
    - Thus, it will be reluctant to give high weights to certain featuers, making weights smaller
