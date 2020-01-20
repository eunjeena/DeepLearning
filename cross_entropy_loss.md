# Tasks
## Multi-Class Classification
- One-of-many classification
- Each sample can belong to ONE class
- e.g: [0 0 1] [1 0 0] [0 1 0]
## Multi-Label Classification
- Each sample can belong to more than one class
- e.g: [1 0 1] [1 0 0] [1 1 1]

# Output Activation Functions
transformations we apply to vectors coming out from CNNs before the loss computation

## Sigmoid
- squashes a vector in the range (0,1)
- <=> Logistic function

## Softmax
- squarshes a vector in the range (0,1) and all the resulting elements add up to 1
- can be interpreted as class probabilities

# Losses
## Cross-Entropy Loss (CE Loss)
- <=> Logistic Loss, Multinomial Logictic Loss

## Categorical Cross-Entropy Loss
- <=> Softmax Loss ( Softmax activation + Cross-Entropy loss)
- used for multi-class classification with one-hot labels


## Binary Cross-Entropy Loss
- <=> Sigmoid Cross-Entropy Loss (Sigmoid activation + Cross-Entropy loss)
- Unless Softmax loss, it is independent for each vector component (class)
- used for multi-label classification


- 
