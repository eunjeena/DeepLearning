This project follows instructions based on Coursera- 'Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning'
# Image processing
## Convolution
narrows down the content of the image to focus on specific, distinct, details.
good for computer vision, because information going to the dense layers is more focused, and possible more accurate.

## Pooling
reduces the information in an image whilte maintaining features.

### Takeaways
    1. Adding more neurons (than 128), more calculation, slow learning,
but more accurate based on training data.

    2. Input layer should have same shape of the data. 28*28 input would be infeasible, so better to 'flatten' into 28*28=784*1 by using Flatten().

    3. Increasing epochs has decreasing loss, but it could overfit the data.



