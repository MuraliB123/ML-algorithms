# Linear regression
Implementation is explained below

## Approach
   1. The numpy library is imported to provide support for efficient numerical operations.
   2. A class called MyLinearRegression is defined to encapsulate the functionality of the linear regression model.
   3. The __init__ method is the constructor of the class. It initializes the learning rate (learning_rate) and the number of iterations (num_iterations) with default values. The weights (weights) and bias (bias) are set to None since they will be assigned during the training process.
   4. The train method takes the input data X and target variable y as arguments and performs the training of the linear regression model.
   5.The number of samples (num_samples) and number of features (num_features) are extracted from the shape of the input data X.
   6.The weights and bias are initialized to zero using np.zeros(num_features) and 0, respectively. The weights are represented as a 1-dimensional array of size num_features.
   7.The training process iterates num_iterations times using a for loop.
   8 Within each iteration, the predicted values (y_pred) are calculated by taking the dot product of the input data X and the weights, and adding the bias.
   9. The gradients of the weights (dw) and bias (db) are calculated using the derivative of the loss function with respect to the weights and bias. These gradients represent the direction and magnitude of the changes needed to minimize the loss.
   10. The weights and bias are updated by subtracting the product of the learning rate and the gradients (self.learning_rate * dw and self.learning_rate * db, respectively). This step implements the gradient descent algorithm, adjusting the model parameters in the direction of steepest descent to minimize the loss.
   11. The predict method takes the input data X as an argument and returns the predicted values (y_pred). It calculates the dot product of the input data X and the weights, and adds the bias term.



