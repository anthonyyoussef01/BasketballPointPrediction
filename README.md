# Basketball Point Prediction

### Instructions on running notebook and seeing results for linear and ridge regression models:

The notebook for the linear and ridge regression models is called DS4400_Project.ipynb

In order to run the linear and ridge regression models. Simply run all cells in the provided
notebook in sequential order.

### Instructions to run neural network and see results

1. Install venv if not installed.
2. Create a new virtual environment and install requirements.txt using pip in the environment
3. Run the command below in a terminal window
    
`python neural_network.py`

Note, default configuration variables are set in file:

#### For data parsing
games_to_look_back = 10

#### For the model architecture
#### Defining input size, hidden layer size, output size and batch size respectively
n_in, n_h, n_out, hidden_layers, activation_function = 44, 50, 1, 3, 'relu'

#### For training and evaluation
k_folds = 10
epochs = 150

There are 6 outputs to acknowledge:
1. Average training accuracy over the number of folds
2. Average training loss over the number of folds
3. Average testing accuracy over the number of folds
4. A list of all training accuracies during k-fold validation
5. A list of all training losses during k-fold validation
6. A list of all testing accuracies during k-fold validation




