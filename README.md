Interpolate a feedforward neural network to a 1D function given as a symbolic argument from the command line.

To execute: 

python NN_represent_func.py -f "f(x)"

where f(x) can be any kind of function (e.g. cos(x) * exp(-x**2) + x**2 * sin(x) or whatever).

python NN_represent_func.py -h gives the full list of arguments that can be parsed (will overrule the default).

##################

The script (python 3.7.0) relies on:

Numpy;
Matplotlib and seaborn (optional) for plotting;
PyTorch to define and train the neural network model;
Scikit-learn to split the dataset in training + test;
SymPy to handle symbolic mathematical expressions.
The interpolation of data files (e.g. time series) will be implemented in the near future.
