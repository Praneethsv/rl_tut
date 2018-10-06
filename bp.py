import numpy as np





def initialize_network(n_inputs, n_hidden, n_outputs):

    """ EACH NEURON HAS A SET OF WEIGHTS THAT NEEDS TO BE MAINTAINED. ONE WEIGHT FOR EACH I/P CONNECTION AND AN ADDITIONAL
    WEIGHT FOR BIAS"""

    model = list()
    hidden_layer = [{"weights":[np.random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    model.append(hidden_layer)
    output_layer = [{"weights":[np.random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    model.append(output_layer)
    return model

def activate(weights, inputs):

    activation = weights[-1]

    for i in range(len(weights)-1): ## removing the bias weight
        activation += weights[i] * inputs[i]
    return activation

def transfer(activation):

    return 1.0 / (1.0 + np.exp(-activation))

def forward_propagate(model, row):

    inputs = row

    for layer in model:
        new_inputs=[]  # for output layers and result of hidden layer is stored in here
        for neuron in layer:
            activation = activate(weights=neuron['weights'], inputs=inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

def derivative(output):

    return output(1.0 - output)

def backward_propagate_error(model, expected):

    for i in reversed(range(len(model))): # starting from output layer
        errors = list()
        layers = model[i]
        if i != len(model)-1:

            for j in range(len(layers)):
                error = 0.0
                for neuron in model[i+1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)

        else:

            for j in range(len(layers)):
                neuron = layers[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layers)):
            neuron = layers[j]
            neuron['delta'] = errors[j] * derivative(neuron['output'])





