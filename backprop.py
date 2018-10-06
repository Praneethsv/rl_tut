import numpy as np




def initialize_model(n_inputs, n_outputs, n_hidden):

    model = list()
    hidden_layer = [{"weights": [np.random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    model.append(hidden_layer)
    output_layer = [{"weights": [np.random.random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    model.append(output_layer)

    return model


def activate(weights, inputs):
    activation = weights[-1]

    for i in range(len(weights)-1):

        activation += weights[i] * inputs[i]

    return activation

def transfer_activation(activation):

    return 1.0 / (1.0 + np.exp(activation))

def forward_propagate(model, row):

    inputs = row

    for layer in model:
        new_inputs = []
        for neuron in layer:
            activation = activate(model['weights'], inputs)
            neuron['output'] = transfer_activation(activation=activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs

    return inputs

def transfer_derivative(output):

    return output(1.0 - output)


##### practice here one more time back prop here ########





