import numpy as np

def logistic_function(x):
    return 1 / (1 + np.exp(-x))

def calculate_probabilities(parameters, data, utilities):
    probabilities = {}
    for key, utility_func in utilities.items():
        V = utility_func(parameters, data)
        exp_V = np.exp(V)
        denominator = sum([av * exp for av, exp in zip(data[key], exp_V)])
        probabilities[key] = [av * exp / denominator for av, exp in zip(data[key], exp_V)]
    return probabilities

def utility_V1(parameters, data):
    return parameters['β01'] + parameters['β1'] * data['X1'] + parameters['βS1,13'] * data['S1']

def utility_V2(parameters, data):
    return parameters['β02'] + parameters['β2'] * data['X2'] + parameters['βS1,23'] * data['S1']

def utility_V3(parameters, data):
    return parameters['β03'] + parameters['β1'] * data['Sero'] + parameters['β2'] * data['Sero']

parameters = {
    'β01': 0.1, 'β1': -0.5, 'β2': -0.4,
    'β02': 1, 'β03': 0, 'βS1,13': 0.33, 'βS1,23': 0.58
}

data = {
    'X1': [2, 1, 3, 4, 2, 1, 8, 7, 3, 2],
    'X2': [8, 7, 4, 1, 4, 7, 2, 2, 3, 1],
    'Sero': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'S1': [3, 8, 4, 7, 1, 6, 5, 9, 2, 3],
    'AV1': [1, 1, 1, 1, 1, 0, 0, 1, 1, 0],
    'AV2': [1, 1, 1, 0, 0, 1, 1, 1, 0, 1],
    'AV3': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]
}

utilities = {
    'P1': utility_V1,
    'P2': utility_V2,
    'P3': utility_V3
}

probabilities = calculate_probabilities(parameters, data, utilities)

# Save probabilities to a text file
with open('output.txt', 'w') as file:
    for key, values in probabilities.items():
        file.write(f'{key}: {values}\n')
