import numpy
import pandas
import matplotlib.pyplot as plt

# requirement 1 read files
def read_signal(file_path):
    with open(file_path, 'r') as file:
        # First row: Read the number of samples (N)
        N = int(file.readline().strip())

        # Initialize lists to store the sample indices and values
        sample_indices = []
        sample_values = []

        # Loop over the remaining lines and parse the sample index and value
        for _ in range(N):
            line = file.readline().strip()
            sample_index, sample_value = map(float, line.split())
            sample_indices.append(int(sample_index))
            sample_values.append(sample_value)

    return sample_indices, sample_values


file_path = 'Signal1.txt'
sample_indices, sample_values = read_signal(file_path)

file_path2 = 'Signal2.txt'
sample_indices2, sample_values2 = read_signal(file_path2)


# requirement 2 visualize the signals
def visualize_signal(sample_indices, sample_values):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, sample_values, marker='o', linestyle='-', color='b', label='Signal')

    plt.title("Signal Visualization")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Value")
    plt.grid(True)
    plt.legend()
    plt.show()


# first signal
visualize_signal(sample_indices, sample_values)

# second signal
visualize_signal(sample_indices2, sample_values2)


# requirement 3 add signals
def addSignals(sample_indices1, sample_values1, sample_indices2, sample_values2):
    min_index = min(min(sample_indices1), min(sample_indices2))
    max_index = max(max(sample_indices1), max(sample_indices2))

    result_indices = list(range(min_index, max_index + 1))
    result_values = []

    for index in result_indices:
        value1 = 0
        value2 = 0

        if index in sample_indices1:
            value1 = sample_values1[sample_indices1.index(index)]

        if index in sample_indices2:
            value2 = sample_values2[sample_indices2.index(index)]

        result_values.append(value1 + value2)

    return result_indices, result_values

# Adding the signals
result_indices, result_values = addSignals(sample_indices, sample_values, sample_indices2, sample_values2)

print("Addition")
print(result_indices)
print(result_values)
visualize_signal(result_indices, result_values)


# requirement 4 subtract signals
def subtractSignals(sample_indices1, sample_values1, sample_indices2, sample_values2):
    min_index = min(min(sample_indices1), min(sample_indices2))
    max_index = max(max(sample_indices1), max(sample_indices2))

    # Initialize the result lists
    result_indices = list(range(min_index, max_index + 1))
    result_values = []

    # Loop through each index in the full range
    for index in result_indices:
        value1 = 0
        value2 = 0

        if index in sample_indices1:
            value1 = sample_values1[sample_indices1.index(index)]

        if index in sample_indices2:
            value2 = sample_values2[sample_indices2.index(index)]

        result_values.append(value1 - value2)

    return result_indices, result_values

# Subtracting the signals
result_indices, result_values = subtractSignals(sample_indices, sample_values, sample_indices2, sample_values2)

print("Subtraction")
print(result_indices)
print(result_values)
visualize_signal(result_indices, result_values)

