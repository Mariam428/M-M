import numpy as np
import pandas
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

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
    interpolator = interp1d(sample_indices, sample_values, kind='cubic')

    smooth_indices = np.linspace(min(sample_indices), max(sample_indices), 500)
    smooth_values = interpolator(smooth_indices)

    plt.figure(figsize=(10, 6))
    plt.plot(smooth_indices, smooth_values, color='b', label='Signal')  # Smooth curve
    plt.scatter(sample_indices, sample_values, color='red', marker='o')  # Original sample points for reference

    plt.title("Signal Visualization (Smooth Curve)")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Value")
    plt.grid(True)
    plt.legend()
    plt.show()


# first signal
# visualize_signal(sample_indices, sample_values)

# second signal
# visualize_signal(sample_indices2, sample_values2)


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
# result_indices, result_values = addSignals(sample_indices, sample_values, sample_indices2, sample_values2)
#
# print("Addition")
# print(result_indices)
# print(result_values)
# visualize_signal(result_indices, result_values)

# requirement 4 Multiply signal by a constant
def multiplySignal(sample_indices, sample_values, const):
    result_indices = sample_indices
    result_values = []  # Initialize an empty list to store the multiplied values

    # Iterate through the sample values using a for loop
    for i in range(len(result_indices)):
        # Multiply each value by the constant and append to result_values
        result_values.append(const * sample_values[i])

    return result_indices, result_values


# result_indices, result_values = multiplySignal(sample_indices, sample_values,5)
# print(result_indices)
# print(result_values)
# visualize_signal(result_indices, result_values)


# requirement 5 subtract signals
def subtractSignals(sample_indices1, sample_values1, sample_indices2, sample_values2):
    sample_indices3, sample_values3 = multiplySignal(sample_indices2, sample_values2, -1)
    result_indices, result_values = addSignals(sample_indices1, sample_values1,sample_indices3, sample_values3)

    return result_indices, result_values

# Subtracting the signals
# result_indices, result_values = subtractSignals(sample_indices, sample_values, sample_indices2, sample_values2)
#
# print("Subtraction")
# print(result_indices)
# print(result_values)
# visualize_signal(result_indices, result_values)


# delaying a signal

def delayingSignals(sample_indices1, sample_values1,const):
    result_indices = list(range((sample_indices1[0]+const), sample_indices1[-1]+const +1))
    result_values = sample_values1

    return result_indices, result_values

# Multiplying the signals
# result_indices, result_values = delayingSignals(sample_indices, sample_values,3)
#
# print("delayingSignals")
# print(result_indices)
# print(result_values)
# visualize_signal(result_indices, result_values)


# advancing a signal

def advancingSignals(sample_indices1, sample_values1, const):
    result_indices = list(range((sample_indices1[0]-const), sample_indices1[-1]-const+1))
    result_values = sample_values1

    return result_indices, result_values

# result_indices, result_values = advancingSignals(sample_indices, sample_values,3)
#
# print("advancingSignals")
# print(result_indices)
# print(result_values)
# visualize_signal(result_indices, result_values)

def foldingSignals(sample_indices1, sample_values1):
    result_indices = []
    result_values = []

    for value in reversed(sample_indices1):
        result_indices.append(value * -1)

    for value in reversed(sample_values1):
        result_values.append(value)

    return result_indices, result_values


result_indices, result_values = foldingSignals(sample_indices, sample_values)

print("advancingSignals")
print(result_indices)
print(result_values)
# visualize_signal(result_indices, result_values)
