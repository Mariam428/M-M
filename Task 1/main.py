import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tkinter import *
from tkinter import filedialog, messagebox
import testcases as test
import QuanTest2
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
levels_flag=False
bits_flag=False
def quantize_signal(num):
    global levels_flag, bits_flag
    levels=0
    bits=0
    print("in quantize signal")
    load_signal()
    global sample_indices1, sample_values1
    quantized_values=[]
    encoded_values=[]
    print("input indices")
    print(sample_indices1)
    print("input values")
    print(sample_values1)
    max_value = max(sample_values1)
    min_value = min(sample_values1)
    print("Maximum value:", max_value)
    print("Minimum value:", min_value)
    if levels_flag:
        print("user entered levels")
        levels=int(num)
        Delta=round((max_value-min_value)/levels,2)
    elif bits_flag:
        print("user entered bits")
        bits=int(num)
        Delta = (max_value - min_value) / pow(2,bits)
        levels=pow(2,bits)
    print(f"delta: {Delta}")
    intervals = [0] * int(levels+1)
    mids=levels
    midpoints= [0] * mids
    #print(f"intervals before fill {intervals}")
    intervals[0]=min_value
    intervals[levels]=max_value

    for i in range(1, levels):
        intervals[i] =round( intervals[i - 1] + Delta, 3)

    print(f"intervals after fill {intervals}")
    for i in range(levels):
        midpoints[i] = round((intervals[i] + intervals[i + 1]) / 2,3)

    print(f"midpoints: {midpoints}")
    # Assuming 'sample_values1' and 'midpoints' are already defined
    quantized_values = []
    quantized_error = []
    interval_index = 0
    for value in sample_values1:
        # Initialize minimum distance with a large number and nearest midpoint as None
        #print(f"value{value}")
        min_distance = float('inf')
        nearest_midpoint = None
        counter =0
        for midpoint in midpoints:
            # Calculate the distance between the sample value and the midpoint
            distance = round(abs(value - midpoint),2)

            # Update nearest midpoint if a closer one is found
            if distance < min_distance:
                #print(distance)
                #print(min_distance)
                min_distance = distance
                nearest_midpoint = midpoint
                interval_index=counter
            counter+=1
        # Append the nearest midpoint to quantized_values
        quantized_values.append(nearest_midpoint)
        quantized_error.append(nearest_midpoint-value)
        encoded_values.append(interval_index)

    quantized_values = [round(value, 3) for value in quantized_values]
    print(f"Quantized Signal values {quantized_values}")

    quantized_error = [round(value, 3) for value in  quantized_error]
    print(f"Quantized Error { quantized_error}")
    print(f"encoded values { encoded_values}")
    encoded_values_bits=[]
    # Loop to convert each value to binary and print it
    for value in encoded_values:
        binary_representation = bin(value)[2:].zfill(3)
        encoded_values_bits.append( binary_representation)

    print(f"Encoded values  {  encoded_values_bits}")
    visualize_discrete_signal(sample_indices1, quantized_values)

    indices=[]
    for value in encoded_values:
        indices.append(value + 1)
    print(f"indices  {indices}")
    #QuantizationTest2('Quan2_out.txt', indices, encoded_values_bits, quantized_values, quantized_error)
    QuantizationTest1('Quan1_out.txt', encoded_values_bits, quantized_values)


file_path = 'Quan1_input.txt'
sample_indices, sample_values = read_signal(file_path)

file_path2 = 'Signal2.txt'
sample_indices2, sample_values2 = read_signal(file_path2)


# requirement 2 visualize the signals
def visualize_continuous_signal(sample_indices, sample_values):
    plt.figure(figsize=(10, 6))
    plt.plot(sample_indices, sample_values, marker='o', linestyle='-', color='b', label='Continuous Signal')

    plt.title("Signal Visualization")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Value")
    plt.grid(True)
    plt.legend()
    plt.show()

def visualize_discrete_signal(sample_indices, sample_values):
    plt.figure(figsize=(10, 6))
    plt.stem(sample_indices, sample_values, linefmt='b-', markerfmt='bo', basefmt='r-', label='Discrete Signal')

    plt.title("Discrete Signal Visualization")
    plt.xlabel("Sample Index")
    plt.ylabel("Sample Value")
    plt.grid(True)
    plt.legend()
    plt.show()


# add signals
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

    print("Result Indices (Addition):", result_indices)
    print("Result Values (Addition):", result_values)
    return result_indices, result_values



# Multiply signal by a constant
def multiplySignal(sample_indices, sample_values, const):
    result_indices = sample_indices
    result_values = []  # Initialize an empty list to store the multiplied values

    # Iterate through the sample values using a for loop
    for i in range(len(result_indices)):
        # Multiply each value by the constant and append to result_values
        result_values.append(const * sample_values[i])

    print("Result Indices (Multiplication):", result_indices)
    print("Result Values (Multiplication):", result_values)

    return result_indices, result_values


# subtract signals
def subtractSignals(sample_indices1, sample_values1, sample_indices2, sample_values2):
    sample_indices3, sample_values3 = multiplySignal(sample_indices2, sample_values2, -1)
    result_indices, result_values = addSignals(sample_indices1, sample_values1,sample_indices3, sample_values3)

    print("Result Indices (Subtraction):", result_indices)
    print("Result Values (Subtraction):", result_values)
    return result_indices, result_values

# delaying a signal

def delayingSignals(sample_indices1, sample_values1,const):
    result_indices = list(range((sample_indices1[0]+const), sample_indices1[-1]+const +1))
    result_values = sample_values1

    print("Result Indices (Delay):", result_indices)
    print("Result Values (Delay):", result_values)

    return result_indices, result_values


# advancing a signal

def advancingSignals(sample_indices1, sample_values1, const):
    result_indices = list(range((sample_indices1[0]-const), sample_indices1[-1]-const+1))
    result_values = sample_values1

    print("Result Indices (Advance):", result_indices)
    print("Result Values (Advance):", result_values)

    return result_indices, result_values

def foldingSignals(sample_indices1, sample_values1):
    result_indices = []
    result_values = []

    for value in reversed(sample_indices1):
        result_indices.append(value * -1)

    for value in reversed(sample_values1):
        result_values.append(value)

    print("Result Indices (Folding):", result_indices)
    print("Result Values (Folding):", result_values)

    return result_indices, result_values


def generate_signal(signal_type, amplitude, phase_shift, Fmax, Fs, duration=1.0):
    # Ensure the sampling theorem is respected
    signal = 0
    if Fs < 2 * Fmax:
        raise ValueError(
            "Fs >= 2 * Fmax")

    t = np.arange(0, duration, 1 / Fs)

    if signal_type == 'sine':
        signal = amplitude * np.sin(2 * np.pi * Fmax * t + phase_shift)
    elif signal_type == 'cosine':
        signal = amplitude * np.cos(2 * np.pi * Fmax * t + phase_shift)

    return t, signal


# GUI Functions
def load_signal():
    file_path = filedialog.askopenfilename()
    if file_path:
        global sample_indices1, sample_values1
        sample_indices1, sample_values1 = read_signal(file_path)
        visualize_continuous_signal(sample_indices1, sample_values1)
    else:
        messagebox.showerror("Error", "No file selected")

def load_second_signal():
    file_path = filedialog.askopenfilename()
    if file_path:
        global sample_indices2, sample_values2
        sample_indices2, sample_values2 = read_signal(file_path)
        visualize_continuous_signal(sample_indices2, sample_values2)
    else:
        messagebox.showerror("Error", "No file selected")

def add_signals_gui():
    result_indices, result_values = addSignals(sample_indices1, sample_values1, sample_indices2, sample_values2)
    visualize_continuous_signal(result_indices, result_values)

def subtract_signals_gui():
    result_indices, result_values = subtractSignals(sample_indices1, sample_values1, sample_indices2, sample_values2)
    visualize_continuous_signal(result_indices, result_values)

def multiply_signal_gui():
    const = float(entry_const.get())
    result_indices, result_values = multiplySignal(sample_indices1, sample_values1, const)
    visualize_continuous_signal(result_indices, result_values)

def delay_signal_gui():
    const = int(entry_const.get())
    result_indices, result_values = delayingSignals(sample_indices1, sample_values1, const)
    visualize_continuous_signal(result_indices, result_values)

def advance_signal_gui():
    const = int(entry_const.get())
    result_indices, result_values = advancingSignals(sample_indices1, sample_values1, const)
    visualize_continuous_signal(result_indices, result_values)

def fold_signal_gui():
    result_indices, result_values = foldingSignals(sample_indices1, sample_values1)
    visualize_continuous_signal(result_indices, result_values)
def quantize_signal_gui():
    const = entry_const.get()
    quantize_signal(const)
def open_signal_generation_menu():
    # Create a new top-level window for signal generation
    signal_window = Toplevel(root)
    signal_window.title("Signal Generation")

    # Input fields for signal parameters
    Label(signal_window, text="Amplitude:").grid(row=0, column=0, padx=10, pady=10)
    amplitude_entry = Entry(signal_window)
    amplitude_entry.grid(row=0, column=1, padx=10, pady=10)

    Label(signal_window, text="Phase Shift:").grid(row=1, column=0, padx=10, pady=10)
    phase_shift_entry = Entry(signal_window)
    phase_shift_entry.grid(row=1, column=1, padx=10, pady=10)

    Label(signal_window, text="Fmax:").grid(row=2, column=0, padx=10, pady=10)
    fmax_entry = Entry(signal_window)
    fmax_entry.grid(row=2, column=1, padx=10, pady=10)

    Label(signal_window, text="Fs:").grid(row=3, column=0, padx=10, pady=10)
    fs_entry = Entry(signal_window)
    fs_entry.grid(row=3, column=1, padx=10, pady=10)

    Label(signal_window, text="Duration:").grid(row=4, column=0, padx=10, pady=10)
    duration_entry = Entry(signal_window)
    duration_entry.grid(row=4, column=1, padx=10, pady=10)

    # Function to generate signal based on user input
    def generate_selected_signal(signal_type):
        try:
            amplitude = float(amplitude_entry.get())
            phase_shift = float(phase_shift_entry.get())
            Fmax = float(fmax_entry.get())
            Fs = float(fs_entry.get())
            duration = float(duration_entry.get())
            t, signal = generate_signal(signal_type, amplitude, phase_shift, Fmax, Fs, duration)
            return t, signal
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for all fields")

    # Function to visualize continuous signal
    def generate_continuous():
        t, signal = generate_selected_signal(selected_signal_type.get())
        visualize_continuous_signal(t, signal)

    # Function to visualize discrete signal
    def generate_discrete():
        t, signal = generate_selected_signal(selected_signal_type.get())
        visualize_discrete_signal(t, signal)

    # Create a variable to store the selected signal type
    selected_signal_type = StringVar(signal_window, "sine")

    # Buttons for sine, cosine, and visualization
    Button(signal_window, text="Sine", command=lambda: selected_signal_type.set("sine")).grid(row=5, column=0, padx=10, pady=10)
    Button(signal_window, text="Cosine", command=lambda: selected_signal_type.set("cosine")).grid(row=5, column=1, padx=10, pady=10)

    Button(signal_window, text="Generate Continuous", command=generate_continuous).grid(row=6, column=0, padx=10, pady=10)
    Button(signal_window, text="Generate Discrete", command=generate_discrete).grid(row=6, column=1, padx=10, pady=10)

# Tkinter GUI Setup
root = Tk()
root.title("Signal Processing")

# Load buttons
Button(root, text="Load Signal 1", command=load_signal).grid(row=0, column=0, padx=10, pady=10)
Button(root, text="Load Signal 2", command=load_second_signal).grid(row=0, column=1, padx=10, pady=10)

# Operation buttons
Button(root, text="Add Signals", command=add_signals_gui).grid(row=1, column=0, padx=10, pady=10)
Button(root, text="Subtract Signals", command=subtract_signals_gui).grid(row=1, column=1, padx=10, pady=10)
Button(root, text="Fold Signal", command=fold_signal_gui).grid(row=2, column=0, padx=10, pady=10)

# Entry for constants
Label(root, text="Enter constant:").grid(row=3, column=0, padx=10, pady=10)
entry_const = Entry(root)
entry_const.grid(row=3, column=1, padx=10, pady=10)

# More operation buttons
Button(root, text="Multiply Signal", command=multiply_signal_gui).grid(row=4, column=0, padx=10, pady=10)
Button(root, text="Delay Signal", command=delay_signal_gui).grid(row=4, column=1, padx=10, pady=10)
Button(root, text="Advance Signal", command=advance_signal_gui).grid(row=5, column=0, padx=10, pady=10)
Button(root, text="Signal Generation", command=open_signal_generation_menu).grid(row=6, column=0, padx=10, pady=10)
Button(root, text="Quantize Signal", command=quantize_signal_gui).grid(row=5, column=1, padx=10, pady=10)
Button(root, text="Read Levels", command=lambda: globals().update({'levels_flag': True})).grid(row=2, column=1, padx=10, pady=10)
Button(root, text="Read Bits", command=lambda: globals().update({'bits_flag': True})).grid(row=2, column=2, padx=10, pady=10)


def QuantizationTest2(file_name, Your_IntervalIndices, Your_EncodedValues, Your_QuantizedValues, Your_SampledError):
    expectedIntervalIndices = []
    expectedEncodedValues = []
    expectedQuantizedValues = []
    expectedSampledError = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 4:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = str(L[1])
                V3 = float(L[2])
                V4 = float(L[3])
                expectedIntervalIndices.append(V1)
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                expectedSampledError.append(V4)
                line = f.readline()
            else:
                break
    if (len(Your_IntervalIndices) != len(expectedIntervalIndices)
            or len(Your_EncodedValues) != len(expectedEncodedValues)
            or len(Your_QuantizedValues) != len(expectedQuantizedValues)
            or len(Your_SampledError) != len(expectedSampledError)):
        print("QuantizationTest2 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_IntervalIndices)):
        if (Your_IntervalIndices[i] != expectedIntervalIndices[i]):
            print("QuantizationTest2 Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(Your_EncodedValues)):
        if (Your_EncodedValues[i] != expectedEncodedValues[i]):
            print(
                "QuantizationTest2 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return

    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print(
                "QuantizationTest2 Test case failed, your QuantizedValues have different values from the expected one")
            return
    for i in range(len(expectedSampledError)):
        if abs(Your_SampledError[i] - expectedSampledError[i]) < 0.01:
            continue
        else:
            print("QuantizationTest2 Test case failed, your SampledError have different values from the expected one")
            return
    print("QuantizationTest2 Test case passed successfully")
def QuantizationTest1(file_name,Your_EncodedValues,Your_QuantizedValues):
    expectedEncodedValues=[]
    expectedQuantizedValues=[]
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L=line.strip()
            if len(L.split(' '))==2:
                L=line.split(' ')
                V2=str(L[0])
                V3=float(L[1])
                expectedEncodedValues.append(V2)
                expectedQuantizedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedEncodedValues)) or (len(Your_QuantizedValues)!=len(expectedQuantizedValues))):
        print("QuantizationTest1 Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedEncodedValues[i]):
            print("QuantizationTest1 Test case failed, your EncodedValues have different EncodedValues from the expected one")
            return
    for i in range(len(expectedQuantizedValues)):
        if abs(Your_QuantizedValues[i] - expectedQuantizedValues[i]) < 0.01:
            continue
        else:
            print("QuantizationTest1 Test case failed, your QuantizedValues have different values from the expected one")
            return
    print("QuantizationTest1 Test case passed successfully")

root.mainloop()