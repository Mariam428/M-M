import math
import signalcompare as signal_compare
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tkinter import *
from tkinter import filedialog, messagebox, simpledialog
import testcases as test
import QuanTest2
import test
from tkinter import ttk
from tkinter import filedialog, messagebox
from decimal import Decimal, ROUND_HALF_UP

# requirement 1 read files
def classify_signal():
    class11values = read_sample_values(r"Correlation Task Files/point3 Files/Class 1/down1.txt")
    class12values = read_sample_values(r"Correlation Task Files/point3 Files/Class 1/down2.txt")
    class13values = read_sample_values(r"Correlation Task Files/point3 Files/Class 1/down3.txt")
    class14values = read_sample_values(r"Correlation Task Files/point3 Files/Class 1/down4.txt")
    class15values = read_sample_values(r"Correlation Task Files/point3 Files/Class 1/down5.txt")

    class21values = read_sample_values(r"Correlation Task Files/point3 Files/Class 2/up1.txt")
    class22values = read_sample_values(r"Correlation Task Files/point3 Files/Class 2/up2.txt")
    class23values = read_sample_values(r"Correlation Task Files/point3 Files/Class 2/up3.txt")
    class24values = read_sample_values(r"Correlation Task Files/point3 Files/Class 2/up4.txt")
    class25values = read_sample_values(r"Correlation Task Files/point3 Files/Class 2/up5.txt")

    class1 = [class11values ,class12values , class13values , class14values ,class15values]
    class2 = [class21values , class22values ,class23values ,class24values ,class25values]
    corr_c1 = []
    corr_c2 = []
    global sample_indices2, sample_values2 ,sample_indices1

    # Loop through class1 signals
    for i, signal in enumerate(class1, start=1):
        sample_values2 = signal  # Update global variables
        sample_indices2 = sample_indices1 = list(range(len(sample_values2)))
        _, corr_class1 = calculate_correlation()  # Use updated globals in calculation
        corr_c1.append(corr_class1)

    # Loop through class2 signals
    for i, signal in enumerate(class2, start=1):
        sample_values2 = signal  # Update global variables
        sample_indices2 = sample_indices1 = list(range(len(sample_values2)))
        _, corr_class2 = calculate_correlation()  # Use updated globals in calculation
        corr_c2.append(corr_class2)

    # Determine the classification based on maximum correlation
    max_corr_class1 = np.max(corr_c1)
    max_corr_class2 = np.max(corr_c2)

    if max_corr_class1 > max_corr_class2:
        return "A", corr_c1
    else:
        return "B", corr_c2

def sharpen_signal():
    print("in sharpen signal")
    load_signal()
    first_derivative_values=[]
    first_derivative_indices =[]
    second_derivative_values = []
    second_derivative_indices = []
    global sample_indices1, sample_values1
    #print(sample_values1)
    #Y(n) = x(n)-x(n-1)
    #Y(n)= x(n+1)-2x(n)+x(n-1)
    for i in range (1,len(sample_values1)):
        result_first=sample_values1[i]-sample_values1[i-1]
        first_derivative_values.append(int(result_first))
        first_derivative_indices.append(i-1)
    for i in range(1, len(sample_values1)-1):
        result_second=sample_values1[i+1]-2*sample_values1[i]+sample_values1[i-1]
        second_derivative_values.append(result_second)
        second_derivative_indices.append(i-1)
    print('first derivative: ')
    print(first_derivative_indices)
    print(first_derivative_values)
    visualize_continuous_signal(first_derivative_indices,first_derivative_values)
    print('second derivative: ')
    print(second_derivative_indices)
    print(second_derivative_values)
    visualize_continuous_signal(second_derivative_indices, second_derivative_values)
    CompareSignal("task5files/1st_derivative_out.txt",first_derivative_indices,first_derivative_values)
    CompareSignal("task5files/2nd_derivative_out.txt",second_derivative_indices, second_derivative_values)
    return
def moving_average(window_size):
    #print("in moving average")
    print("window size is: ")
    print(window_size)
    load_signal()
    global sample_indices1, sample_values1
    #print(sample_values1)
    output_values=[]
    output_indices=[]
    encoded_values=[]
    for i in range(len(sample_values1)-window_size+1):
        sum=0
        for j in range(i, i+window_size):
            sum+=sample_values1[j]
        output_values.append(round(sum / window_size, 3))
        output_indices.append(i)
    visualize_continuous_signal(output_indices,output_values)
    print(output_indices)
    print(output_values)
    encoded_values.append(0)
    encoded_values.append(0)
    encoded_values.append(output_indices)
    CompareSignal("task5files/MovingAvg_out1.txt",output_indices,output_values)
    return

def convolve():
    print("in convolve")
    output_indices = []
    output_values = []
    global sample_indices1, sample_values1, sample_indices2, sample_values2
    dict1 = {sample_indices1[i]: sample_values1[i] for i in range(len(sample_indices1))}
    # Create dict2 for sample_indices2 and sample_values2
    dict2 = {sample_indices2[i]: sample_values2[i] for i in range(len(sample_indices2))}
    # Calculate the range of output indices
    first_index = sample_indices1[0] + sample_indices2[0]
    last_index = sample_indices1[-1] + sample_indices2[-1]
    # Populate output_indices
    for i in range(first_index, last_index + 1):
        output_indices.append(i)
    print("Output Indices:", output_indices)
    for i in range(first_index, last_index + 1):
        value = 0  # Initialize the convolution sum for the current output index
        # Iterate over the indices of the first signal
        for n in range(len(sample_indices1)):
            flipped_index = i - sample_indices1[n]  # Calculate the corresponding index in sample_values2_flipped

            # Check if the flipped_index is valid in dict2
            if flipped_index in dict2:
                value += dict1[sample_indices1[n]] * dict2[flipped_index]  # Add the product of overlapping values

        # Append the computed convolution value for the current output index
        output_values.append(value)


    visualize_continuous_signal(output_indices,output_values)
    print("Output Values:", output_values)
    #CompareSignal("task5files/Conv_output.txt",output_indices,output_values)
    Compare_Signals(r"FIR test cases/Testcase 7/BSFCoefficients.txt", output_indices, output_values)
    return


def DFT(signal):
    signal2 = np.array(signal)
    values = signal2  # Directly use the 1-dimensional array
    N = len(values)
    # DFT computation
    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += values[n] * np.exp(-2j * np.pi * k * n / N)

    X = np.round(X, decimals=10)

    return X
def IDFT(X):
    N = len(X)  # Length of the signal
    x_reconstructed = np.zeros(N, dtype=complex)

    # Perform IDFT
    for k in range(N):
        for n in range(N):
            x_reconstructed[n] += X[k] * np.exp(2j * np.pi * k * n / N)

    # Scale by 1/N
    x_reconstructed /= N
    return x_reconstructed.real

def FastConvolution():
    global sample_indices1, sample_values1, sample_indices2, sample_values2

    # Extract signal values
    signal_values = np.array(sample_values1)
    h_values = np.array(sample_values2)

    # Get lengths
    N1 = len(signal_values)
    N2 = len(h_values)
    padded_length = N1 + N2 - 1  # Length of the convolution result

    # Pad signals to prevent circular convolution
    signal_padded = np.pad(signal_values, (0, padded_length - N1))
    h_padded = np.pad(h_values, (0, padded_length - N2))

    # Perform FFT convolution
    signal_fft = DFT(signal_padded)
    h_fft = DFT(h_padded)
    convolution_freq_domain = signal_fft * h_fft
    convolution_time_domain = IDFT(convolution_freq_domain)

    # Round the real part of the result to 8 decimal places
    convolution_time_domain_rounded = [round(x.real, 8) for x in convolution_time_domain]

    # Compute indices for the convolution result
    convolution_indices = np.arange(
        sample_indices1[0] + sample_indices2[0],
        sample_indices1[0] + sample_indices2[0] + padded_length
    )
    print("Convolution Indices:", convolution_indices)
    print("Convolution Samples (Rounded with np.round):", convolution_time_domain_rounded)
    # Return indices and rounded samples
    return convolution_indices, convolution_time_domain_rounded

def generate_filter():
    global filter_type, fs_entry, fc_entry, transition_band_entry, attenuation_entry
    N = 0
    w = []
    h = []
    indices = []
    filtertype = filter_type.get()
    samplingfreq = float(fs_entry.get())
    f1 = float(fc_entry.get())
    transition = float(transition_band_entry.get())
    attenuation = float(attenuation_entry.get())

    # Logic for calculating N, w, and h based on filter type and other inputs
    if attenuation <= 21:
        N = np.ceil((samplingfreq * 0.9) / transition)
        if N % 2 == 0:
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(1)

    elif attenuation <= 44:
        N = np.ceil((samplingfreq * 3.1) / transition)
        if N % 2 == 0:
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.5 + (0.5 * np.cos((2 * np.pi * i) / N)))

    elif attenuation <= 53:
        N = np.ceil((samplingfreq * 3.3) / transition)
        if N % 2 == 0:
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.54 + (0.46 * np.cos((2 * np.pi * i) / N)))

    elif attenuation <= 74:
        N = np.ceil((samplingfreq * 5.5) / transition)
        if N % 2 == 0:
            N += 1

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            indices.append(i)
            w.append(0.42 + (0.5 * np.cos((2 * np.pi * i) / (N - 1))) + (0.08 * np.cos((4 * np.pi * i) / (N - 1))))

    N = int(N)
    filter_index=[]
    temp = int(N/2)
    for i in range(-temp, temp + 1, 1):
        filter_index.append(i)


    if filtertype == "lowpass":
        Fc1 = (f1 + (transition / 2)) / samplingfreq
        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            if i == 0:
                h.append(2 * Fc1)
                continue
            h.append(2 * Fc1 * ((np.sin(i * (2 * np.pi * Fc1))) / (i * (2 * np.pi * Fc1))))

    elif filtertype == "highpass":
        Fc1 = (f1 - (transition / 2)) / samplingfreq
        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            if i == 0:
                h.append(1 - (2 * Fc1))
                continue
            h.append(-2 * Fc1 * (np.sin(i * (2 * np.pi * Fc1)) / (i * (2 * np.pi * Fc1))))

    elif filtertype == "bandpass":
        f2 = float(simpledialog.askstring("Input", "Enter F2:"))
        Fc1 = (f1 - (transition / 2)) / samplingfreq
        Fc2 = (f2 + (transition / 2)) / samplingfreq

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            if i == 0:
                h.append(2 * (Fc2 - Fc1))
                continue
            h.append((2 * Fc2 * (np.sin(i * (2 * np.pi * Fc2)) / (i * (2 * np.pi * Fc2)))) -
                     (2 * Fc1 * (np.sin(i * (2 * np.pi * Fc1)) / (i * (2 * np.pi * Fc1)))))

    elif filtertype == "bandstop":
        f2 = float(simpledialog.askstring("Input", "Enter F2:"))
        Fc1 = (f1 + (transition / 2)) / samplingfreq
        Fc2 = (f2 - (transition / 2)) / samplingfreq

        for i in range(int(0 - ((N - 1) / 2)), int(((N - 1) / 2) + 1)):
            if i == 0:
                h.append(1 - (2 * (Fc2 - Fc1)))
                continue
            h.append((2 * Fc1 * (np.sin(i * (2 * np.pi * Fc1)) / (i * (2 * np.pi * Fc1)))) -
                     (2 * Fc2 * (np.sin(i * (2 * np.pi * Fc2)) / (i * (2 * np.pi * Fc2)))))

    # Multiply h and w to create the final result
    multiplied_signal = []
    indices=[]
    length = len(w)
    assert len(h) == len(w), f"Length of w and h must be equal but are {len(w)} and {len(h)}"

    for i in range(length):
        multiplied_signal.append(w[i] * h[i])
    #print(multiplied_signal)
    #print(filter_index)
    global sample_indices2 , sample_values2
    sample_indices2 = filter_index
    sample_values2 = multiplied_signal
    final_indices,final_values2 = FastConvolution()
    Compare_Signals(r"FIR test cases/Testcase 2/ecg_low_pass_filtered.txt", final_indices, final_indices)
    #convolve()
    #print(multiplied_signal)
    #print(filter_index)
    return


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
        bits=math.log2(levels)
    elif bits_flag:
        print("user entered bits")
        bits=int(num)
        Delta = (max_value - min_value) / pow(2,bits)
        levels=pow(2,bits)
    #print(f"delta: {Delta}")
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
        binary_representation = bin(value)[2:].zfill(int(bits))
        encoded_values_bits.append( binary_representation)

    print(f"Encoded values  {  encoded_values_bits}")
    visualize_discrete_signal(sample_indices1, quantized_values)

    indices=[]
    for value in encoded_values:
        indices.append(value + 1)
    print(f"indices  {indices}")
    #QuantizationTest2('Quan2_out.txt', indices, encoded_values_bits, quantized_values, quantized_error)
    QuantizationTest1('Quan1_out.txt', encoded_values_bits, quantized_values)


# file_path = 'input_Signal_DFT.txt'
# sample_indices, sample_values = read_signal(file_path)
# file_path2 = 'Output_Signal_DFT_A,Phase.txt'
# sample_indices2, sample_values2 = read_signal(file_path2)


# requirement 2 visualize the signals

def read_sample_values(file_path):
    try:
        with open(file_path, 'r') as file:
            sample_values = []

            for line in file:
                sample_value = float(line.strip())
                sample_values.append(sample_value)

        return sample_values
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read sample values: {e}")
        return []

def read_signal(file_path):
    try:
        with open(file_path, 'r') as file:
            N = int(file.readline().strip())  # Number of samples
            sample_indices = []
            sample_values = []

            for _ in range(N):
                line = file.readline().strip()
                sample_index, sample_value = map(float, line.split())
                sample_indices.append(int(sample_index))
                sample_values.append(sample_value)
        return sample_indices, sample_values
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read signal: {e}")
        return [], []
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

def calculate_correlation():
    # Use global variables instead of reading from paths
    signalindecies, signalvalues = sample_indices1, sample_values1
    signalindecies2, signalvalues2 = sample_indices2, sample_values2

    # Check if signals are empty
    if not signalindecies or not signalindecies2:
        raise ValueError("One or both signals are empty. Please ensure valid signals are loaded.")

    float_point = 8
    N = len(signalindecies)

    # Check for zero-length signals
    if N == 0:
        raise ValueError("Signal length is zero. Please provide valid signals.")

    result = []

    #calculate p12_denominator
    x1_square = [i ** 2 for i in signalvalues]
    x2_square = [i ** 2 for i in signalvalues2]
    p12_denominator = math.sqrt((sum(x1_square) * sum(x2_square))) / N
    p12_denominator = round(p12_denominator, float_point)

    # from r1 to r_end
    for i in range(1, N + 1):
        signal2_shifted = signalvalues2[i:] + signalvalues2[0:N - (N - i)]
        r = round(sum([signalvalues[i] * signal2_shifted[i] for i in range(N)]) / N, float_point)
        p = round(r / p12_denominator, float_point)
        result.append(p)

    # r0 == r_end
    result = [result[N - 1]] + result
    result = result[:N]
    indices = list(range(N))
    print("indices :",indices)
    print("results :", result)
    #Compare_Signals(r"Correlation Task Files/Point1 Correlation/CorrOutput.txt", indices, result)
    return indices, result

def compute_time_delay_with_correlation(sampling_rate):
    # Use the custom correlation function
    indices, correlation_result = calculate_correlation()

    # Find the index of the maximum correlation value
    max_correlation_index = correlation_result.index(max(correlation_result))

    # Compute the lag (in samples)
    lag = indices[max_correlation_index]

    # Convert lag to time delay (in seconds)
    time_delay = lag / sampling_rate
    return time_delay

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

def read_File2(file_path):
    try:
        with open(file_path, 'r') as file:
            N = int(file.readline().strip())  # Number of samples
            sample_indices = []
            sample_values = []

            for _ in range(N):
                line = file.readline().strip()
                sample_index, sample_value = map(float, line.split())
                sample_indices.append(sample_index)
                sample_values.append(sample_value)
        return sample_indices, sample_values
    except Exception as e:
        messagebox.showerror("Error", f"Failed to read signal: {e}")
        return [], []
def dft(sample_values):
    N = len(sample_values)
    dft_result = []
    amplitude = np.zeros(N)
    phaseShift = np.zeros(N)

    for k in range(N):  # Outer loop X[k]
        X_k = 0
        for n in range(N):  # Inner loop summation
            X_k += sample_values[n] * np.exp(-2j * np.pi * k * n / N)  # DFT
        dft_result.append(X_k)
    for i in range(N):
        amplitude[i] = np.sqrt((dft_result[i].real ** 2) + (dft_result[i].imag ** 2))  # square root (a^2 + b^2)
        phaseShift[i] = np.arctan2(dft_result[i].imag, dft_result[i].real)  # shift tan (y/x)

    sample_indices, sample_values = read_File2('Output_Signal_DFT_A,Phase.txt')
    result = signal_compare.SignalComapreAmplitude(amplitude, sample_indices)
    if result:
        print("Test passed successfully amplitude")
    else:
        print("Error: Signal amplitude comparison failed.")
    # print("test amplitude",sample_indices)
    result2 = signal_compare.SignalComaprePhaseShift(phaseShift,sample_values)
    if result2:
        print("Test passed successfully phase shift")
    else:
        print("Error: Signal amplitude comparison failed.")
    # print("test phase", sample_values)
    return np.array(dft_result), amplitude, phaseShift

def idft(amplitudes, phases):
    N = len(amplitudes)
    X_k = np.zeros(N, dtype=complex)

    # get X[k] from amplitude and phase shift
    for k in range(N):
        # X[k] = amplitude[k] * exp(j * phase[k])
        X_k[k] = amplitudes[k] * (np.cos(phases[k]) + 1j * np.sin(phases[k]))

    # calculate IDFT
    x_n = np.zeros(N, dtype=complex)
    for n in range(N):
        for k in range(N):
            x_n[n] += X_k[k] * np.exp(2j * np.pi * k * n / N)  # IDFT
        x_n[n] /= N
    x_n = np.real(x_n)
    x_n2= [round(signal) for signal in x_n.tolist()]
    print(x_n2)
    sample_indices, sample_values = read_File2('Output_Signal_IDFT.txt')
    result2 = signal_compare.SignalComapreAmplitude(x_n2, sample_values)
    if result2:
        print("Test passed successfully")
    else:
        print("Error: Signal amplitude comparison failed.")
    return x_n2

# GUI Functions

def load_signal():
    file_path = filedialog.askopenfilename()
    if file_path:
        global sample_indices1, sample_values1
        sample_indices1, sample_values1 = read_signal(file_path)
        visualize_continuous_signal(sample_indices1, sample_values1)
    else:
        messagebox.showerror("Error", "No file selected")

def load_classify_signal():
    file_path = filedialog.askopenfilename()
    if file_path:
        global sample_values1
        sample_values1 = read_sample_values(file_path)
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

def apply_dft():
    if 'sample_values1' not in globals():
        messagebox.showerror("Error", "No signal loaded for DFT")
        return
    N = len(sample_values1)
    _, amplitude, phase_shift = dft(sample_values1)
    if amplitude is not None and phase_shift is not None:
        print("Amplitude:", amplitude)
        print("Phase Shift:", phase_shift)
    sampling_frequency = int(entry_const.get())
    frequency = np.fft.fftfreq(N, d=1 / sampling_frequency)
    visualize_discrete_signal(frequency,amplitude)
    visualize_discrete_signal(frequency,phase_shift)

def apply_idft():
    if 'sample_values1' and 'sample_indices1' not in globals():
        messagebox.showerror("Error", "No signal loaded for IDFT")
        return
    N = len(sample_values1)
    values = idft(sample_indices1,sample_values1)
    timeline = np.arange(N)
    visualize_discrete_signal(timeline,values)

def apply_correlation():
    if 'sample_indices1' in globals() and 'sample_values1' in globals() and \
            'sample_indices2' in globals() and 'sample_values2' in globals():
        # Call the calculate_correlation function with loaded signals
        indices, result = calculate_correlation()
        # Visualize the correlation output
        visualize_continuous_signal(indices, result)
    else:
        messagebox.showerror("Error", "Please load both signals before applying correlation")

def apply_correlation_with_delay():
    if not ('sample_values1' in globals() and 'sample_values2' in globals()):
        messagebox.showerror("Error", "Please load both signals before applying correlation.")
        return

    try:
        # Get the sampling rate from user input
        sampling_rate = float(fs_entry.get())

        # Compute the time delay using the loaded signals
        time_delay = compute_time_delay_with_correlation(sampling_rate)
        messagebox.showinfo("Time Delay", f"The time delay is {time_delay:.4f} seconds.")
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
def apply_classify_signal():
    classif, result = classify_signal()
    if 'sample_indices1' in globals() and 'sample_values1' in globals() and \
            'sample_indices2' in globals() and 'sample_values2' in globals():
        # Call the calculate_correlation function with loaded signals
        messagebox.showinfo("classification", f"signal belongs to class {classif}.")

    else:
        messagebox.showerror("Error", "Please load both signals before applying correlation")

# Tkinter GUI Setup
root = Tk()
root.title("Signal Processing")
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Set the window size to be as wide as the screen but with a fixed height
window_width = int(screen_width*0.7)
window_height = int(screen_height * 0.7)  # Half the height of the screen

# Position the window at the center of the screen
x_position = 0
y_position = (screen_height - window_height) // 2

root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")
# Load buttons
Button(root, text="Load Signal 1", command=load_signal).grid(row=0, column=0, padx=10, pady=10)
Button(root, text="Load Signal 2", command=load_second_signal).grid(row=0, column=1, padx=10, pady=10)
Button(root, text="Moving Average", command=lambda: moving_average(int(entry_const.get()))).grid(row=0, column=4, padx=10, pady=10)
Button(root, text="Sharpen Signal", command=sharpen_signal).grid(row=1, column=4, padx=10, pady=10)
Button(root, text="Convolve", command=convolve).grid(row=2, column=4, padx=10, pady=10)
Button(root, text="read classify signal", command=load_classify_signal).grid(row=0, column=2, padx=10, pady=10)


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
Button(root, text="Apply DFT", command=apply_dft).grid(row=7, column=2, padx=10, pady=10)
Button(root, text="Apply IDFT", command=apply_idft).grid(row=8, column=2, padx=10, pady=10)
Button(root, text="Apply correlation", command=apply_correlation).grid(row=9, column=2, padx=10, pady=10)
Button(root, text="compute delay", command=apply_correlation_with_delay).grid(row=10, column=2, padx=10, pady=10)
Button(root, text="classify signal", command=apply_classify_signal).grid(row=11, column=2, padx=10, pady=10)

# Filter Specification Inputs
Label(root, text="Filter Type:").grid(row=0, column=5, padx=5, pady=5)
filter_type = ttk.Combobox(root, values=["lowpass", "highpass", "bandpass", "bandstop"])
filter_type.grid(row=1, column=5, padx=5, pady=5)
Label(root, text="Sampling Frequency (FS):").grid(row=2, column=5, padx=5, pady=5)
fs_entry = Entry(root)
fs_entry.grid(row=3, column=5, padx=5, pady=5)
Label(root, text="Cutoff Frequency (FC):").grid(row=4, column=5, padx=5, pady=5)
fc_entry = Entry(root)
fc_entry.grid(row=5, column=5, padx=5, pady=5)
Label(root, text="Transition Band (Hz):").grid(row=6, column=5, padx=5, pady=5)
transition_band_entry = Entry(root)
transition_band_entry.grid(row=7, column=5, padx=5, pady=5)
Label(root, text="Stopband Attenuation (dB):").grid(row=8, column=5, padx=5, pady=5)
attenuation_entry = Entry(root)
attenuation_entry.grid(row=9, column=5, padx=5, pady=5)
Button(text="Generate Filter", command=generate_filter).grid(row=10, column=5, columnspan=2,pady=10)


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
def CompareSignal(file_name, Your_EncodedValues, Your_Values):
    expectedIndices=[]
    expectedValues=[]
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
                V2=int(L[0])
                V3=float(L[1])
                expectedIndices.append(V2)
                expectedValues.append(V3)
                line = f.readline()
            else:
                break
    if( (len(Your_EncodedValues)!=len(expectedIndices)) or (len(Your_Values) != len(expectedValues))):
        messagebox.showerror("Test Case Failed",
                             " Test case failed, your signal has a different length from the expected one.")
        print(Your_EncodedValues)
        print(expectedIndices)
        return
    for i in range(len(Your_EncodedValues)):
        if(Your_EncodedValues[i]!=expectedIndices[i]):

            messagebox.showerror("Test Case Failed",
                                 " Test case failed, your Values are different from the expected one.")
            return
    for i in range(len(expectedValues)):
        if abs(Your_Values[i] - expectedValues[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Test Case Failed",
                                 "Test case failed, your Values have different values from the expected one.")
            return
    messagebox.showinfo("Test Case Passed", " Test case passed successfully.")
def Compare_Signals(file_name,Your_indices,Your_samples):
    expected_indices=[]
    expected_samples=[]
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
                V1=int(L[0])
                V2=float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    print("Current Output Test file is: ")
    print(file_name)
    print("\n")
    if (len(expected_samples)!=len(Your_samples)) and (len(expected_indices)!=len(Your_indices)):
        print("Test case failed, your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if(Your_indices[i]!=expected_indices[i]):
            print("Test case failed, your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            print("Test case failed, your signal have different values from the expected one")
            return
    print("Test case passed successfully")

root.mainloop()