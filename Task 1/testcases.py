import tkinter as tk
from tkinter import messagebox
def ReadSignalFile(file_name):
    expected_indices = []
    expected_samples = []
    with open(file_name, 'r') as f:
        line = f.readline()
        line = f.readline()
        line = f.readline()
        line = f.readline()
        while line:
            # process line
            L = line.strip()
            if len(L.split(' ')) == 2:
                L = line.split(' ')
                V1 = int(L[0])
                V2 = float(L[1])
                expected_indices.append(V1)
                expected_samples.append(V2)
                line = f.readline()
            else:
                break
    return expected_indices, expected_samples


def AddSignalSamplesAreEqual(userFirstSignal, userSecondSignal, Your_indices, Your_samples):
    if (userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal2.txt'):
        file_name = "add.txt"  # write here the path of the add output file
    expected_indices, expected_samples = ReadSignalFile(file_name)
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        messagebox.showerror("Addition Test case" ," Addition Test case failed: your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            messagebox.showerror("Addition Test case","Addition Test case failed: your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Addition Test case","Addition Test case failed: your signal have different values from the expected one")
            return
    messagebox.showinfo("Addition Test case","Addition Test case passed successfully")


# AddSignalSamplesAreEqual("Signal1.txt", "Signal2.txt",indicies,samples) # call this function with your computed indicies and samples


def SubSignalSamplesAreEqual(userFirstSignal, userSecondSignal, Your_indices, Your_samples):
    if (userFirstSignal == 'Signal1.txt' and userSecondSignal == 'Signal2.txt'):
        file_name = "subtract.txt"  # write here the path of the subtract output file

    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        messagebox.showerror("Subtraction Test case","Subtraction Test case failed: your signal have different length from the expected one")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            messagebox.showerror(
                "Subtraction Test case","Subtraction Test case failed: your signal have different indicies from the expected one")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            messagebox.showerror(
                "Subtraction Test case","Subtraction Test case failed: your signal have different values from the expected one")
            return
    messagebox.showinfo("Subtraction Test case","Subtraction Test case passed successfully")


# SubSignalSamplesAreEqual("Signal1.txt", "Signal2.txt",indicies,samples)  # call this function with your computed indicies and samples


# %%


def MultiplySignalByConst(User_Const, Your_indices, Your_samples):
    if (User_Const == 5):
        file_name = "mul5.txt"  # write here the path of the mul5 output file

    expected_indices, expected_samples = ReadSignalFile(file_name)
    if (len(expected_samples) != len(Your_samples)) and (len(expected_indices) != len(Your_indices)):
        messagebox.showerror("Multiplication Test Case", f"Multiply by {User_Const} Test case failed: different lengths.")
        return
    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            messagebox.showerror("Multiplication Test Case",
                                 f"Multiply by {User_Const} Test case failed: different indices.")
            return
    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Multiplication Test Case",
                                 f"Multiply by {User_Const} Test case failed: different values.")
            return
    messagebox.showinfo("Multiplication Test Case", f"Multiply by {User_Const} Test case passed successfully.")


# MultiplySignalByConst(5,indicies, samples)# call this function with your computed indicies and samples


def ShiftSignalByConst(Shift_value, Your_indices, Your_samples):
    if (Shift_value == 3):  # x(n+k)
        file_name = "advance3.txt"  # write here the path of advance3 output file
    elif (Shift_value == -3):  # x(n-k)
        file_name = "delay3.txt"  # write here the path of delay3 output file

    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        messagebox.showerror("Error", f"Shift by {Shift_value}: Test case failed. Your signal has different lengths.")
        return

    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            messagebox.showerror("Error",
                                 f"Shift by {Shift_value}: Test case failed. Your signal has different indices.")
            return

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Error",
                                 f"Shift by {Shift_value}: Test case failed. Your signal has different values.")
            return

    messagebox.showinfo("Success", f"Shift by {Shift_value}: Test case passed successfully.")
# ShiftSignalByConst(3, indicies, samples)  # call this function with your computed indicies and samples

def Folding(Your_indices, Your_samples):
    file_name = "folding.txt"  # write here the path of the folding output file
    expected_indices, expected_samples = ReadSignalFile(file_name)

    if (len(expected_samples) != len(Your_samples)) or (len(expected_indices) != len(Your_indices)):
        messagebox.showerror("Error", "Folding Test case failed. Your signal has different lengths.")
        return

    for i in range(len(Your_indices)):
        if (Your_indices[i] != expected_indices[i]):
            messagebox.showerror("Error", "Folding Test case failed. Your signal has different indices.")
            return

    for i in range(len(expected_samples)):
        if abs(Your_samples[i] - expected_samples[i]) < 0.01:
            continue
        else:
            messagebox.showerror("Error", "Folding Test case failed. Your signal has different values.")
            return

    messagebox.showinfo("Success", "Folding Test case passed successfully.")
# Folding(indicies, samples)  # call this function with your computed indicies and samples