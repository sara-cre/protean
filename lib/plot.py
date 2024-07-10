import matplotlib.pyplot as plt
import ast
import os
import numpy as np

def plot_fl_accuracies(file_path):
    # Reading accuracies from the file
    with open(file_path, 'r') as file:
        # Assume the file contains one line with a list of accuracies
        accuracies = ast.literal_eval(file.readline().strip())

    # Generating round numbers
    rounds = list(range(1, len(accuracies) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Round')
    plt.xticks(rounds)  # Ensuring x-axis has integer ticks for each round
    plt.grid(True)
    #plt.show()
    # Extracting the base name of the file path and saving the plot
    base_name = os.path.basename(file_path)
    save_path = os.path.join("../save", f"figure_{base_name}.pdf")
    plt.savefig(save_path)


def plot_fedproto_accuracies(file_path):
    # Reading accuracies from the file
    accuracies = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse the line as a list of floats
            round_accuracies = ast.literal_eval(line.strip())
            # Calculate the mean accuracy for the round
            mean_accuracy = np.mean(round_accuracies)
            accuracies.append(mean_accuracy)

    # Generating round numbers
    rounds = list(range(1, len(accuracies) + 1))

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b')
    plt.ylim(0, 1)  # Setting y-axis limits from 0 to 1 (0% to 100%)
    plt.xlabel('Round')
    plt.ylabel('Mean Accuracy')
    plt.title('Mean Accuracy by Round')
    plt.xticks(rounds)  # Ensuring x-axis has integer ticks for each round
    plt.grid(True)
    
    # Extracting the base name of the file path and saving the plot
    base_name = os.path.basename(file_path)
    save_path = os.path.join("../save", f"mean_figure_{base_name}.pdf")
    plt.savefig(save_path)
    #plt.show()

# Replace with the actual path to your file
file_path = '../save/accuracies_FLxiiotid_4w10000s1e_2u.txt'
plot_fl_accuracies(file_path)
file_path2 = '../save/accuracies_FedProto_wxiiotid_4w10000s1e_2u.txt'
plot_fedproto_accuracies(file_path2)
