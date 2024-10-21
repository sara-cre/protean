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

def plot_metrics(acc_file_name, f1_file_name, macro_acc_file_name, macro_f1_file_name, precision_file_name, output_file_name):

    # Reading accuracies from the file
    with open(acc_file_name, 'r') as file:
        # Assume the file contains one line with a list of accuracies
        accuracies = ast.literal_eval(file.readline().strip())
    # Reading macro accuracies from the file
    with open(macro_acc_file_name, 'r') as file:
        # Assume the file contains one line with a list of accuracies
        macro_accuracies = ast.literal_eval(file.readline().strip())
    # Reading F1 scores from the file
    with open(f1_file_name, 'r') as file:
        # Assume the file contains one line with a list of F1 scores
        f1_scores = ast.literal_eval(file.readline().strip())
    # Reading macro F1 scores from the file
    with open(macro_f1_file_name, 'r') as file:
        # Assume the file contains one line with a list of F1 scores
        macro_f1_scores = ast.literal_eval(file.readline().strip())
    with open(precision_file_name, 'r') as file:
        # Assume the file contains one line with a list of F1 scores
        precision_scores = ast.literal_eval(file.readline().strip())
    # Generating round numbers
    rounds = list(range(1, len(accuracies) + 1))
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.plot(rounds, f1_scores, marker='o', linestyle='-', color='r', label='F1 Score')
    plt.plot(rounds, macro_accuracies, marker='o', linestyle='-', color='g', label='Macro Accuracy')
    #plt.plot(rounds, macro_f1_scores, marker='o', linestyle='-', color='m', label='Macro F1 Score')
    plt.plot(rounds, precision_scores, marker='o', linestyle='-', color='y', label='Precision Score')
    plt.ylim(0, 1)
    plt.xlabel('Round')
    plt.ylabel('Metric')
    plt.title('Metrics by Round')
    plt.xticks(rounds)  # Ensuring x-axis has integer ticks for each round
    plt.legend()  
    plt.grid(True)
    #plt.show()
    # creating the output file
    
    plt.savefig(output_file_name)
    
def plot_accuracy_comparison(args, before_file, after_file):
    # Load data from the files
    with open(before_file, 'r') as bf:
        before_data = eval(bf.read().strip())
        print(before_data)
        
    with open(after_file, 'r') as af:
        after_data = eval(af.read().strip())
        print(after_data)
    
    # Ensure that the data is in list of lists form
    if isinstance(before_data, list) and isinstance(before_data[0], float):
        before_data = [before_data]  # Wrapping in a list if it's a single user
    if isinstance(after_data, list) and isinstance(after_data[0], float):
        after_data = [after_data]  # Wrapping in a list if it's a single user
    
    num_users = len(before_data)
    print(f'Number of users: {num_users}')
    
    for user_index in range(num_users):
        before_acc = before_data[user_index]
        print(f'User {user_index + 1} - Before FL: {before_acc}')
        after_acc = after_data[user_index]
        print(f'User {user_index + 1} - After FL: {after_acc}')
        
        # Ensure both before_acc and after_acc have the same length
        max_length = max(len(before_acc), len(after_acc))
        
        # Pad with zeros if lengths are not equal
        before_acc = np.pad(before_acc, (0, max_length - len(before_acc)), 'constant')
        after_acc = np.pad(after_acc, (0, max_length - len(after_acc)), 'constant')
        
        num_classes = max_length
        
        # Set up the plot
        fig, ax = plt.subplots()
        index = np.arange(num_classes)
        bar_width = 0.35
        
        # Plot data
        bars_before = ax.bar(index, before_acc, bar_width, label='Before FL')
        bars_after = ax.bar(index + bar_width, after_acc, bar_width, label='After FL')
        
        # Add labels, title, and legend
        ax.set_xlabel('Classes')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'User {user_index + 1} - Accuracy Before and After FL')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([f'Class {i+1}' for i in range(num_classes)])
        ax.legend()
        
        # Display the plot
        plt.tight_layout()
        if args.attack_type == 'none':
            file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
        else:
            file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+ '/' + args.alg + '/'

        if not os.path.exists(file_folder):
            os.makedirs(file_folder)
        file_ext = 'acc_comparaision_'+'user'+ str(user_index) + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
        output_file_name = file_folder + file_ext + '.pdf'
        plt.savefig(output_file_name)



def plot_accuracy_comparison_global(args, before_file, after_file):
    # Load data from the files
    with open(before_file, 'r') as bf:
        before_data = eval(bf.read().strip())
        print(before_data)
        
    with open(after_file, 'r') as af:
        after_data = eval(af.read().strip())
        print(after_data)
    
    # Ensure that the data is in list of lists form
    if isinstance(before_data, list) and isinstance(before_data[0], float):
        before_data = [before_data]  # Wrapping in a list if it's a single user
    if isinstance(after_data, list) and isinstance(after_data[0], float):
        after_data = [after_data]  # Wrapping in a list if it's a single user
    
    num_users = len(before_data)
    print(f'Number of users: {num_users}')
    
    for user_index in range(num_users):
        before_acc = before_data[user_index]
        print(f'User {user_index + 1} - Before FL: {before_acc}')
        after_acc = after_data[0]
        print(f'User {user_index + 1} - After FL: {after_acc}')
        
        # Ensure both before_acc and after_acc have the same length
        max_length = max(len(before_acc), len(after_acc))
        
        # Pad with zeros if lengths are not equal
        before_acc = np.pad(before_acc, (0, max_length - len(before_acc)), 'constant')
        after_acc = np.pad(after_acc, (0, max_length - len(after_acc)), 'constant')
        
        num_classes = max_length
        
        # Set up the plot
        fig, ax = plt.subplots()
        index = np.arange(num_classes)
        bar_width = 0.35
        
        # Plot data
        bars_before = ax.bar(index, before_acc, bar_width, label='Before FL')
        bars_after = ax.bar(index + bar_width, after_acc, bar_width, label='After FL')
        
        # Add labels, title, and legend
        ax.set_xlabel('Classes')
        ax.set_ylabel('Accuracy')
        ax.set_title(f'User {user_index + 1} - Accuracy Before and After FL')
        ax.set_xticks(index + bar_width / 2)
        ax.set_xticklabels([f'Class {i+1}' for i in range(num_classes)])
        ax.legend()
        
        # Display the plot
        plt.tight_layout()
        if args.attack_type == 'none':
            file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
        else:
            file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+ '/' + args.alg + '/'
        file_ext = 'acc_comparaision_global_'+'user'+ str(user_index) + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
        output_file_name = file_folder + file_ext + '.pdf'
        plt.savefig(output_file_name)


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
