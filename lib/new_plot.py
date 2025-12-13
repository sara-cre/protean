# import matplotlib.pyplot as plt
# import ast
# import os
# import numpy as np

# def plot_accuracy_comparison_global(args, before_file, after_file):
#     # Load data from the files
#     with open(before_file, 'r') as bf:
#         before_data = eval(bf.read().strip())
#         print(before_data)
        
#     with open(after_file, 'r') as af:
#         after_data = eval(af.read().strip())
#         print(after_data)
    
#     # Ensure that the data is in list of lists form
#     if isinstance(before_data, list) and isinstance(before_data[0], float):
#         before_data = [before_data]  # Wrapping in a list if it's a single user
#     if isinstance(after_data, list) and isinstance(after_data[0], float):
#         after_data = [after_data]  # Wrapping in a list if it's a single user
    
#     num_users = len(before_data)
#     print(f'Number of users: {num_users}')
    
#     for user_index in range(num_users):
#         before_acc = before_data[user_index]
#         print(f'User {user_index + 1} - FedProx: {before_acc}')
#         after_acc = after_data[0]
#         print(f'User {user_index + 1} - PROTEAN: {after_acc}')
        
#         # Ensure both before_acc and after_acc have the same length
#         max_length = max(len(before_acc), len(after_acc))
        
#         # Pad with zeros if lengths are not equal
#         before_acc = np.pad(before_acc, (0, max_length - len(before_acc)), 'constant')
#         after_acc = np.pad(after_acc, (0, max_length - len(after_acc)), 'constant')
        
#         num_classes = max_length
        
#         # Set up the plot
#         fig, ax = plt.subplots()
#         index = np.arange(num_classes)
#         bar_width = 0.35
        
#         # Plot data
#         bars_before = ax.bar(index, before_acc, bar_width, label='FedProx')
#         bars_after = ax.bar(index + bar_width, after_acc, bar_width, label='PROTEAN')
        
#         # Add labels, title, and legend
#         ax.set_xlabel('Classes')
#         ax.set_ylabel('Accuracy')
#         #ax.set_title(f'User {user_index + 1} ')#- Accuracy Before and After FL')
#         ax.set_xticks(index + bar_width / 2)
#         ax.set_xticklabels([f'Class {i+1}' for i in range(num_classes)])
#         ax.legend()
        
#         # Display the plot
#         plt.tight_layout()
#         file_folder = '../new_save/'
#         # if args.attack_type == 'none':
#         #     file_folder = '../save2/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/' + args.alg + '/'
#         # else:
#         #     file_folder = '../save_attack/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '/_num_attackers'+str(args.num_attackers)+'_ratio'+str(args.flip_ratio)+ '/' + args.alg + '/'
#         file_ext = 'acc_comparaision_global_'+'user'+ str(user_index) + 'data_' + args.dataset + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) #+ '_timestamp' + str(time.time())
#         output_file_name = file_folder + file_ext + '.pdf'
#         plt.savefig(output_file_name)



# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--attack_type', type=str, default='none', help='Type of attack')
#     parser.add_argument('--alpha', type=float, default=0.75, help='Alpha value')
#     parser.add_argument('--num_users', type=int, default=10, help='Number of users')
#     parser.add_argument('--num_attackers', type=int, default=2, help='Number of attackers')
#     parser.add_argument('--flip_ratio', type=float, default=0.1, help='Flip ratio for attack')
#     parser.add_argument('--alg', type=str, default='FedAvg', help='Federated learning algorithm')
#     parser.add_argument('--dataset', type=str, default='xiiotid', help='Dataset name')
#     args = parser.parse_args()
#     args.alg = 'fedprox'  # Force algorithm to FedProx for before data
#     folder = '../save2_paper/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '_rounds10/'+ args.alg + '/'
    
#     before_file = folder+'acc_byclass_data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users) + '.txt'
#     args.alg = 'fedproto'  # Force algorithm to PROTEAN for after data
#     folder = '../save2_paper/_alpha' + str(args.alpha) +  '_num_users' + str(args.num_users) + '_rounds10/'+ args.alg + '/'
    
#     after_file =  folder+'acc_byclient_byclass_data_' + args.dataset + '_alpha' + str(args.alpha) + '_alg' + args.alg + '_num_users' + str(args.num_users) + '.txt'
    
#     plot_accuracy_comparison_global(args, before_file, after_file)



# import os
# import ast
# import numpy as np
# import matplotlib.pyplot as plt

# def _load_as_2d(path):
#     with open(path, "r") as f:
#         data = ast.literal_eval(f.read().strip())
#     if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
#         data = [data]
#     return [[float(x) for x in row] for row in data]

# def _pick_global_vector(data_2d, mode="mean"):
#     arr = np.array(data_2d, dtype=float)
#     if arr.ndim == 1:
#         return arr
#     if mode == "first":
#         return arr[0]
#     return arr.mean(axis=0)

# def plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                    global_mode="mean", out_dir="../new_save/"):

#     beforefl_2d = _load_as_2d(beforefl_file)   # per-client (Local) -> num_users x num_classes
#     fedavg_2d   = _load_as_2d(fedavg_file)
#     fedprox_2d  = _load_as_2d(fedprox_file)
#     fedproto_2d = _load_as_2d(fedproto_file)

#     num_users = len(beforefl_2d)

#     fedavg_g   = _pick_global_vector(fedavg_2d,   mode=global_mode)
#     fedprox_g  = _pick_global_vector(fedprox_2d,  mode=global_mode)
#     fedproto_g = _pick_global_vector(fedproto_2d, mode=global_mode)

#     os.makedirs(out_dir, exist_ok=True)

#     for user_index in range(num_users):
#         local = np.array(beforefl_2d[user_index], dtype=float)

#         max_len = max(len(local), len(fedavg_g), len(fedprox_g), len(fedproto_g))
#         local    = np.pad(local,     (0, max_len - len(local)),     constant_values=0.0)
#         fedavg   = np.pad(fedavg_g,  (0, max_len - len(fedavg_g)),  constant_values=0.0)
#         fedprox  = np.pad(fedprox_g, (0, max_len - len(fedprox_g)), constant_values=0.0)
#         fedproto = np.pad(fedproto_g,(0, max_len - len(fedproto_g)),constant_values=0.0)

#         index = np.arange(max_len)
#         w = 0.20

#         fig, ax = plt.subplots()
#         ax.bar(index + 0*w, local,    w, label="Local")
#         ax.bar(index + 1*w, fedavg,   w, label="Cerberus")     # FedAvg in files, Cerberus in legend
#         ax.bar(index + 2*w, fedprox,  w, label="FedProx")
#         ax.bar(index + 3*w, fedproto, w, label="PROTEAN")

#         ax.set_xlabel("Classes")
#         ax.set_ylabel("Accuracy")
#         ax.set_xticks(index + 1.5*w)
#         ax.set_xticklabels([f"Class {i+1}" for i in range(max_len)])
#         ax.set_ylim(0, 1.0)
#         ax.legend()

#         plt.tight_layout()
#         out_name = (
#             f"acc_comparison_4algos_user{user_index+1}_"
#             f"data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
#         )
#         plt.savefig(os.path.join(out_dir, out_name))
#         plt.close(fig)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--alpha", type=float, default=0.5)
#     parser.add_argument("--num_users", type=int, default=10)
#     parser.add_argument("--dataset", type=str, default="xiiotid")
#     args = parser.parse_args()

#     base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}_rounds10/"

#     # Files are named by your pipeline (beforefl/fedavg/fedprox/fedproto)
#     beforefl_file = base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
#     fedavg_file   = base + "fedavg/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt"
#     fedprox_file  = base + "fedprox/"  + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt"
#     fedproto_file = base + "fedproto/" + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt"

#     plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                    global_mode="mean", out_dir="../new_save/")



# import os
# import ast
# import numpy as np
# import matplotlib.pyplot as plt

# def _load_as_2d(path):
#     with open(path, "r") as f:
#         data = ast.literal_eval(f.read().strip())
#     if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
#         data = [data]
#     return [[float(x) for x in row] for row in data]

# def _pick_global_vector(data_2d, mode="mean"):
#     arr = np.array(data_2d, dtype=float)
#     if arr.ndim == 1:
#         return arr
#     if mode == "first":
#         return arr[0]
#     return arr.mean(axis=0)

# def plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                    global_mode="mean", out_dir="../new_save/"):

#     beforefl_2d = _load_as_2d(beforefl_file)   # per-client (Local) -> num_users x num_classes
#     fedavg_2d   = _load_as_2d(fedavg_file)
#     fedprox_2d  = _load_as_2d(fedprox_file)
#     fedproto_2d = _load_as_2d(fedproto_file)

#     num_users = len(beforefl_2d)

#     fedavg_g   = _pick_global_vector(fedavg_2d,   mode=global_mode)
#     fedprox_g  = _pick_global_vector(fedprox_2d,  mode=global_mode)
#     fedproto_g = _pick_global_vector(fedproto_2d, mode=global_mode)

#     os.makedirs(out_dir, exist_ok=True)

#     for user_index in range(num_users):
#         local = np.array(beforefl_2d[user_index], dtype=float)

#         max_len = max(len(local), len(fedavg_g), len(fedprox_g), len(fedproto_g))
#         local    = np.pad(local,     (0, max_len - len(local)),     constant_values=0.0)
#         fedavg   = np.pad(fedavg_g,  (0, max_len - len(fedavg_g)),  constant_values=0.0)
#         fedprox  = np.pad(fedprox_g, (0, max_len - len(fedprox_g)), constant_values=0.0)
#         fedproto = np.pad(fedproto_g,(0, max_len - len(fedproto_g)),constant_values=0.0)

#         index = np.arange(max_len)
#         w = 0.20

#         fig, ax = plt.subplots()
#         ax.bar(index + 0*w, local,    w)# , label="Local")
#         ax.bar(index + 1*w, fedavg,   w)#, label="Cerberus")     # FedAvg in files, Cerberus in legend
#         ax.bar(index + 2*w, fedprox,  w)#, label="FedProx")
#         ax.bar(index + 3*w, fedproto, w)#, label="PROTEAN")

#         # ax.set_xlabel("Classes")
#         # ax.set_ylabel("Accuracy")
#         ax.set_xlabel("Classes", fontsize=28)
#         ax.set_ylabel("Accuracy", fontsize=28)

#         ax.set_xticks(index + 1.5*w)
#         ax.set_xticklabels([f"Class {i+1}" for i in range(max_len)])
#         ax.set_ylim(0, 1.0)
#         ax.legend()

#         plt.tight_layout()
#         out_name = (
#             f"acc_comparison_4algos_user{user_index+1}_"
#             f"data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
#         )
#         plt.savefig(os.path.join(out_dir, out_name))
#         plt.close(fig)

# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--alpha", type=float, default=0.25)
#     parser.add_argument("--num_users", type=int, default=10)
#     parser.add_argument("--dataset", type=str, default="5gnidd")
#     args = parser.parse_args()

#     base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}_weighted/"
#     #_weighted/"
#     #_rounds10/"

#     # Files are named by your pipeline (beforefl/fedavg/fedprox/fedproto)
#     beforefl_file = base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
#     fedavg_file   = base + "fedavg/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt"
#     fedprox_file  = base + "fedprox/"  + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt"
#     fedproto_file = base + "fedproto/" + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt"

#     plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                    global_mode="mean", out_dir="../new_save/")


import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt

def _load_obj(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read().strip())

def _as_2d(data):
    if isinstance(data, list) and len(data) and isinstance(data[0], (int, float)):
        return [list(map(float, data))]
    if isinstance(data, list) and len(data) and isinstance(data[0], list):
        return [list(map(float, row)) for row in data]
    raise ValueError("Unsupported format in file.")

def _pick_global_vector(data_2d, mode="mean"):
    arr = np.array(data_2d, dtype=float)
    if arr.ndim == 1:
        return arr
    if mode == "first":
        return arr[0]
    return arr.mean(axis=0)

import re
import ast

def _get_rare_classes(dist_file, k_rare=2):
    # ---- Try Python-list format first ----
    try:
        with open(dist_file, "r") as f:
            txt = f.read().strip()
        data = ast.literal_eval(txt)
        if isinstance(data, list):
            rare_by_user = []
            for cnts in data:
                present = [(idx, cnt) for idx, cnt in enumerate(cnts) if cnt > 0]
                least = sorted(present, key=lambda x: x[1])[:k_rare]
                rare_by_user.append([idx for idx, _ in least])
            return rare_by_user
    except Exception:
        pass

    # ---- Text-log format (YOUR EXACT FORMAT) ----
    # Examples:
    # "User 0:"
    # "  Class 4: 1001 instances"
    re_user  = re.compile(r"^\s*User\s+(\d+)\s*:\s*$")
    re_class = re.compile(r"^\s*Class\s+(\d+)\s*:\s*(\d+)\s*(?:instances)?\s*$")

    rare_by_user = []
    cur_counts = {}

    def finalize(counts):
        if not counts:
            return None
        present = [(cid, cnt) for cid, cnt in counts.items() if cnt > 0]
        least = sorted(present, key=lambda x: x[1])[:k_rare]
        return [cid for cid, _ in least]

    with open(dist_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if re_user.match(line):
                fin = finalize(cur_counts)
                if fin is not None:
                    rare_by_user.append(fin)
                cur_counts = {}
                continue

            m = re_class.match(line)
            if m:
                cid = int(m.group(1))
                cnt = int(m.group(2))
                cur_counts[cid] = cnt

    fin = finalize(cur_counts)
    if fin is not None:
        rare_by_user.append(fin)

    if not rare_by_user:
        raise ValueError(f"Could not parse distribution file: {dist_file}")

    return rare_by_user



# def plot_avg_rare_accuracy_4algos(args,
#                                  beforefl_file, fedavg_file, fedprox_file, fedproto_file, dist_file,
#                                  k_rare=2, global_mode="mean", out_dir="../new_save/"):

#     rare_by_user = _get_rare_classes(dist_file, k_rare=k_rare)
#     n_users = len(rare_by_user)

#     beforefl_2d = _as_2d(_load_obj(beforefl_file))[:n_users]

#     fedavg_g   = _pick_global_vector(_as_2d(_load_obj(fedavg_file)),  mode=global_mode)
#     fedprox_g  = _pick_global_vector(_as_2d(_load_obj(fedprox_file)), mode=global_mode)
#     fedproto_g = _pick_global_vector(_as_2d(_load_obj(fedproto_file)),mode=global_mode)

#     fedavg_mat   = [fedavg_g.tolist()]   * n_users
#     fedprox_mat  = [fedprox_g.tolist()]  * n_users
#     fedproto_mat = [fedproto_g.tolist()] * n_users

#     def avg_on_rare(mat):
#         return [float(np.mean([mat[u][c] for c in rare_by_user[u]])) for u in range(n_users)]

#     y_local = avg_on_rare(beforefl_2d)
#     y_avg   = avg_on_rare(fedavg_mat)
#     y_prox  = avg_on_rare(fedprox_mat)
#     y_proto = avg_on_rare(fedproto_mat)

#     os.makedirs(out_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(6.5, 3.5))
#     idx = np.arange(n_users)
#     w = 0.20

#     ax.bar(idx + 0*w, y_local, w)#, label="Local")
#     ax.bar(idx + 1*w, y_avg,   w)#, label="Cerberus")
#     ax.bar(idx + 2*w, y_prox,  w)#, label="FedProx")
#     ax.bar(idx + 3*w, y_proto, w)#, label="PROTEAN")

#     ax.set_xlabel("User", fontsize=28)
#     ax.set_ylabel("Avg Accuracy", fontsize=28)
#     ax.set_xticks(idx + 1.5*w)
#     ax.set_xticklabels([str(u) for u in idx])
#     ax.set_ylim(0, 1.0)
#     ax.legend()

#     plt.tight_layout()
#     out_name = f"rare_acc_comparison_4algos_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
#     plt.savefig(os.path.join(out_dir, out_name))
#     plt.close(fig)


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--alpha", type=float, default=0.5)
#     parser.add_argument("--num_users", type=int, default=10)
#     parser.add_argument("--dataset", type=str, default="xiiotid")
#     args = parser.parse_args()

#     # ---- SAME BASE AS YOUR 2nd CODE (accuracies) ----
#     base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}"
#     if args.dataset == "5gnidd":
#         base += "_weighted/"
#     else:
#         base += "_rounds10/"

#     beforefl_file = base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
#     fedavg_file   = base + "fedavg/"    + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt"
#     fedprox_file  = base + "fedprox/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt"
#     fedproto_file = base + "fedproto/"  + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt"

#     # ---- distribution EXACTLY like your first-code style ----
#     # dist_base = '../save2_paper/_alpha' + str(args.alpha) + '_num_users' + str(args.num_users)

#     # dist_file = dist_base + 'classes_distribution_data_' + args.dataset \
#     #     + '_alpha' + str(args.alpha) + '_num_users' + str(args.num_users) + '.txt'
#     dist_file = "../save2_paper/" +f"_alpha{args.alpha}_num_users{args.num_users}classes_distribution_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
#     #0.75_num_users20classes_distribution_data_5gnidd_alpha0.75_num_users20.txt

#     plot_avg_rare_accuracy_4algos(
#         args,
#         beforefl_file, fedavg_file, fedprox_file, fedproto_file, dist_file,
#         k_rare=2,
#         global_mode="mean",
#         out_dir="../new_save/"
#     )


# def plot_avg_rare_accuracy_3algos(args,
#                                  fedavg_file, fedprox_file, fedproto_file, dist_file,
#                                  k_rare=2, global_mode="mean", out_dir="../new_save/"):

#     rare_by_user = _get_rare_classes(dist_file, k_rare=k_rare)
#     n_users = len(rare_by_user)

#     fedavg_g   = _pick_global_vector(_as_2d(_load_obj(fedavg_file)),  mode=global_mode)
#     fedprox_g  = _pick_global_vector(_as_2d(_load_obj(fedprox_file)), mode=global_mode)
#     fedproto_g = _pick_global_vector(_as_2d(_load_obj(fedproto_file)),mode=global_mode)

#     fedavg_mat   = [fedavg_g.tolist()]   * n_users
#     fedprox_mat  = [fedprox_g.tolist()]  * n_users
#     fedproto_mat = [fedproto_g.tolist()] * n_users

#     def avg_on_rare(mat):
#         return [float(np.mean([mat[u][c] for c in rare_by_user[u]])) for u in range(n_users)]

#     y_avg   = avg_on_rare(fedavg_mat)
#     y_prox  = avg_on_rare(fedprox_mat)
#     y_proto = avg_on_rare(fedproto_mat)

#     os.makedirs(out_dir, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(6.5, 3.5))
#     idx = np.arange(n_users)
#     w = 0.25

#     ax.bar(idx + 0*w, y_avg,   w, color="tab:green")#,   label="Cerberus")
#     ax.bar(idx + 1*w, y_prox,  w,   color="tab:brown")#,              label="FedProx")   # keep default color
#     ax.bar(idx + 2*w, y_proto, w, color="tab:orange")#,label="PROTEAN")

#     ax.set_xlabel("User", fontsize=28)
#     ax.set_ylabel("Avg Accuracy", fontsize=28)
#     ax.set_xticks(idx + 1.0*w)
#     ax.set_xticklabels([str(u) for u in idx])
#     ax.set_ylim(0, 1.0)
#     ax.legend()

#     plt.tight_layout()
#     out_name = f"rare_acc_comparison_3algos_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
#     plt.savefig(os.path.join(out_dir, out_name))
#     plt.close(fig)




# def _load_as_2d(path):
#     with open(path, "r") as f:
#         data = ast.literal_eval(f.read().strip())
#     if isinstance(data, list) and len(data) > 0 and isinstance(data[0], (int, float)):
#         data = [data]
#     return [[float(x) for x in row] for row in data]

# def _pick_global_vector(data_2d, mode="mean"):
#     arr = np.array(data_2d, dtype=float)
#     if arr.ndim == 1:
#         return arr
#     if mode == "first":
#         return arr[0]
#     return arr.mean(axis=0)

# def plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                    global_mode="mean", out_dir="../new_save/"):

#     beforefl_2d = _load_as_2d(beforefl_file)   # per-client (Local) -> num_users x num_classes
#     fedavg_2d   = _load_as_2d(fedavg_file)
#     fedprox_2d  = _load_as_2d(fedprox_file)
#     fedproto_2d = _load_as_2d(fedproto_file)

#     num_users = len(beforefl_2d)

#     fedavg_g   = _pick_global_vector(fedavg_2d,   mode=global_mode)
#     fedprox_g  = _pick_global_vector(fedprox_2d,  mode=global_mode)
#     fedproto_g = _pick_global_vector(fedproto_2d, mode=global_mode)

#     os.makedirs(out_dir, exist_ok=True)

#     for user_index in range(num_users):
#         local = np.array(beforefl_2d[user_index], dtype=float)

#         max_len = max(len(local), len(fedavg_g), len(fedprox_g), len(fedproto_g))
#         local    = np.pad(local,     (0, max_len - len(local)),     constant_values=0.0)
#         fedavg   = np.pad(fedavg_g,  (0, max_len - len(fedavg_g)),  constant_values=0.0)
#         fedprox  = np.pad(fedprox_g, (0, max_len - len(fedprox_g)), constant_values=0.0)
#         fedproto = np.pad(fedproto_g,(0, max_len - len(fedproto_g)),constant_values=0.0)

#         index = np.arange(max_len)
#         w = 0.20

#         #fig, ax = plt.subplots()
#         plt.rcParams.update({
  
#             "ytick.labelsize": 28,
            
#         })
#             # "font.size":       20,
#             # "axes.titlesize":  22,
#             # "axes.labelsize":  20,
#             # "xtick.labelsize": 18,
#             # "legend.fontsize": 18,
#         fig, ax = plt.subplots(figsize=(6.5, 3.5))
#         ax.bar(index + 0*w, local,    w, color="tab:blue")# , label="Local")
#         ax.bar(index + 1*w, fedavg,   w, color="tab:green")#, label="Cerberus")     # FedAvg in files, Cerberus in legend
#         ax.bar(index + 2*w, fedprox,  w, color="tab:brown")#, label="FedProx")
#         ax.bar(index + 3*w, fedproto, w, color="tab:orange")#, label="PROTEAN")

#         # ax.set_xlabel("Classes")
#         # ax.set_ylabel("Accuracy")
#         ax.set_xlabel("Classes", fontsize=38)
#         ax.set_ylabel("Accuracy", fontsize=38)

#         ax.set_xticks(index + 1.5*w)
#         ax.set_xticklabels([f"{i+1}" for i in range(max_len)], fontsize=28)
#         ax.set_ylim(0, 1.0)
#         ax.set_title(
#             f"User {user_index + 1}",
#             fontsize=30,
#             pad=12
#         )

#         ax.legend()

#         plt.tight_layout()
#         out_name = (
#             f"acc_comparison_4algos_user{user_index+1}_"
#             f"data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
#         )
#         plt.savefig(os.path.join(out_dir, out_name))
#         plt.close(fig)

# # if __name__ == "__main__":
# #     import argparse
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument("--alpha", type=float, default=0.25)
# #     parser.add_argument("--num_users", type=int, default=10)
# #     parser.add_argument("--dataset", type=str, default="5gnidd")
# #     args = parser.parse_args()

# #     base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}_weighted/"
# #     #_weighted/"
# #     #_rounds10/"

# #     # Files are named by your pipeline (beforefl/fedavg/fedprox/fedproto)
# #     beforefl_file = base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
# #     fedavg_file   = base + "fedavg/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt"
# #     fedprox_file  = base + "fedprox/"  + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt"
# #     fedproto_file = base + "fedproto/" + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt"

# #     plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
# #                                    global_mode="mean", out_dir="../new_save/")



# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--alpha", type=float, default=0.25)
#     parser.add_argument("--num_users", type=int, default=10)
#     parser.add_argument("--dataset", type=str, default="xiiotid")
#     args = parser.parse_args()

#     base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}"
#     if args.dataset == "5gnidd":
#         base += "_weighted/"
#     else:
#         base += "_rounds10/"

#     beforefl_file = base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
# #   

#     fedavg_file   = base + "fedavg/"    + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt"
#     fedprox_file  = base + "fedprox/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt"
#     fedproto_file = base + "fedproto/"  + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt"

#     dist_file = "../save2_paper/" + f"_alpha{args.alpha}_num_users{args.num_users}classes_distribution_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"

#     # plot_avg_rare_accuracy_3algos(
#     #     args,
#     #     fedavg_file, fedprox_file, fedproto_file, dist_file,
#     #     k_rare=2,
#     #     global_mode="mean",
#     #     out_dir="../new_save/"
#     # )

#     plot_accuracy_comparison_4algos(args, beforefl_file, fedavg_file, fedprox_file, fedproto_file,
#                                         global_mode="mean", out_dir="../new_save/")



#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt

# -------------------- fixed, consistent figure size -------------------- #
FIGSIZE = (6.5, 3.5)

# -------------------- labels + colors (your choices) ------------------- #
# ALGO_LABEL = {
#     "local":   "Local",
#     "fedavg":  "Cerberus",
#     "fedprox": "FedProx",
#     "fedproto":"PROTEAN",
# }

ALGO_LABEL = {
    "local":   "",
    "fedavg":  "",
    "fedprox": "",
    "fedproto":"",
}

ALGO_COLOR = {
    "local":   "tab:blue",
    "fedavg":  "tab:green",
    "fedprox": "tab:brown",
    "fedproto":"tab:orange",
}

# ----------------------------- IO helpers ------------------------------ #
def _load_obj(path):
    with open(path, "r") as f:
        return ast.literal_eval(f.read().strip())

def _as_2d(data):
    # 1D -> 2D
    if isinstance(data, list) and len(data) and isinstance(data[0], (int, float)):
        return [list(map(float, data))]
    # 2D
    if isinstance(data, list) and len(data) and isinstance(data[0], list):
        return [list(map(float, row)) for row in data]
    raise ValueError("Unsupported format in file (expected list or list-of-lists).")

def _pick_global_vector(data_2d, mode="mean"):
    arr = np.array(data_2d, dtype=float)
    if arr.ndim == 1:
        return arr
    if mode == "first":
        return arr[0]
    return arr.mean(axis=0)

def _parse_algos_csv(s: str):
    algos = [a.strip().lower() for a in s.split(",") if a.strip()]
    valid = set(ALGO_LABEL.keys())
    bad = [a for a in algos if a not in valid]
    if bad:
        raise ValueError(f"Unknown algos: {bad}. Valid: {sorted(valid)}")
    if len(set(algos)) != len(algos):
        raise ValueError("Duplicate algo in --algos.")
    return algos

# ------------------ distribution parsing (list OR log) ------------------ #
def _get_rare_classes(dist_file, k_rare=2):
    """
    Supports BOTH formats:
      (1) Python list: [[c0,c1,...],[...],...]
      (2) Text log like:
            User 0:
              Class 4: 1001 instances
              Class 1: 16 instances
    Returns rare_by_user[u] = indices of k_rare smallest positive-count classes.
    """
    # ---- try python-list format ----
    try:
        txt = open(dist_file, "r").read().strip()
        data = ast.literal_eval(txt)
        if isinstance(data, list):
            rare_by_user = []
            for cnts in data:
                present = [(idx, cnt) for idx, cnt in enumerate(cnts) if cnt > 0]
                least = sorted(present, key=lambda x: (x[1], x[0]))[:k_rare]
                rare_by_user.append([idx for idx, _ in least])
            return rare_by_user
    except Exception:
        pass

    # ---- log format (your exact style) ----
    re_user  = re.compile(r"^\s*User\s+(\d+)\s*:\s*$")
    re_class = re.compile(r"^\s*Class\s+(\d+)\s*:\s*(\d+)\s*(?:instances)?\s*$")

    rare_by_user = []
    cur_counts = {}

    def finalize(counts):
        if not counts:
            return None
        present = [(cid, cnt) for cid, cnt in counts.items() if cnt > 0]
        least = sorted(present, key=lambda x: (x[1], x[0]))[:k_rare]
        return [cid for cid, _ in least]

    with open(dist_file, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            if re_user.match(line):
                fin = finalize(cur_counts)
                if fin is not None:
                    rare_by_user.append(fin)
                cur_counts = {}
                continue

            m = re_class.match(line)
            if m:
                cid = int(m.group(1))
                cnt = int(m.group(2))
                cur_counts[cid] = cnt

    fin = finalize(cur_counts)
    if fin is not None:
        rare_by_user.append(fin)

    if not rare_by_user:
        raise ValueError(f"Could not parse distribution file: {dist_file}")

    return rare_by_user

# ----------------------- build files from base -------------------------- #
def build_paths(args):
    """
    Same naming + folders as your second code.
    """
    base = f"../save2_paper/_alpha{args.alpha}_num_users{args.num_users}"
    if args.dataset == "5gnidd":
        base += "_weighted/"
    else:
        base += "_rounds10/"

    algo_files = {
        "local":    base + "before_fl/" + f"acc_byclient_byclass_before_fl_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt",
        "fedavg":   base + "fedavg/"    + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedavg_num_users{args.num_users}.txt",
        "fedprox":  base + "fedprox/"   + f"acc_byclass_data_{args.dataset}_alpha{args.alpha}_algfedprox_num_users{args.num_users}.txt",
        "fedproto": base + "fedproto/"  + f"acc_byclient_byclass_data_{args.dataset}_alpha{args.alpha}_algfedproto_num_users{args.num_users}.txt",
    }

    # distribution file (same weird no-slash convention you used)
    dist_file = "../save2_paper/" + (
        f"_alpha{args.alpha}_num_users{args.num_users}"
        f"classes_distribution_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.txt"
    )

    return algo_files, dist_file

# ------------------------- PLOT: rare classes --------------------------- #
def plot_avg_rare_accuracy_selectable(args, algo_files, dist_file, algos_to_plot,
                                     k_rare=2, global_mode="mean", out_dir="../new_save/"):
    rare_by_user = _get_rare_classes(dist_file, k_rare=k_rare)
    n_users = len(rare_by_user)

    mats = {}
    for algo in algos_to_plot:
        path = algo_files[algo]
        if algo == "local":
            mats[algo] = _as_2d(_load_obj(path))[:n_users]
        else:
            g = _pick_global_vector(_as_2d(_load_obj(path)), mode=global_mode)
            mats[algo] = [g.tolist()] * n_users

    def avg_on_rare(mat):
        return [float(np.mean([mat[u][c] for c in rare_by_user[u]])) for u in range(n_users)]

    ys = {algo: avg_on_rare(mats[algo]) for algo in algos_to_plot}

    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.tick_params(axis="y", labelsize=28)
    ax.tick_params(axis="x", labelsize=28)


    idx = np.arange(n_users)
    w = 0.80 / max(1, len(algos_to_plot))
    start = -0.5 * (len(algos_to_plot) - 1) * w

    for j, algo in enumerate(algos_to_plot):
        ax.bar(idx + (start + j*w), ys[algo], w,
               color=ALGO_COLOR[algo], label=ALGO_LABEL[algo])

    ax.set_xlabel("User", fontsize=28)
    ax.set_ylabel("Avg Accuracy", fontsize=28)
    ax.set_xticks(idx)
    ax.set_xticklabels([str(u) for u in idx])
    ax.set_ylim(0, 1.0)
    ax.legend()

    plt.tight_layout()
    out_name = f"rare_acc_{'-'.join(algos_to_plot)}_data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
    plt.savefig(os.path.join(out_dir, out_name))
    plt.close(fig)

# ---------------------- PLOT: per-class (by user) ------------------------ #
def plot_accuracy_byclass_selectable(args, algo_files, algos_to_plot,
                                     global_mode="mean", out_dir="../new_save/"):
    os.makedirs(out_dir, exist_ok=True)

    # If local is included -> true per-user figures.
    # If not -> make ONE global figure (since there is no per-user vector).
    has_local = ("local" in algos_to_plot)

    if has_local:
        local_2d = _as_2d(_load_obj(algo_files["local"]))
        num_users = len(local_2d)
    else:
        local_2d = None
        num_users = 1  # one global plot

    # pre-load global vectors once
    globals_vec = {}
    for algo in algos_to_plot:
        if algo == "local":
            continue
        globals_vec[algo] = _pick_global_vector(_as_2d(_load_obj(algo_files[algo])), mode=global_mode)

    for user_index in range(num_users):
        series = {}

        if has_local:
            series["local"] = np.array(local_2d[user_index], dtype=float)

        for algo in algos_to_plot:
            if algo == "local":
                continue
            series[algo] = np.array(globals_vec[algo], dtype=float)

        max_len = max(len(v) for v in series.values())
        for k in series:
            series[k] = np.pad(series[k], (0, max_len - len(series[k])), constant_values=0.0)

        index = np.arange(max_len)
        w = 0.80 / max(1, len(algos_to_plot))
        start = -0.5 * (len(algos_to_plot) - 1) * w

        plt.rcParams.update({"ytick.labelsize": 28})
        fig, ax = plt.subplots(figsize=FIGSIZE)

        for j, algo in enumerate(algos_to_plot):
            ax.bar(index + (start + j*w), series[algo], w,
                   color=ALGO_COLOR[algo], label=ALGO_LABEL[algo])

        ax.set_xlabel("Classes", fontsize=38)
        ax.set_ylabel("Accuracy", fontsize=38)
        ax.set_xticks(index)
        ax.set_xticklabels([f"{i+1}" for i in range(max_len)], fontsize=28)
        ax.set_ylim(0, 1.0)

        if has_local:
            ax.set_title(f"User {user_index + 1}", fontsize=30, pad=12)
            tag = f"user{user_index+1}"
        else:
            ax.set_title("Global", fontsize=30, pad=12)
            tag = "global"

        ax.legend()
        plt.tight_layout()

        out_name = (
            f"acc_byclass_{'-'.join(algos_to_plot)}_{tag}_"
            f"data_{args.dataset}_alpha{args.alpha}_num_users{args.num_users}.pdf"
        )
        plt.savefig(os.path.join(out_dir, out_name))
        plt.close(fig)

# --------------------------------- main --------------------------------- #
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--num_users", type=int, default=10)
    parser.add_argument("--dataset", type=str, default="xiiotid", choices=["xiiotid", "5gnidd"])

    # choose algos manually
    parser.add_argument("--algos", type=str, default="local,fedavg,fedprox,fedproto",
                        help="Comma-separated subset of: local,fedavg,fedprox,fedproto")
    # choose plot type
    parser.add_argument("--mode", type=str, default="byclass", choices=["byclass", "rare"])
    # rare settings
    parser.add_argument("--k_rare", type=int, default=2)
    # how to convert 2D algo files into global vector
    parser.add_argument("--global_mode", type=str, default="mean", choices=["mean", "first"])
    parser.add_argument("--out_dir", type=str, default="../new_save/")
    args = parser.parse_args()

    algos_to_plot = _parse_algos_csv(args.algos)
    algo_files, dist_file = build_paths(args)

    # quick file existence check for selected algos (fail fast, clear error)
    for a in algos_to_plot:
        if not os.path.exists(algo_files[a]):
            raise FileNotFoundError(f"Missing file for {a}: {algo_files[a]}")
    if args.mode == "rare":
        if not os.path.exists(dist_file):
            raise FileNotFoundError(f"Missing distribution file: {dist_file}")

        plot_avg_rare_accuracy_selectable(
            args,
            algo_files=algo_files,
            dist_file=dist_file,
            algos_to_plot=algos_to_plot,
            k_rare=args.k_rare,
            global_mode=args.global_mode,
            out_dir=args.out_dir,
        )
    else:
        plot_accuracy_byclass_selectable(
            args,
            algo_files=algo_files,
            algos_to_plot=algos_to_plot,
            global_mode=args.global_mode,
            out_dir=args.out_dir,
        )
