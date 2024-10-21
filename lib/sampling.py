#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import random
import torch


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


# def mnist_noniid(dataset, num_users):
#     """
#     Sample non-I.I.D client data from MNIST dataset
#     :param dataset:
#     :param num_users:
#     :return:
#     """
#     # 60,000 training imgs -->  200 imgs/shard X 300 shards
#     num_shards, num_imgs = 200, 300
#     idx_shard = [i for i in range(num_shards)]
#     dict_users = {i: np.array([]) for i in range(num_users)}
#     idxs = np.arange(num_shards*num_imgs)
#     labels = dataset.train_labels.numpy()
#
#     # sort labels
#     idxs_labels = np.vstack((idxs, labels))
#     idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
#     idxs = idxs_labels[0, :]
#
#     # divide and assign 2 shards/client
#     for i in range(num_users):
#         rand_set = set(np.random.choice(idx_shard, 2, replace=False))
#         idx_shard = list(set(idx_shard) - rand_set)
#         for rand in rand_set:
#             dict_users[i] = np.concatenate(
#                 (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
#     return dict_users

def mnist_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 6000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = test_dataset.train_labels.numpy()
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 40 # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i*40 + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def femnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

def femnist_noniid(args, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from FEMNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    dict_users = {}
    classes_list = []
    classes_list_gt = []

    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        print("classes_gt:", classes)
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k_len * num_users + i * k_len
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def femnist_noniid_lt(args, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    dict_users = {}

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        user_data = np.array([])
        for class_idx in classes:
            begin = class_idx * k * num_users + i * k
            user_data = np.concatenate((user_data, np.arange(begin, begin + k)), axis=0)
        dict_users[i] = user_data

    return dict_users


def femnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users

def cifar10_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 5000
    dict_users = {}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    classes_list = []
    classes_list_gt = []
    k_len = args.train_shots_max
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0, args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin: begin + k]), axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)
        classes_list_gt.append(classes)

    return dict_users, classes_list, classes_list_gt

def cifar10_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 10, 1000
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = args.test_shots
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users



def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users



def cifar100_noniid(args, dataset, num_users, n_list, k_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 500
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            begin = label_begin[each_class.item()] + i*5
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list


def cifar100_noniid_lt(test_dataset, num_users, classes_list):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 100, 100
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    labels = np.array(test_dataset.targets)
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    for i in range(num_users):
        k = 5 # 每个类选多少张做测试
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*5 + label_begin[each_class.item()]
            begin = random.randint(0,90) + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users
    #
    #
    #
    #
    #
    # # divide and assign 2 shards/client
    # for i in range(num_users):
    #     rand_set = set(np.random.choice(idx_shard, n_list[i], replace=False))
    #     idx_shard = list(set(idx_shard) - rand_set)
    #     for rand in rand_set:
    #         dict_users[i] = np.concatenate(
    #             (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    # return dict_users

def xiiotid_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_classes = len(np.unique(dataset.targets))
    num_items = int(len(dataset)/num_users)
    classes_list = []
    #n = list(range(num_classes))
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        classes = np.arange(num_classes) #random.sample(range(0,num_classes), n)
        #classes = np.sort(classes)
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
        classes_list.append(classes)
    return dict_users, classes_list

def xiiotid_noniid(args, dataset, num_users, n_list, k_list):
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_instances = dataset.__len__()
    num_classes = len(np.unique(dataset.targets)) #10
    num_shards, num_imgs = num_classes, num_instances // num_classes
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    if isinstance(dataset.labels, np.ndarray):
        labels = dataset.labels
    else:
        labels = dataset.labels.numpy()
    min_length = min(len(idxs), len(labels))
    idxs = idxs[:min_length]
    labels = labels[:min_length]
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1

    classes_list = []
    for i in range(num_users):
        n = n_list[i]
        k = k_list[i]
        k_len = args.train_shots_max
        classes = random.sample(range(0,args.num_classes), n)
        classes = np.sort(classes)
        print("user {:d}: {:d}-way {:d}-shot".format(i + 1, n, k))
        print("classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            # begin = i*10 + label_begin[each_class.item()]
            begin = i * k_len + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data
        classes_list.append(classes)

    return dict_users, classes_list

def xiiotid_noniid_lt(args, test_dataset, num_users, n_list, k_list, classes_list):

    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_instances = test_dataset.__len__()
    num_classes = 10
    num_shards, num_imgs = 10, num_instances // num_classes
    idx_shard = [i for i in range(num_shards)]
    dict_users = {}
    idxs = np.arange(num_shards*num_imgs)
    if isinstance(test_dataset.labels, np.ndarray):
        labels = test_dataset.labels
    else:
        labels = test_dataset.labels.numpy()
    min_length = min(len(idxs), len(labels))
    idxs = idxs[:min_length]
    labels = labels[:min_length]
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt=0
    for i in idxs_labels[1,:]:
        if i not in label_begin:
                label_begin[i] = cnt
        cnt+=1
    
    for i in range(num_users):
        k = 1000#40 # How many images should be selected for each class for testing
        classes = classes_list[i]
        print("local test classes:", classes)
        user_data = np.array([])
        for each_class in classes:
            k = min(k, np.sum(labels == each_class.item()))
            begin = i * k + label_begin[each_class.item()]
            user_data = np.concatenate((user_data, idxs[begin : begin+k]),axis=0)
        dict_users[i] = user_data


    return dict_users

def xiotid_noniid_lt_all(args, test_dataset, num_users):
    # Number of instances and classes
    num_instances = test_dataset.__len__()
    num_classes = 10  # Assuming 10 classes
    
    # Generate indices for the entire dataset
    idxs = np.arange(num_instances)
    
    if isinstance(test_dataset.labels, np.ndarray):
        labels = test_dataset.labels
    else:
        labels = test_dataset.labels.numpy()
    
    min_length = min(len(idxs), len(labels))
    idxs = idxs[:min_length]
    labels = labels[:min_length]
    
    # Sort indices based on labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    label_begin = {}
    cnt = 0
    for i in idxs_labels[1, :]:
        if i not in label_begin:
            label_begin[i] = cnt
        cnt += 1

    dict_users = {i: np.array([], dtype=int) for i in range(num_users)}
    num_items_per_user = num_instances // num_users

    for i in range(num_users):
        user_data = np.array([], dtype=int)
        for j in range(num_classes):
            class_idxs = idxs_labels[0, idxs_labels[1, :] == j]
            num_class_items_per_user = len(class_idxs) // num_users
            user_data = np.concatenate((user_data, class_idxs[i * num_class_items_per_user:(i + 1) * num_class_items_per_user]), axis=0)
        dict_users[i] = user_data

    return dict_users




def xiiotid_noniid_dirichlet(dataset, y_train, n_parties, partition):
    """
    Partition data in a non-IID manner using Dirichlet distribution.

    Args:
        dataset (str): Name of the dataset.
        y_train (numpy array): Labels of the training data.
        n_parties (int): Number of parties to partition the data into.
        partition (str): Partitioning strategy.

    Returns:
        dict: A dictionary where keys are party indices and values are lists of data indices for that party.
    """
    num = eval(partition[13:])
    if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
        num = 1
        K = 2
    else:
        K = 10
    if dataset == "cifar100":
        K = 100
    elif dataset == "tinyimagenet":
        K = 200

    if num == 10:
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(10):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, n_parties)
            for j in range(n_parties):
                net_dataidx_map[j] = np.append(net_dataidx_map[j], split[j])
    else:
        times = [0 for _ in range(K)]
        contain = []
        for i in range(n_parties):
            current = [i % K]
            times[i % K] += 1
            j = 1
            while j < num:
                ind = random.randint(0, K-1)
                if ind not in current:
                    j += 1
                    current.append(ind)
                    times[ind] += 1
            contain.append(current)
        
        net_dataidx_map = {i: np.ndarray(0, dtype=np.int64) for i in range(n_parties)}
        for i in range(K):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            ids = 0
            for j in range(n_parties):
                if i in contain[j]:
                    net_dataidx_map[j] = np.append(net_dataidx_map[j], split[ids])
                    ids += 1

    return net_dataidx_map




def xiiotid_noniid_dirichlet3(args, dataset, num_users, alpha, min_require_size=100):
    """
    Splits dataset indices for non-IID distribution using Dirichlet allocation.
    If a class has fewer instances than min_require_size, assign all its instances to one client.

    Args:
        args: Arguments containing necessary parameters like num_classes and seed.
        dataset: The dataset to split.
        num_users: Number of clients/users.
        alpha: Dirichlet distribution parameter.
        min_require_size: Minimum number of instances per client.

    Returns:
        net_dataidx_map: A dictionary mapping each client to its data indices.
        classes_list: A list containing the unique classes assigned to each client.
    """
    min_size = 0
    K = args.num_classes
    print("-----------------------------in xiiotid_noniid_dirichlet2-----------------------------")
    print("--alpha:", alpha)
    num_instances = len(dataset)
    num_classes = len(np.unique(dataset.targets))  # e.g., 10
    N = num_instances

    # Set random seed for reproducibility
    np.random.seed(args.seed)

    net_dataidx_map = {i: [] for i in range(num_users)}
    n_parties = num_users

    # Iterate until all clients have at least min_require_size samples
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(np.array(dataset.targets) == k)[0]
            np.random.shuffle(idx_k)
            if len(idx_k) < min_require_size:
                # Assign all instances of this class to the client with the least data
                client_id = np.argmin([len(batch) for batch in idx_batch])
                idx_batch[client_id].extend(idx_k.tolist())
            else:
                # Use Dirichlet distribution to allocate data among clients
                proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
                # Balance proportions to ensure no client exceeds N / n_parties
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                # Compute the split indices
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                # Split the indices and assign to each client
                split_idx = np.split(idx_k, proportions)
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, split_idx)]
            # Update the minimum size across all clients
            min_size = min([len(idx_j) for idx_j in idx_batch])
        
        # Check if the current allocation meets the minimum size requirement
        if min_size < min_require_size:
            print(f"Minimum size {min_size} is less than required {min_require_size}. Retrying...")
    
    # Shuffle and assign the indices to each client
    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    
    print("Assigned clients:", net_dataidx_map.keys())

    # Create a list of unique classes for each client
    classes_list = []
    for i in range(num_users):
        classes = np.unique(np.array(dataset.targets)[net_dataidx_map[i]])
        classes_list.append(classes)
    
    print("Classes assigned to each client:", classes_list)
    return net_dataidx_map, classes_list


def xiiotid_noniid_dirichlet2(args, dataset, num_users, alpha):
    min_size = 0
    min_require_size = 10
    K = args.num_classes
    print("-----------------------------in xiiotid_noniid_dirichlet2-----------------------------")
    print("--alpha:", alpha)
    num_instances = dataset.__len__()
    num_classes = len(np.unique(dataset.targets)) #10
    N = num_instances
    #np.random.seed(2020)
    np.random.seed(args.seed)
    net_dataidx_map = {}
    n_parties = num_users
    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(dataset.targets == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(alpha, n_parties))
            # logger.info("proportions1: ", proportions)
            # logger.info("sum pro1:", np.sum(proportions))
            ## Balance
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            # logger.info("proportions2: ", proportions)
            proportions = proportions / proportions.sum()
            # logger.info("proportions3: ", proportions)
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            # logger.info("proportions4: ", proportions)
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])
            # if K == 2 and n_parties <= 10:
            #     if np.min(proportions) < 200:
            #         min_size = 0
            #         break


    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
    print(net_dataidx_map.keys())
    classes_list = []
    for i in range(num_users):
        classes = np.unique(dataset.targets[net_dataidx_map[i]])
        classes_list.append(classes)
    print(classes_list)
    return net_dataidx_map, classes_list

if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
