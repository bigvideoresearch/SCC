import collections
import json
import math
import os
import random
import numpy as np


def active_bce_sampling(sampler, logits, indexes, image_names):
    """Mining the high confidence positive samples and low
       confidence negative samples while active mining ambiguous
       samples

    Args:
        sampler (dict): the configuration of the bce sampler
        logits (numpy array): the predicted logits
        indexes (numpy array): the corresponding indexes of logits
        image_name (list): the corresponding image name of logits
    """

    positive_dictionary = collections.defaultdict(list)
    negative_dictionary = collections.defaultdict(list)
    ambiguous_dictionary = collections.defaultdict(list)
    for i in range(len(logits)):
        logit = logits[i]
        if max(logit) <= sampler.thresh_negative:
            negative_dictionary[str(-1)].append([image_names[indexes[i]], 0])
        else:
            idxs = logit.argsort()[::-1]
            if logit[idxs[1]] > sampler.thresh_negative:
                ambiguous_dictionary[str(idxs[1])].append(
                    [image_names[indexes[i]], logit[idxs[1]]])
                continue
            for cls_idx in idxs:
                cls_score = logit[cls_idx]
                if cls_score > sampler.thresh_negative:
                    positive_dictionary[str(cls_idx)].append(
                        [image_names[indexes[i]], cls_score])

    if sampler.use_al.use:
        active_dictionary = collections.defaultdict(str)
        for label_idx in positive_dictionary.keys():
            active_dictionary[label_idx] = collections.defaultdict(list)

    f_autolabeled = open(sampler.mining_auto_labeled_path, 'w')
    for i, label_idx in enumerate(positive_dictionary.keys()):
        class_items = positive_dictionary[label_idx]
        class_items.sort(key=lambda x: x[1], reverse=True)
        class_items = np.array(class_items)
        for index in range(len(class_items)):
            if float(class_items[index][1]) > (sampler.thresh_positive):
                f_autolabeled.write(class_items[index][0]
                                    + ' [' + str(label_idx) + '] '
                                    + class_items[index][1] + '\n')
            elif sampler.use_al.use:
                score = float(class_items[index][1])
                ceil_score = math.ceil(score * 10) / 10
                active_dictionary[label_idx][ceil_score].append(
                    class_items[index])

    for i, label_idx in enumerate(negative_dictionary.keys()):
        class_items = negative_dictionary[label_idx]
        class_items = np.array(class_items)
        for index in range(len(class_items)):
            f_autolabeled.write(class_items[index][0]
                                + ' [] '
                                + class_items[index][1] + '\n')

    if sampler.use_al.use:
        active_num = 0
        f_activelabeled = open(sampler.use_al.mining_active_labeled_path, 'w')
        if sampler.use_al.classes_map != '':
            data = json.load(open(sampler.use_al.classes_map))
            class_dict = collections.defaultdict(str)
            for key in data.keys():
                class_dict[str(data[key]['id'])] = key
        for i, label_idx in enumerate(ambiguous_dictionary.keys()):
            if sampler.use_al.classes_map != '':
                active_class_dir = os.path.join(
                    sampler.use_al.mining_active_labeled_dir,
                    class_dict[label_idx])
            else:
                active_class_dir = os.path.join(
                    sampler.use_al.mining_active_labeled_dir, label_idx)
            if not os.path.exists(active_class_dir):
                os.makedirs(active_class_dir)
            class_items = ambiguous_dictionary[label_idx]
            class_items = np.array(class_items)
            for index in range(len(class_items)):
                item = class_items[index]
                f_activelabeled.write(item[0]
                                      + ' ' + str(label_idx)
                                      + ' ' + item[1] + '\n')
                active_num += 1

        for i, label_idx in enumerate(active_dictionary.keys()):
            if sampler.use_al.classes_map != '':
                active_class_dir = os.path.join(
                    sampler.use_al.mining_active_labeled_dir,
                    class_dict[label_idx])
            else:
                active_class_dir = os.path.join(
                    sampler.use_al.mining_active_labeled_dir, label_idx)
            if not os.path.exists(active_class_dir):
                os.makedirs(active_class_dir)
            for j, score in enumerate(active_dictionary[label_idx].keys()):
                items = active_dictionary[label_idx][score]
                random.shuffle(items)
                length = int(len(items) * sampler.use_al.percent)
                for item in items[:length]:
                    f_activelabeled.write(item[0]
                                          + ' ' + str(label_idx)
                                          + ' ' + item[1] + '\n')
                    active_num += 1
    f_autolabeled.close()
    if sampler.use_al.use:
        print('active_bce_sampling done, with active mining,\
              to labeled num: {}'
              .format(active_num))
    else:
        print('active_bce_sampling done, without active mining')
