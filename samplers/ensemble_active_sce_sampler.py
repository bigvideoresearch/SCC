import collections
import json
import math
import numpy as np
import os
import random
from tqdm import tqdm


def ensemble_active_sce_sampling(sampler, logits_dict, indexes, image_names):
    """Mining the positive samples with high confidence of top-k
       of each class while split the score to active mining hard
       samples with ensemble two models

    Args:
        sampler (dict): the configuration of the sce sampler
        logits (numpy array): the predicted logits
        indexes (numpy array): the corresponding indexes of logits
        image_name (list): the corresponding image name of logits
    """

    classes_dictionary = collections.defaultdict(list)
    for i in tqdm(range(len(indexes))):
        logit = 0
        for key, value in logits_dict.items():
            logit += value[i]
        logit = logit / len(logits_dict.keys())
        idxs = logit.argsort()[::-1][:sampler.topK]
        for cls_idx in idxs:
            cls_score = logit[cls_idx]
            classes_dictionary[str(cls_idx)].append(
                [image_names[indexes[i]], cls_score])

    if sampler.use_al.use:
        active_dictionary = collections.defaultdict(str)
        for label_idx in classes_dictionary.keys():
            active_dictionary[label_idx] = collections.defaultdict(list)

    f_autolabeled = open(sampler.mining_auto_labeled_path, 'w')

    for i, label_idx in enumerate(classes_dictionary.keys()):
        class_items = classes_dictionary[label_idx]
        class_items.sort(key=lambda x: x[1], reverse=True)
        class_items = np.array(class_items)
        min_length = min(sampler.topN, len(class_items))
        for index in range(0, min_length):
            item = class_items[index][0]
            if index >= min_length * 0.95 and sampler.use_al.use:
                score = float(class_items[index][1])
                ceil_score = math.ceil(score * 10) / 10
                active_dictionary[label_idx][ceil_score].append(
                    class_items[index])
            else:
                f_autolabeled.write(item + ' '
                                    + str(label_idx) + ' '
                                    + class_items[index][1] + '\n')

    if sampler.use_al.use:
        active_set = set()
        f_activelabeled = open(sampler.use_al.mining_active_labeled_path, 'w')
        if sampler.use_al.classes_map != '':
            data = json.load(open(sampler.use_al.classes_map))
            class_dict = collections.defaultdict(str)
            for key in data.keys():
                class_dict[str(data[key]['id'])] = key
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
                    f_activelabeled.write(item[0] + ' '
                                          + str(label_idx) + ' '
                                          + item[1] + '\n')
                    active_set.add(item[0])
    f_autolabeled.close()
    if sampler.use_al.use:
        print(
            'ensemble_active_sce_sampling done, with active mining, \
             need to human-labeled num: {}'
            .format(len(active_set)))
    else:
        print('ensemble_active_sce_sampling done, without active mining')
