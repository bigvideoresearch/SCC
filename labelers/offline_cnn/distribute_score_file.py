import numpy as np
import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--file_root', type=str, default='../data/google500/', help='Feature/score file root')
parser.add_argument('--keyword', type=str, default='google500_base', help='score filename prefix')
parser.add_argument('--get_new_reweight', type=str, default='False', help='score filename prefix')
args = parser.parse_args()

file_root = args.file_root
prefix = args.keyword
get_new_reweight = args.get_new_reweight
read_score_file = file_root + '{}_score.npy'.format(prefix)
read_imglist_file = file_root + '{}_imglist.txt'.format(prefix)
save_imglist_file = file_root + '{}_reweighted_imglist.txt'.format(prefix)
save_root = file_root + '{}_score/'.format(prefix)

new_score = np.load(read_score_file)

print('Begin to save {} as file.'.format(save_root), flush=True)

def getlabelarray(imglist_path):
    img_name = []
    label_set = []
    import re
    with open(imglist_path, 'r') as reader:
        for line in reader.readlines():
            words = re.split(',| |{|}|"|:|\n', line)
            words = list(filter(None, words))
            img_name.append(words[0])
            if 'label' in words:
                label = words[words.index('label') + 1]
            else:
                label = words[1]
            label_set.append(int(label))
    img_name = np.array(img_name)
    label_set = np.array(label_set)
    return img_name, label_set

def get_reweight_ratio(pred_labels):
    try:
        from collections import Counter
        class_and_counts = np.array(sorted(Counter(pred_labels).items()))
        np.testing.assert_array_equal(
            class_and_counts[:, 0],
            np.arange(class_and_counts.shape[0]),
        )
        class_freq = class_and_counts[:, 1]
        class_weight = 1 / class_freq
        sample_weight = class_weight[pred_labels]
        sample_weight *= 1 / sample_weight.mean()
    except:
        print('Some classes are missing!!!', flush=True)
        sample_weight = np.ones(pred_labels.shape)
    return sample_weight

def get_revised_imglist(train_imglist_path, pred_labels, revised_imglist_path):
    train_imgname, train_label = getlabelarray(train_imglist_path)
    assert len(train_imgname) == len(pred_labels), "train imglist should not equal to prediction list"
    sample_reweight = get_reweight_ratio(pred_labels)
    newlines = []
    import json
    for i, imgname in enumerate(train_imgname):
        target_line = '{} {}\n'.format(
            imgname,
            json.dumps({
                'label': int(pred_labels[i]),
                'sample_weight': sample_reweight[i],
            }, sort_keys=True)
        )
        newlines.append(target_line)
    with open(revised_imglist_path, 'w') as writer:
        writer.writelines(newlines)

if get_new_reweight == 'True':
    pred_labels = np.argmax(new_score, axis=1)
    get_revised_imglist(read_imglist_file, pred_labels, save_imglist_file)

with open(read_imglist_file, 'r') as reader:
    lines = reader.readlines()

def multi_thread_save_feat(imglist, score, save_root):
    for idx, line in enumerate(imglist):
        new_name = line.split(' ')[0].split('.jpg')[0]
        save_name = save_root + new_name + '.npy'
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        np.save(save_name, score[idx])

correct_score = new_score[:len(lines)]

n_threads = 40
threads = []
length = len(lines) // n_threads
for i in range(n_threads):
    start = i * length
    if i == n_threads - 1:
        end = len(lines)
    else:
        end = start + length
    t = threading.Thread(target=multi_thread_save_feat, args=(lines[start:end], correct_score[start:end], save_root))
    threads.append(t)
for i in range(n_threads):
    threads[i].start()
# wait for all threads to finish
for i in range(n_threads):
    threads[i].join()

print('Completed!', flush=True)
