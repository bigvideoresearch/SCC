import numpy as np
import argparse
import os
import threading

parser = argparse.ArgumentParser()
parser.add_argument('--file_root', type=str, default='../data/google500/', help='Feature file root')
parser.add_argument('--sgc_index', type=str, default='myGNN', help='sgc file name')
parser.add_argument('--threshold', type=float, default=0.7, help='threshold for merge scores.')
args = parser.parse_args()

file_root = args.file_root
sgc_idx = args.sgc_index
read_imglist_file = file_root + sgc_idx + '/sgc_revised_imglist.txt'
sgc_score_file = file_root + sgc_idx + '/sgc_revised_score.npy'
base_score_file = file_root + 'train_supervised_score.npy'
prefix = sgc_idx
threshold = args.threshold
lambdasgc = 0.5
method = 'th_' + str(threshold) + '_lambda_' + str(lambdasgc) # 'use_onehot', 'use_soft', 'half_add', 'use_soft_half_add'


generate_save_name = '{}_{}'.format(prefix, method)
save_root = file_root + '{}_score/'.format(generate_save_name)
save_imglist_file = file_root + '{}_imglist.txt'.format(generate_save_name)
save_score_file = file_root + '{}_score.npy'.format(generate_save_name)

sgc_model_score = np.load(sgc_score_file)
base_model_score = np.load(base_score_file)

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

def get_new_score(sgc_model_score, base_model_score, method, threshold):
    clean_score = np.zeros(sgc_model_score.shape)
    clean_score[np.where(sgc_model_score > threshold)[0], np.where(sgc_model_score > threshold)[1]] = 1
    clean_seed_idx = np.where(np.sum(clean_score, axis=1) > 0)[0].astype(np.int32)

    base_model_score = lambdasgc * sgc_model_score + (1 - lambdasgc) * base_model_score
    base_model_score[clean_seed_idx] = sgc_model_score[clean_seed_idx]

    print('Totally {} sgc labels kept.'.format(len(clean_seed_idx)), flush=True)
    del sgc_model_score
    return base_model_score

new_score = get_new_score(sgc_model_score, base_model_score, method, threshold)
del sgc_model_score, base_model_score
pred_labels = np.argmax(new_score, axis=1)
get_revised_imglist(read_imglist_file, pred_labels, save_imglist_file)
np.save(save_score_file, new_score)

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

# srun -p tbd -n 1 --gres gpu:1 --ntasks-per-node 1 --job-name lc python mergeScore.py
