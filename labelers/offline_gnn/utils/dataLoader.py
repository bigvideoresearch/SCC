import re
import json
import numpy as np
import pandas as pd
from collections import Counter
from .knn_utils import global_build, sgc_precompute
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def getlabelarray(imglist_path):
    img_name = []
    label_set = []
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

def create_df(imglist_path, df_file_path, feature_file=None):
    [img_name, label] = getlabelarray(imglist_path)
    df = pd.DataFrame()
    df['img_name'] = img_name
    df['y'] = label
    if feature_file is not None:
        df = add_tsne(df, feature_file)
    df.to_pickle(df_file_path)
    return df

def add_pseudo(df, pseudo_path, select_idx, class_set):
    img_name, label = getlabelarray(pseudo_path)
    plabel = label[select_idx]
    df['pseudo'] = plabel
    for idx, pseudo_label in enumerate(plabel):
        if pseudo_label not in class_set:
            plabel[idx] = 1000
    df['yp'] = plabel
    return df

def add_tsne(df, feature_file):
    x = np.load(feature_file)
    assert len(x) == len(df); print('Df number does not match feature number')
    df = tsne_compute(df, x)
    return df


def tsne_compute(df, x):
    pca_50 = PCA(n_components=50)
    pca_result_50 = pca_50.fit_transform(x)
    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000)
    tsne_pca_results = tsne.fit_transform(pca_result_50)
    df['tsne-pca50-one'] = tsne_pca_results[:, 0]
    df['tsne-pca50-two'] = tsne_pca_results[:, 1]
    return df

def get_conf(df, text_path, label_path, imglist_path, class_num, topk=5):
    text_feature = np.load(text_path)
    label_description = np.load(label_path)
    label_set = np.array(df['y'])
    conf_list = []
    conf_mark_list = []
    for i in class_num:
        class_text = text_feature[np.where(label_set == i)]
        conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
        conf_mark = 10000 * np.ones(conf.shape)
        conf_mark[np.argsort(-conf)[:topk]] = i
        conf_mark_list.extend(conf_mark)
        conf_list.extend(conf)
    conf_mark_list = np.array(conf_mark_list).astype(np.int)
    conf_list = np.array(conf_list)
    df['conf'] = conf_list
    df['seed'] = conf_mark_list
    return df

def get_knn_conf(df, text_path, visual_path, label_path, imglist_path, class_num, save_filename, dist_def, k=5, topk=5,
                 self_weight=0, edge_weight=True, build_graph_method='gpu'):
    text_feature = np.load(text_path)
    knn_graph = global_build(visual_path, dist_def, k, save_filename, build_graph_method)
    text_feature = sgc_precompute(text_feature, knn_graph, self_weight=self_weight, edge_weight=edge_weight, degree=1)
    label_description = np.load(label_path)
    label_set = np.array(df['y'])
    conf_list = np.zeros(np.shape(label_set))
    conf_mark_list = np.zeros(np.shape(label_set))
    for i in class_num:
        class_google_idx = np.array(df[df['y'] == i].index)
        class_text = text_feature[class_google_idx]
        conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
        conf_mark = 10000 * np.ones(conf.shape)
        conf_mark[np.argsort(-conf)[:topk]] = i
        conf_list[class_google_idx] = conf
        conf_mark_list[class_google_idx] = np.array(conf_mark).astype(np.int)
        if i%100==0:
            print('Seed selection class-{} completed.'.format(i), flush=True)
    conf_mark_list = np.array(conf_mark_list).astype(np.int)
    conf_list = np.array(conf_list)
    df['conf'] = conf_list
    df['seed'] = conf_mark_list
    return df

def get_knn_conf_multisource(df, text_path, visual_path, label_path, imglist_path, class_num, save_filename, dist_def, k=5, topk=5,
                 self_weight=0, edge_weight=True, build_graph_method='gpu'):
    text_feature = np.load(text_path)
    knn_graph = global_build(visual_path, dist_def, k, save_filename, build_graph_method)
    text_feature = sgc_precompute(text_feature, knn_graph, self_weight=self_weight, edge_weight=edge_weight, degree=1)
    label_description = np.load(label_path)
    label_set = np.array(df['y'])
    conf_list = np.zeros(np.shape(label_set))
    conf_mark_list = np.zeros(np.shape(label_set))
    for i in class_num:
        class_google_idx = np.array(df[df['img_name'].str.startswith('google') & (df['y']==i)].index)
        class_text = text_feature[class_google_idx]
        conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
        conf_mark = 10000 * np.ones(conf.shape)
        conf_mark[np.argsort(-conf)[:topk]] = i
        conf_list[class_google_idx] = conf
        conf_mark_list[class_google_idx] = np.array(conf_mark).astype(np.int)

        class_flickr_idx = np.array(df[df['img_name'].str.startswith('flickr') & (df['y']==i)].index)
        if len(class_flickr_idx) > 0:
            class_text = text_feature[class_flickr_idx]
            conf = cosine_similarity(class_text, label_description[i].reshape(1, -1)).flatten()
            conf_mark = 10000 * np.ones(conf.shape)
            conf_mark[np.argsort(-conf)[:topk]] = i
            conf_list[class_flickr_idx] = conf
            conf_mark_list[class_flickr_idx] = np.array(conf_mark).astype(np.int)
        else:
            print('Class {} does not contain Flickr files.'.format(i),flush=True)
        if i%100==0:
            print('Seed selection class-{} completed.'.format(i),flush=True)
    conf_mark_list = np.array(conf_mark_list).astype(np.int)
    conf_list = np.array(conf_list)
    df['conf'] = conf_list
    df['seed'] = conf_mark_list
    return df

def data_slicing(train_imglist_path, train_feature_path, train_graph_feature_path,
                 val_imglist_path, val_feature_path, val_graph_feature_path):

    train_imgname, train_label = getlabelarray(train_imglist_path)
    val_imgname, val_label = getlabelarray(val_imglist_path)

    train_feature = np.load(train_feature_path)
    val_feature = np.load(val_feature_path)

    train_graph_feature = np.load(train_graph_feature_path)
    val_graph_feature = np.load(val_graph_feature_path)

    imglist = np.concatenate((train_imgname, val_imgname), axis=0)
    feature = np.concatenate((train_feature, val_feature), axis=0)
    graph_feature = np.concatenate((train_graph_feature, val_graph_feature), axis=0)
    label = np.concatenate((train_label, val_label), axis=0)

    train_idx = np.arange(len(train_feature))
    val_idx = np.arange(len(val_feature)) + len(train_feature)

    return imglist, feature, graph_feature, label, train_idx, val_idx

def get_revised_imglist(train_imglist_path, pred_labels, revised_imglist_path):
    train_imgname, train_label = getlabelarray(train_imglist_path)
    assert len(train_imgname) == len(pred_labels), "train imglist should not equal to prediction list"
    sample_reweight = get_reweight_ratio(pred_labels)
    newlines = []
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

def get_reweight_ratio(pred_labels):
    try:
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

