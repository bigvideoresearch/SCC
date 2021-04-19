import os
import torch
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

import sys
sys.path.append('..')
import utils
import model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saveroot', type=str, default='', help='Job name for file saving, e.g. ./google500_sgc1.')

    # Use text feature for graph building and select seeds.
    parser.add_argument('--text_topk', type=int, default=10, help='Select top k seeds from each class.')
    parser.add_argument("--graph_method", type=str, default='gpu', help="Build graph on CPU/GPU/Approximate_GPU")
    parser.add_argument("--use_multisource",  type=str, default='False', help="Select seed from Google and Flickr")
    parser.add_argument("--text_dist_def", type=str, default='cosine', help="Cosine or Euclidean dist for Text Graph")
    parser.add_argument('--text_k', type=int, default=5, help='k of kNN for text graph building.')
    parser.add_argument("--text_self_weight", type=float, default=0, help="Self weight when aggregation on text graph")
    parser.add_argument("--text_edge_weight", type=str, default='False', help="Aggregate edge weight on text graph")

    parser.add_argument('--text_imglist_path', type=str, default='', help='text imglist path.')
    parser.add_argument('--text_graph_feature_path', type=str, default='', help='text graph feature path.')
    parser.add_argument('--text_feature_path', type=str, default='', help='text feature path.')
    parser.add_argument('--label_des_path', type=str, default='', help='label description feature path.')

    # Build graph for SGC training
    parser.add_argument('--dist_def', type=str, default='', help='Cosine or Euclidean dist for SGC Graph')
    parser.add_argument('--k', type=int, default=5, help='k of kNN for SGC graph building.')
    parser.add_argument("--self_weight", type=float, default=0, help="Self weight when aggregation on text graph")
    parser.add_argument("--edge_weight", type=str, default='True', help="Use edge weight when aggregation SGC graph")

    # Training and Val features path
    parser.add_argument('--train_imglist_path', type=str, default='', help='SGC training imglist path.')
    parser.add_argument('--train_feature_path', type=str, default='', help='SGC training feature path.')
    parser.add_argument('--train_graph_feature_path', type=str, default='', help='SGC training graph feature path.')

    parser.add_argument('--val_imglist_path', type=str, default='', help='SGC val imglist path.')
    parser.add_argument('--val_feature_path', type=str, default='', help='SGC val feature path.')
    parser.add_argument('--val_graph_feature_path', type=str, default='', help='SGC val graph feature path.')

    # Training config
    parser.add_argument('--cat_ori_feat', type=str, default='True', help='Concatenate aggregation and origin features.')
    parser.add_argument('--epochs_num', type=int, default=5000, help='Total epoches to train.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate ratio.')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay rate')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')
    parser.add_argument('--batch_size', type=int, default=10000, help='Number of Batch size.')
    parser.add_argument('--loss_type', type=str, default='bce', help='SCE loss or BCE loss.')
    parser.add_argument('--reweight', type=str, default='True', help='Use Reweight or not.')

    args = parser.parse_args()

    ###################################################################################################################
    #                                      Select seed based on text feature                                          #
    ###################################################################################################################
    # Config
    save_root = args.saveroot
    text_topk = args.text_topk
    graph_method = args.graph_method
    use_multisource = (args.use_multisource == 'True')
    text_dist_def = args.text_dist_def
    text_k = args.text_k
    text_self_weight = args.text_self_weight
    text_edge_weight = (args.text_edge_weight == 'True')
    # file path
    text_imglist_path = args.text_imglist_path
    text_graph_feature_path = args.text_graph_feature_path
    text_feature_path = args.text_feature_path
    label_des_path = args.label_des_path
    os.makedirs(save_root, exist_ok=True)
    ###################################################################################################################
    if text_edge_weight:
        save_seed_idxname = '{}/seed_{}knn_k{}_sw{}_top{}.npy'.format(save_root, text_dist_def, text_k,
                                                                      text_self_weight, text_topk)
    else:
        save_seed_idxname = '{}/seed_{}knn_k{}_sw{}negw_top{}.npy'.format(save_root, text_dist_def, text_k,
                                                                          text_self_weight, text_topk)

    textdf_file_path = os.path.join(save_root, 'data_frame.pkl')
    if os.path.exists(textdf_file_path):
        df = pd.read_pickle(textdf_file_path)
    else:
        df = utils.create_df(text_imglist_path, textdf_file_path)
    class_num = np.unique(np.array(df['y']))
    if os.path.exists(save_seed_idxname):
        seed_index = np.load(save_seed_idxname)
        print('Read Seed File from {}.'.format(save_seed_idxname), flush=True)
    else:
        if text_edge_weight:
            save_textknn_filename = '{}/textknn_{}knn_k{}_sw{}'.format(save_root, text_dist_def, text_k,
                                                                       text_self_weight)
        else:
            save_textknn_filename = '{}/textknn_{}knn_k{}_sw{}negw'.format(save_root, text_dist_def, text_k,
                                                                           text_self_weight)
        # select seed and return index
        if text_k > 0:
            if use_multisource:
                df = utils.get_knn_conf_multisource(df, text_feature_path, text_graph_feature_path, label_des_path,
                                                text_imglist_path, class_num, save_textknn_filename, text_dist_def,
                                                text_k, text_topk, text_self_weight, text_edge_weight, graph_method)
            else:
                df = utils.get_knn_conf(df, text_feature_path, text_graph_feature_path, label_des_path,
                                        text_imglist_path, class_num, save_textknn_filename, text_dist_def, text_k,
                                        text_topk, text_self_weight, text_edge_weight, graph_method)   
        else:
            df = utils.get_conf(df, text_feature_path, label_des_path, text_imglist_path, class_num, text_topk)
        seed_index = np.array(df[df['seed'] != 10000].index)
        # save seed index
        np.save(save_seed_idxname, seed_index)

        # save seed as imglist
        save_seed_imglist = save_seed_idxname.replace('.npy', '.txt')
        if (not os.path.exists(save_seed_imglist)):
            df = pd.read_pickle(textdf_file_path)
            imgname_seed = np.array(df['img_name'])[seed_index]
            label_seed = np.array(df['y'])[seed_index]
            with open(save_seed_imglist, 'a+') as writer:
                for imgname, label in zip(imgname_seed, label_seed):
                    newline = imgname + ' ' + str(label) + '\n'
                    writer.write(newline)

    ###################################################################################################################
    #                                        Preparing features for SGC                                               #
    ###################################################################################################################
    dist_def = args.dist_def
    k = args.k
    self_weight = args.self_weight
    edge_weight = (args.edge_weight == 'True')

    # concatenate val features
    train_imglist_path = args.train_imglist_path
    train_feature_path = args.train_feature_path
    train_graph_feature_path = args.train_graph_feature_path

    val_imglist_path = args.val_imglist_path
    val_feature_path = args.val_feature_path
    val_graph_feature_path = args.val_graph_feature_path
    ##################################################################################################################
    imglist, feature, graph_feature, label, train_idx, val_idx = utils.data_slicing(train_imglist_path,
                                                                                    train_feature_path,
                                                                                    train_graph_feature_path,
                                                                                    val_imglist_path,
                                                                                    val_feature_path,
                                                                                    val_graph_feature_path)
    if edge_weight:
        save_knn_filename = '{}/knn_{}knn_k{}_sw{}'.format(save_root, dist_def, k, self_weight)
    else:
        save_knn_filename = '{}/knn_{}knn_k{}_sw{}negw'.format(save_root, dist_def, k, self_weight)

    # training preparation finished
    knn_graph = utils.global_build(graph_feature, dist_def, k, save_knn_filename, graph_method)
    agg_feature = utils.sgc_precompute(feature, knn_graph, self_weight=self_weight, edge_weight=edge_weight, degree=1)

    ###################################################################################################################
    #                                          Training period                                                        #
    ###################################################################################################################
    # config
    cat_ori_feat = (args.cat_ori_feat == 'True')
    epochs_num = args.epochs_num
    lr = args.lr
    weight_decay = args.weight_decay
    dropout = args.dropout
    batch_size = args.batch_size
    loss_type = args.loss_type
    reweight = (args.reweight == 'True')
    ##################################################################################################################
    revised_imglist_path = '{}/sgc_revised_imglist.txt'.format(save_root)
    revised_val_imglist_path = '{}/sgc_revised_val_imglist.txt'.format(save_root)
    revised_score_path = '{}/sgc_revised_score.npy'.format(save_root)
    revised_val_score_path = '{}/sgc_revised_val_score.npy'.format(save_root)
    model_path = '{}/sgc_model.pth'.format(save_root)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs(os.path.dirname(revised_imglist_path), exist_ok=True)


    #################
    # a labeling mapping trick only use in noisy10
    if graph_method == 'cpu':
        for map_label, real_label in enumerate(np.unique(label)):
            label[np.where(label==real_label)]=map_label
    #################

    feature = torch.FloatTensor(feature)
    agg_feature = torch.FloatTensor(agg_feature)
    label = torch.LongTensor(label)

    if cat_ori_feat:
        features = torch.cat([feature, agg_feature], dim=-1)
    else:
        features = agg_feature

    use_cuda = (graph_method != 'cpu')
    if os.path.exists(model_path):
        sgc_model = torch.load(model_path).cuda()
    else:
        cls_num = len(class_num)
        sgc_model = model.get_model('SGC', features.size(1), cls_num, nhid=0, dropout=dropout, cuda=use_cuda)
        if graph_method != 'cpu':
            sgc_model, train_time_0, acc_val_0 = model.train_regression(sgc_model, features[seed_index], label[seed_index],
                                                                        features[val_idx],
                                                                        label[val_idx], epochs_num, weight_decay, lr,
                                                                        loss_type, batch_size, reweight, dropout)
        else:
            sgc_model, train_time_0, acc_val_0 = model.train_regression_nocuda(sgc_model, features[seed_index],
                                                                               label[seed_index],
                                                                               features[val_idx],
                                                                               label[val_idx], epochs_num, weight_decay, lr,
                                                                               loss_type, batch_size, reweight, dropout)
        torch.save(sgc_model, model_path)
    # save train pseudo labels
    if graph_method != 'cpu':
        pred_labels, pred_logits = model.test_regression(sgc_model, features, label, train_idx, batch_size)
    else:
        pred_labels, pred_logits = model.test_regression_nocuda(sgc_model, features, label, train_idx, batch_size)
    pred_labels = np.array(torch.tensor(pred_labels).cpu())
    pred_logits = np.array(torch.sigmoid(pred_logits))
    utils.get_revised_imglist(train_imglist_path, pred_labels, revised_imglist_path)
    np.save(revised_score_path, pred_logits)
    # save val set
    if graph_method != 'cpu':
        val_pred_labels, val_logits = model.test_regression(sgc_model, features, label, val_idx, batch_size)
    else:
        val_pred_labels, val_logits = model.test_regression_nocuda(sgc_model, features, label, val_idx, batch_size)
    val_pred_labels = np.array(torch.tensor(val_pred_labels).cpu())
    val_logits = np.array(torch.sigmoid(val_logits))
    utils.get_revised_imglist(val_imglist_path, val_pred_labels, revised_val_imglist_path)
    np.save(revised_val_score_path, val_logits)
    print('Successfully save revised imglist {}'.format(revised_imglist_path), flush=True)

