import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from time import perf_counter
from tqdm import tqdm

import sys
sys.path.append('..')
from labelers.offline_gnn import utils

def train_regression(model, train_features, train_labels, val_features,val_labels, epochs,
                     weight_decay, lr, loss_type='bce', bs=None, reweight=True, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    if reweight:
        reweight_list = utils.get_reweight_ratio((train_labels.data).cpu().numpy())
    else:
        reweight_list = np.ones(np.shape((train_labels.data).cpu().numpy()))
    reweight_list = torch.FloatTensor(reweight_list)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # shuffle data
        shuffle_idx = torch.randperm(data_length)
        if bs is None or data_length < bs:
            output = model(train_features[shuffle_idx].cuda())
            if loss_type == 'sce':
                loss_train = F.cross_entropy(output, train_labels[shuffle_idx].cuda())
            elif loss_type == 'bce':
                shuffled_train_softlabels = torch.zeros(output.size(0), output.size(1)).scatter_(1, train_labels[shuffle_idx].cpu().view(-1,1), 1).cuda()
                losses = F.binary_cross_entropy_with_logits(output, shuffled_train_softlabels, reduction='none')
                loss_train = (losses.sum(dim=1) * reweight_list[shuffle_idx].type_as(losses)).mean()
            else:
                print('Unknown Loss Type {}'.format(loss_type), flush=True)
            loss_train.backward()
            optimizer.step()
        else:
            if data_length % bs > 0:
                sub_epoch_num = data_length // bs + 1
            else:
                sub_epoch_num = data_length // bs
            for i in range(sub_epoch_num):
                optimizer.zero_grad()
                train_features_sub = train_features[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                train_labels_sub = train_labels[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                train_reweight_sub = reweight_list[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                output = model(train_features_sub)
                if loss_type == 'sce':
                    loss_train = F.cross_entropy(output, train_labels_sub)
                elif loss_type == 'bce':
                    train_softlabels_sub = torch.zeros(output.size(0), output.size(1)).scatter_(1,train_labels_sub.cpu().view(-1, 1), 1).cuda()
                    losses = F.binary_cross_entropy_with_logits(output, train_softlabels_sub, reduction='none')
                    loss_train = (losses.sum(dim=1) * train_reweight_sub.type_as(losses)).mean()
                else:
                    print('Unknown Loss Type {}'.format(loss_type), flush=True)
                loss_train.backward()
                optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features.cuda())
        #         acc_val = accuracy(output, val_labels.cuda())
        #         print('eval on train: {:.4f}'.format(acc_val))
        torch.cuda.empty_cache()

    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features.cuda())
        acc_val = accuracy(output, val_labels.cuda())
        print('eval on val: {:.4f}'.format(acc_val))
    if bs is None or data_length < bs:
        del optimizer, reweight_list, output, losses, loss_train, shuffled_train_softlabels
    else:
        del optimizer, reweight_list, output, losses, loss_train, train_features_sub, train_softlabels_sub, train_labels_sub, train_reweight_sub
    del train_features, train_labels, val_features,val_labels
    torch.cuda.empty_cache()
    return model, train_time, acc_val

def train_regression_fast(model, train_features, train_labels, val_features,val_labels, epochs,
                     weight_decay, lr, loss_type='bce', bs=None, reweight=True, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    if reweight:
        reweight_list = utils.get_reweight_ratio((train_labels.data).cpu().numpy())
    else:
        reweight_list = np.ones(np.shape((train_labels.data).cpu().numpy()))
    reweight_list = torch.FloatTensor(reweight_list).cuda()
    train_features = train_features.cuda()
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        if loss_type == 'sce':
            loss_train = F.cross_entropy(output, train_labels)
        elif loss_type == 'bce':
            train_softlabels = torch.zeros(output.size(0), output.size(1)).scatter_(1, train_labels.cpu().view(-1, 1),
                                                                                    1).cuda()
            losses = F.binary_cross_entropy_with_logits(output, train_softlabels, reduction='none')
            loss_train = (losses.sum(dim=1) * reweight_list.type_as(losses)).mean()
        else:
            print('Unknown Loss Type {}'.format(loss_type), flush=True)
        loss_train.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features.cuda())
        #         acc_val = accuracy(output, val_labels.cuda())
        #         print('eval on train: {:.4f}'.format(acc_val))

    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features.cuda())
        acc_val = accuracy(output, val_labels.cuda())
        print('eval on val: {:.4f}'.format(acc_val))
    del optimizer, reweight_list, output, losses, loss_train, train_softlabels
    del train_features, train_labels, val_features,val_labels
    torch.cuda.empty_cache()
    return model, train_time, acc_val

def test_regression(model, features, labels, idx_test, bs=None):
    model.eval()
    if bs is None or bs <= 0:
        output = model(features[idx_test].cuda())
        acc_test, pred_labels = accuracy(output.cuda(), labels[idx_test].cuda(), ret_preds=True)
        logits = output.detach().cpu()
    else:
        acc_test = 0
        pred_labels = []
        logits_list = torch.FloatTensor([])
        cnt = len(idx_test) // bs + 1
        for i in range(cnt):
            local_idx_test = idx_test[i * bs:(i + 1) * bs]
            if len(local_idx_test) <= 0:
                continue
            output_sub = model(features[local_idx_test].cuda())
            acc, preds = accuracy(output_sub, labels[local_idx_test].cuda(), ret_preds=True)
            acc_test += float(acc) * len(local_idx_test)
            pred_labels.extend(preds)
            logits_list = torch.cat([logits_list, output_sub.detach().cpu()])
            torch.cuda.empty_cache()
        acc_test /= len(idx_test)
        logits = torch.FloatTensor(logits_list)
    print("Test Accuracy: {:.4f}".format(acc_test),flush=True)
    return pred_labels, logits

def accuracy(output, labels, ret_preds=False, ret_topk=False, maxk=1):
    logit = F.softmax(output, dim=1)
    _, top_k = logit.topk(maxk, 1, True, True)
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    acc = correct / len(labels)
    if ret_topk:
        return acc, preds, logit, top_k
    if ret_preds:
        return acc, preds
    return acc

def train_regression_with_softlabel(model, train_features, train_labels, val_features, val_labels, reweight_list,
                                    epochs, weight_decay, lr, loss_type='bce', bs=None, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # shuffle data
        shuffle_idx = torch.randperm(data_length)
        if bs is None or data_length < bs:
            output = model(train_features[shuffle_idx].cuda())
            if loss_type == 'sce':
                loss_train = F.cross_entropy(output, train_labels[shuffle_idx].cuda())
            elif loss_type == 'bce':
                losses = F.binary_cross_entropy_with_logits(output, train_labels[shuffle_idx].cuda(), reduction='none')
                loss_train = (losses.sum(dim=1) * reweight_list[shuffle_idx].cuda().type_as(losses)).mean()
            else:
                print('Unknown Loss Type {}'.format(loss_type), flush=True)
            loss_train.backward()
            optimizer.step()
        else:
            if data_length % bs > 0:
                sub_epoch_num = data_length // bs + 1
            else:
                sub_epoch_num = data_length // bs
            for i in range(sub_epoch_num):
                optimizer.zero_grad()
                train_features_sub = train_features[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                train_labels_sub = train_labels[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                train_reweight_sub = reweight_list[shuffle_idx][i * bs:(i + 1) * bs].cuda()
                output = model(train_features_sub)
                if loss_type == 'sce':
                    loss_train = F.cross_entropy(output, train_labels_sub)
                elif loss_type == 'bce':
                    losses = F.binary_cross_entropy_with_logits(output, train_labels_sub, reduction='none')
                    loss_train = (losses.sum(dim=1) * train_reweight_sub.type_as(losses)).mean()
                else:
                    print('Unknown Loss Type {}'.format(loss_type), flush=True)
                loss_train.backward()
                optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features.cuda())
        #         acc_val = accuracy(output, val_labels.cuda())
        #         print('eval on train: {:.4f}'.format(acc_val),flush=True)
        torch.cuda.empty_cache()
    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features.cuda())
        acc_val = accuracy(output, val_labels.cuda())
        print('eval on val: {:.4f}'.format(acc_val),flush=True)
    if bs is None or data_length < bs:
        del optimizer, reweight_list, output, losses, loss_train
    else:
        del optimizer, reweight_list, output, losses, loss_train, train_features_sub, train_labels_sub, train_reweight_sub
    del train_features, train_labels, val_features,val_labels
    torch.cuda.empty_cache()
    return model, train_time, acc_val

def train_regression_with_softlabel_fast(model, train_features, train_labels, val_features, val_labels, reweight_list,
                                    epochs, weight_decay, lr, loss_type='bce', bs=None, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    train_features = train_features.cuda()
    train_labels = train_labels.cuda()
    reweight_list = reweight_list.cuda()
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        output = model(train_features)
        if loss_type == 'sce':
            loss_train = F.cross_entropy(output, train_labels)
        elif loss_type == 'bce':
            losses = F.binary_cross_entropy_with_logits(output, train_labels, reduction='none')
            loss_train = (losses.sum(dim=1) * reweight_list.type_as(losses)).mean()
        else:
            print('Unknown Loss Type {}'.format(loss_type), flush=True)
        loss_train.backward()
        optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features.cuda())
        #         acc_val = accuracy(output, val_labels.cuda())
        #         print('eval on train: {:.4f}'.format(acc_val))
    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features.cuda())
        acc_val = accuracy(output, val_labels.cuda())
        print('eval on val: {:.4f}'.format(acc_val))

    del optimizer, reweight_list, output, losses, loss_train
    del train_features, train_labels, val_features,val_labels
    return model, train_time, acc_val

def train_regression_nocuda(model, train_features, train_labels, val_features,val_labels, epochs,
                     weight_decay, lr, loss_type='bce', bs=None, reweight=True, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    if reweight:
        reweight_list = utils.get_reweight_ratio((train_labels.data).cpu().numpy())
    else:
        reweight_list = np.ones(np.shape((train_labels.data).cpu().numpy()))
    reweight_list = torch.FloatTensor(reweight_list)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # shuffle data
        shuffle_idx = torch.randperm(data_length)
        if bs is None or data_length < bs:
            output = model(train_features[shuffle_idx])
            if loss_type == 'sce':
                loss_train = F.cross_entropy(output, train_labels[shuffle_idx])
            elif loss_type == 'bce':
                shuffled_train_softlabels = torch.zeros(output.size(0), output.size(1)).scatter_(1, train_labels[shuffle_idx].view(-1,1), 1)
                losses = F.binary_cross_entropy_with_logits(output, shuffled_train_softlabels, reduction='none')
                loss_train = (losses.sum(dim=1) * reweight_list[shuffle_idx].type_as(losses)).mean()
            else:
                print('Unknown Loss Type {}'.format(loss_type), flush=True)
            loss_train.backward()
            optimizer.step()
        else:
            if data_length % bs > 0:
                sub_epoch_num = data_length // bs + 1
            else:
                sub_epoch_num = data_length // bs
            for i in range(sub_epoch_num):
                optimizer.zero_grad()
                train_features_sub = train_features[shuffle_idx][i * bs:(i + 1) * bs]
                train_labels_sub = train_labels[shuffle_idx][i * bs:(i + 1) * bs]
                train_reweight_sub = reweight_list[shuffle_idx][i * bs:(i + 1) * bs]
                output = model(train_features_sub)
                if loss_type == 'sce':
                    loss_train = F.cross_entropy(output, train_labels_sub)
                elif loss_type == 'bce':
                    train_softlabels_sub = torch.zeros(output.size(0), output.size(1)).scatter_(1,train_labels_sub.cpu().view(-1, 1), 1)
                    losses = F.binary_cross_entropy_with_logits(output, train_softlabels_sub, reduction='none')
                    loss_train = (losses.sum(dim=1) * train_reweight_sub.type_as(losses)).mean()
                else:
                    print('Unknown Loss Type {}'.format(loss_type), flush=True)
                loss_train.backward()
                optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features)
        #         acc_val = accuracy(output, val_labels)
        #         print('eval on train: {:.4f}'.format(acc_val))

    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
        print('eval on val: {:.4f}'.format(acc_val))
    if bs is None or data_length < bs:
        del optimizer, reweight_list, output, losses, loss_train, shuffled_train_softlabels
    else:
        del optimizer, reweight_list, output, losses, loss_train, train_features_sub, train_softlabels_sub, train_labels_sub, train_reweight_sub
    del train_features, train_labels, val_features,val_labels
    return model, train_time, acc_val

def test_regression_nocuda(model, features, labels, idx_test, bs=None):
    model.eval()
    if bs is None or bs <= 0:
        output = model(features)
        acc_test, pred_labels = accuracy(output[idx_test], labels[idx_test], ret_preds=True)
        logits = output.detach().cpu()
    else:
        acc_test = 0
        pred_labels = []
        logits_list = torch.FloatTensor([])
        cnt = len(idx_test) // bs + 1
        for i in range(cnt):
            local_idx_test = idx_test[i * bs:(i + 1) * bs]
            if len(local_idx_test) <= 0:
                continue
            output_sub = model(features[local_idx_test])
            acc, preds = accuracy(output_sub, labels[local_idx_test], ret_preds=True)
            acc_test += float(acc) * len(local_idx_test)
            pred_labels.extend(preds)
            logits_list = torch.cat([logits_list, output_sub.detach().cpu()])
        acc_test /= len(idx_test)
        logits = torch.FloatTensor(logits_list)
    print("Test Accuracy: {:.4f}".format(acc_test))
    return pred_labels, logits

def train_regression_with_softlabel_nocuda(model, train_features, train_labels, val_features, val_labels, reweight_list,
                                    epochs, weight_decay, lr, loss_type='bce', bs=None, dropout=0):
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)
    t = perf_counter()
    data_length = len(train_features)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        # shuffle data
        shuffle_idx = torch.randperm(data_length)
        if bs is None or data_length < bs:
            output = model(train_features[shuffle_idx])
            if loss_type == 'sce':
                loss_train = F.cross_entropy(output, train_labels[shuffle_idx])
            elif loss_type == 'bce':
                losses = F.binary_cross_entropy_with_logits(output, train_labels[shuffle_idx], reduction='none')
                loss_train = (losses.sum(dim=1) * reweight_list[shuffle_idx].type_as(losses)).mean()
            else:
                print('Unknown Loss Type {}'.format(loss_type), flush=True)
            loss_train.backward()
            optimizer.step()
        else:
            if data_length % bs > 0:
                sub_epoch_num = data_length // bs + 1
            else:
                sub_epoch_num = data_length // bs
            for i in range(sub_epoch_num):
                optimizer.zero_grad()
                train_features_sub = train_features[shuffle_idx][i * bs:(i + 1) * bs]
                train_labels_sub = train_labels[shuffle_idx][i * bs:(i + 1) * bs]
                train_reweight_sub = reweight_list[shuffle_idx][i * bs:(i + 1) * bs]
                output = model(train_features_sub)
                if loss_type == 'sce':
                    loss_train = F.cross_entropy(output, train_labels_sub)
                elif loss_type == 'bce':
                    losses = F.binary_cross_entropy_with_logits(output, train_labels_sub, reduction='none')
                    loss_train = (losses.sum(dim=1) * train_reweight_sub.type_as(losses)).mean()
                else:
                    print('Unknown Loss Type {}'.format(loss_type), flush=True)
                loss_train.backward()
                optimizer.step()
        # if epoch % 100 == 0:
        #     with torch.no_grad():
        #         model.eval()
        #         output = model(val_features)
        #         acc_val = accuracy(output, val_labels)
        #         print('eval on train: {:.4f}'.format(acc_val))
    train_time = perf_counter() - t
    with torch.no_grad():
        model.eval()
        output = model(val_features)
        acc_val = accuracy(output, val_labels)
        print('eval on val: {:.4f}'.format(acc_val))
    if bs is None or data_length < bs:
        del optimizer, reweight_list, output, losses, loss_train
    else:
        del optimizer, reweight_list, output, losses, loss_train, train_features_sub, train_labels_sub, train_reweight_sub
    del train_features, train_labels, val_features,val_labels
    return model, train_time, acc_val
