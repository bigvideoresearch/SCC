__all__ = ['topk_accuracy']


def topk_accuracies(output, label, ks=(1,), reduce=True, class_average=False):
    assert output.dim() == 2
    assert label.dim() == 1
    assert output.size(0) == label.size(0)

    maxk = max(ks)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    expand_label = label.unsqueeze(1).expand_as(pred)
    correct = pred.eq(expand_label).float()

    accu_list = []
    for k in ks:
        accu = correct[:, :k].sum(1)
        if reduce:
            if class_average:
                class_accu_list = []
                for i in label.unique(sorted=True):
                    positions = (label == i).nonzero().view(-1)
                    class_accu = accu.index_select(0, positions)
                    class_accu_list.append(class_accu.mean())
                accu = sum(class_accu_list) / len(class_accu_list)
            else:
                accu = accu.mean()
        accu_list.append(accu)
    return accu_list


def topk_accuracy(output, label, k, reduce=True, class_average=False):
    return topk_accuracies(output, label, (k,), reduce, class_average)[0]

