__all__ = ['IndexSingleLabel', 'OnehotSingleLabel', 'SingleLabel', 'SmoothSingleLabel']


import json
import torch


class IndexSingleLabel:

    def __init__(self, num_classes=None):
        self.num_classes = num_classes

    def __call__(self, label):
        if type(label) not in (int, float, str):
            raise TypeError('invalid type {} for input label {}'.format(type(label), label))
        label_index = int(label)
        if self.num_classes is not None and \
                (label_index < 0 or label_index >= self.num_classes):
            raise ValueError('label_index not in range [0, {})'.format(self.num_classes))
        return label_index


class OnehotSingleLabel:

    def __init__(self, num_classes, smooth_epsilon=0):
        self.index_single_label = IndexSingleLabel()
        self.num_classes = num_classes
        self.smooth_epsilon = smooth_epsilon
        self.pos_prob = 1 - smooth_epsilon
        self.neg_prob = smooth_epsilon / (num_classes - 1)

    def __call__(self, label):
        label_index = self.index_single_label(label)
        final_label = torch.Tensor(self.num_classes)
        final_label.fill_(self.neg_prob)
        final_label[label_index] = self.pos_prob
        return final_label


SingleLabel = OnehotSingleLabel


class SmoothSingleLabel:

    def __init__(self, epsilon, num_classes, keymap={}):
        self.pos_prob = 1 - epsilon
        self.neg_prob = epsilon / (num_classes - 1)
        self.num_classes = num_classes
        self.keymap = keymap

    def __call__(self, extra_str):
        result = json.loads(extra_str)
        if isinstance(result, int):
            result = {'label': result}
        elif isinstance(result, dict):
            result = {
                self.keymap.get(key, key): value
                for key, value in result.items()
            }
        else:
            raise RuntimeError('unknown type: {}'.format(type(result)))
        soft_label = torch.Tensor(self.num_classes)
        soft_label.fill_(self.neg_prob)
        soft_label[result['label']] = self.pos_prob
        result['soft_label'] = soft_label
        return result
