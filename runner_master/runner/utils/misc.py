__all__ = ['divide_metric_names']


def divide_metric_names(speed_dict, names):
    divide_names = {'fast': [], 'slow': []}
    for name in names:
        divide_names[speed_dict[name]].append(name)
    return divide_names['fast'], divide_names['slow']
