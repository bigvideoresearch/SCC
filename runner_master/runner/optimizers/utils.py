__all__ = ['check_optimizer']


def check_optimizer(optimizer):
    default_keys = set(optimizer.defaults.keys())
    for i, param_group in enumerate(optimizer.param_groups):
        group_keys = set(param_group.keys())
        group_keys.remove('params')
        if group_keys != default_keys:
            msg = []
            msg.append('Keys mismatch in optimizer.param_groups[{}]:'.format(i))
            msg.append('    default_keys: {}'.format(sorted(default_keys)))
            msg.append('    group_keys: {}'.format(sorted(group_keys)))
            msg.append('    only in default_keys: {}'.format(sorted(default_keys - group_keys)))
            msg.append('    only in group_keys: {}'.format(sorted(group_keys - default_keys)))
            msg.append('Seems there are some incorrect arguments for optimizer.')
            raise RuntimeError('\n'.join(msg))
