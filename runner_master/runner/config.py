'''config模块用于支持通过配置文件和命令行选项灵活配置超参数。
'''
__all__ = ['Config', 'setup_config', 'print_config']


import re
import os
import sys
import json5
import argparse
import oyaml as yaml
from collections import defaultdict, deque, OrderedDict

CFG_DIR = os.path.join(sys.path[0], 'runner_master/runner/cfgs')

###############################################################
# utility functions
###############################################################


def consume_dots(config, key, create_default):
    sub_keys = key.split('.', 1)
    sub_key = sub_keys[0]

    if sub_key in Config.__dict__:
        raise KeyError('"{}" is a preserved API name for runner.Config, '
                       'should not be used as normal dictionary key'.format(sub_key))

    if not OrderedDict.__contains__(config, sub_key) and len(sub_keys) == 2:
        if create_default:
            config[sub_key] = Config()
            # OrderedDict.__setitem__(config, sub_key, Config())
        else:
            raise KeyError(key)

    if len(sub_keys) == 1:
        return config, sub_key
    else:
        sub_config = OrderedDict.__getitem__(config, sub_key)
        if type(sub_config) != Config:
            if create_default:
                sub_config = Config()
                config[sub_key] = sub_config
                # OrderedDict.__setitem__(config, sub_key, sub_config)
            else:
                raise KeyError(key)
        return consume_dots(sub_config, sub_keys[1], create_default)


def traverse_dfs(root, mode, continue_type, only_leaf, key_prefix=''):
    for key, value in root.items():
        full_key = '.'.join([key_prefix, key]).strip('.')
        child_kvs = []
        if isinstance(value, continue_type):
            for kv in traverse_dfs(value, mode, continue_type, only_leaf, full_key):
                child_kvs.append(kv)
        # equivalent: if not (len(child_kvs) > 0 and type(value) == continue_type and only_leaf)
        if (len(child_kvs) == 0) or (not isinstance(value, continue_type)) or (not only_leaf):
            yield {'key': full_key, 'value': value, 'item': (full_key, value)}[mode]
        for kv in child_kvs:
            yield kv


# bfs is not convenient for checking whether a node is empty, deprecated
# def traverse_bfs(root, mode, continue_type, only_leaf):
#     q = [(root, '')]
#     while len(q) > 0:
#         child, key_prefix = q.pop(0)
#         for key, value in child.items():
#             full_key = '.'.join([key_prefix, key]).strip('.')
#             if type(value) != continue_type or not only_leaf:
#                 yield {'key': full_key, 'value': value, 'item': (full_key, value)}[mode]
#             if type(value) == continue_type:
#                 q.append((value, full_key))


def init_assign_dict(config, d):
    for full_key, value in traverse_dfs(d, 'item', continue_type=dict, only_leaf=True):
        sub_cfg, sub_key = consume_dots(config, full_key, create_default=True)
        sub_cfg[sub_key] = value


def init_assign_str(config, cfg_name, load_history=None, included_from=None):
    if load_history is None:
        load_history = set()
    if included_from is None:
        cfg_path1 = cfg_name
    else:
        cfg_path1 = os.path.join(os.path.dirname(included_from), cfg_name)
    cfg_path2 = os.path.join(CFG_DIR, cfg_name)

    if os.path.exists(cfg_path1):
        cfg_path = cfg_path1
    elif os.path.exists(cfg_path2):
        cfg_path = cfg_path2
    else:
        msg = '\n'.join([
            'config files not exist for cfg_name [{}]:'.format(cfg_name),
            '(1): [{}]'.format(cfg_path1),
            '(2): [{}]'.format(cfg_path2),
            'at least one of them should exist',
        ])
        raise RuntimeError(msg)

    if cfg_path in load_history:
        return
    load_history.add(cfg_path)

    with open(cfg_path) as f:
        for line in f:
            if line.startswith('#!include'):
                for included_cfg_name in line.strip().split()[1:]:
                    init_assign_str(config, included_cfg_name, load_history, included_from=cfg_path)

    if cfg_path.endswith('.json') or cfg_path.endswith('.json5'):
        with open(cfg_path) as f:
            raw_dict = json5.load(f)
    elif cfg_path.endswith('.yaml'):
        with open(cfg_path) as f:
            raw_dict = yaml.load(f)
    else:
        raise Exception('unknown file format %s' % cfg_path)
    if raw_dict is None:
        raw_dict = {}
    raw_dict = OrderedDict(raw_dict)
    init_assign_dict(config, raw_dict)
    config.parse_refs(level=2)


def topological_sort(graph):
    # record each node's indegree
    indegree_map = defaultdict(int)  # node to its out-degree
    for node1, node2s in graph.items():
        indegree_map[node1] += 0
        for node2 in node2s:
            indegree_map[node2] += 1
    indegree_map = dict(indegree_map)

    # first collect all zero-indigree nodes
    zero_indegree_nodes = deque()
    for node, indegree in list(indegree_map.items()):
        if indegree == 0:
            zero_indegree_nodes.append(node)
            del indegree_map[node]

    # main loop of topological sorting
    sorted_nodes = []
    while len(zero_indegree_nodes) > 0:
        node1 = zero_indegree_nodes.popleft()
        sorted_nodes.append(node1)
        if node1 not in graph:
            continue  # no other nodes are refered by node1
        for node2 in graph[node1]:
            indegree_map[node2] -= 1
            if indegree_map[node2] == 0:
                zero_indegree_nodes.append(node2)
                del indegree_map[node2]

    # make sure no cycle exists
    if len(indegree_map) > 0:
        msg = ['Find cycle in the edges:']
        for node1, node2s in sorted(graph.items()):
            if len(node2s) == 1:
                msg.append('{} -> {}'.format(node1, node2s[0]))
            else:
                msg.append(node1)
                for node2 in node2s:
                    msg.append('  -> {}'.format(node2))
        raise RuntimeError('\n'.join(msg))

    return sorted_nodes


###############################################################
# main class
###############################################################


class Config(OrderedDict):
    '''Config是runner_master.runner中处理配置文件、命令行选项、训练超参的核心接口
    '''

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__()
        for arg in args:
            if isinstance(arg, str):
                init_assign_str(self, arg)
            elif isinstance(arg, dict):
                init_assign_dict(self, arg)
            else:
                raise TypeError('arg should be an instance of <str> or <dict>')
        if kwargs:
            init_assign_dict(self, kwargs)

    def __call__(self, *args, **kwargs):
        return Config(self, *args, **kwargs)

    def __repr__(self, indent=4, prefix=''):
        r = []
        for key, value in self.items():
            if isinstance(value, Config):
                r.append('{}{}:'.format(prefix, key))
                r.append(value.__repr__(indent, prefix + ' ' * indent))
            else:
                # handling special cases in which yaml dumping is not reversable
                if value is None:
                    value = 'null'
                elif value is '':
                    value = "''"
                elif value is True:
                    value = 'true'
                elif value is False:
                    value = 'false'
                r.append('{}{}: {}'.format(prefix, key, value))
        return '\n'.join(r)

    ###########################################################
    # support for pickle
    ###########################################################

    def __setstate__(self, state):
        init_assign_dict(self, state)

    def __getstate__(self):
        d = OrderedDict()
        for key, value in self.items():
            if type(value) is Config:
                value = value.__getstate__()
            d[key] = value
        return d

    ###########################################################
    # access by '.' -> access by '[]'
    ###########################################################

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]

    ###########################################################
    # access by '[]'
    ###########################################################

    def __getitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        return OrderedDict.__getitem__(sub_cfg, sub_key)

    def __setitem__(self, key, value):
        sub_cfg, sub_key = consume_dots(self, key, create_default=True)
        if sub_key == '__clear__' and value == True:
            sub_cfg.clear()
        elif value in ('__remove__',):
            if sub_cfg.__contains__(sub_key):
                OrderedDict.__delitem__(sub_cfg, sub_key)
        else:
            OrderedDict.__setitem__(sub_cfg, sub_key, value)

    def __delitem__(self, key):
        sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        OrderedDict.__delitem__(sub_cfg, sub_key)
        # del self.__dict__[key]

    ###########################################################
    # access by 'in'
    ###########################################################

    def __contains__(self, key):
        try:
            sub_cfg, sub_key = consume_dots(self, key, create_default=False)
        except KeyError:
            return False
        return OrderedDict.__contains__(sub_cfg, sub_key)

    ###########################################################
    # traverse keys / values/ items
    ###########################################################

    def all_keys(self, only_leaf=True):
        for key in traverse_dfs(self, 'key', continue_type=Config, only_leaf=only_leaf):
            yield key

    def all_values(self, only_leaf=True):
        for value in traverse_dfs(self, 'value', continue_type=Config, only_leaf=only_leaf):
            yield value

    def all_items(self, only_leaf=True):
        for key, value in traverse_dfs(self, 'item', continue_type=Config, only_leaf=only_leaf):
            yield key, value

    ###########################################################
    # for command line arguments
    ###########################################################

    def parse_args(self, cmd_args=None, strict=True):
        unknown_args = []
        if cmd_args is None:
            import sys
            cmd_args = sys.argv[1:]
        index = 0
        while index < len(cmd_args):
            arg = cmd_args[index]
            err_msg = 'invalid command line argument pattern: %s' % arg
            assert arg.startswith('--'), err_msg
            assert len(arg) > 2, err_msg
            assert arg[2] != '-', err_msg

            arg = arg[2:]
            if '=' in arg:
                key, full_value_str = arg.split('=')
                index += 1
            else:
                assert len(cmd_args) > index + 1, 'incomplete command line arguments'
                key = arg
                full_value_str = cmd_args[index + 1]
                index += 2
            if ':' in full_value_str:
                value_str, value_type_str = full_value_str.rsplit(':', maxsplit=1)
                if value_type_str in ['str', 'book', 'int', 'float', 'dict', 'list', 'tuple', 'bool']:
                    value_type = eval(value_type_str)
                else:
                    value_str = full_value_str
                    value_type = None
            else:
                value_str = full_value_str
                value_type = None

            if key not in self:
                if strict:
                    raise KeyError(key)
                else:
                    unknown_args.extend(['--' + key, full_value_str])
                    continue

            if value_type is None:
                value_type = type(self[key])

            if value_type is bool:
                self[key] = {
                    'true': True,
                    'True': True,
                    '1': True,
                    'false': False,
                    'False': False,
                    '0': False,
                }[value_str]
            else:
                self[key] = value_type(value_str)

        return unknown_args

    ###########################################################
    # for key reference
    ###########################################################

    # def parse_refs(self, subconf=None, stack_depth=1, max_stack_depth=10):
    #     if stack_depth > max_stack_depth:
    #         raise Exception((
    #             'Recursively calling `parse_refs` too many times with stack depth > {}. '
    #             'A circular reference may exists in your config.\n'
    #             'If deeper calling stack is really needed, please call `parse_refs` '
    #             'with extra argument like: `parse_refs(max_stack_depth=9999)`'
    #         ).format(max_stack_depth))
    #     if subconf is None:
    #         subconf = self
    #     for key in subconf.keys():
    #         value = subconf[key]
    #         if type(value) is str and value.startswith('@{') and value.endswith('}'):
    #             ref_key = value[2:-1]
    #             ref_value = self[ref_key]
    #             if type(ref_value) is str and ref_value.startswith('@{') and value.endswith('}'):
    #                 raise Exception('Refering key %s to %s, but the value of %s is another reference value %s' % (
    #                     repr(key), repr(value), repr(ref_key), repr(ref_value),
    #                 ))
    #             subconf[key] = ref_value
    #     for key in subconf.keys():
    #         value = subconf[key]
    #         if type(value) is Config:
    #             self.parse_refs(value, stack_depth + 1)

    def parse_refs(self, level=1):
        ref_graph = defaultdict(list)
        ref_type = dict()
        assert level in (1, 2)
        if level == 1:
            object_pattern = re.compile('@\{([^@\$\{\}]*)\}')  # noqa
            string_pattern = re.compile('\$\{([^@\$\{\}]*)\}')  # noqa
        elif level == 2:
            object_pattern = re.compile('@@\{([^@\$\{\}]*)\}')  # noqa
            string_pattern = re.compile('\$\$\{([^@\$\{\}]*)\}')  # noqa

        # extract all reference / containment relations, save into `ref_graph`
        for key, value in self.all_items(only_leaf=True):
            if not isinstance(value, str):
                continue
            object_pattern_fullmatch = re.fullmatch(object_pattern, value)
            object_pattern_search = re.search(object_pattern, value)
            string_pattern_findall = re.findall(string_pattern, value)
            if object_pattern_fullmatch is None and object_pattern_search is not None:
                raise RuntimeError('Pattern "@{{xxx}}" or "@@{{xxx}}" should be used alone, '
                                   'but got: {}'.format(value))

            if object_pattern_fullmatch:
                assert len(string_pattern_findall) == 0
                ref_key = object_pattern_fullmatch.groups()[0]
                if ref_key not in self:
                    raise KeyError('key [{}] refer to an non-existent key [{}]'.format(key, ref_key))
                ref_value = self[ref_key]
                ref_graph[key].append(ref_key)
                ref_type[key] = 'object'
                if isinstance(ref_value, Config):
                    for sub_key, sub_value in ref_value.all_items(only_leaf=True):
                        if not isinstance(sub_value, str):
                            continue
                        if re.fullmatch(object_pattern, sub_value) or re.search(string_pattern, sub_value):
                            ref_graph[ref_key].append(ref_key + '.' + sub_key)
                            ref_type[ref_key] = 'contain'
            elif len(string_pattern_findall) > 0:
                for p in string_pattern_findall:
                    ref_key = p.split(':')[0]
                    if ref_key not in self:
                        raise KeyError('key [{}] refer to an non-existent key [{}]'.format(key, ref_key))
                    ref_graph[key].append(ref_key)
                    ref_type[key] = 'string'
        ref_graph = dict(ref_graph)

        # parse the references according to reverse topological order
        sorted_keys = topological_sort(ref_graph)
        for key in reversed(sorted_keys):
            if key not in ref_type:
                continue  # this key is only referenced by other keys, no need to parse

            if ref_type[key] == 'contain':
                continue
            elif ref_type[key] == 'object':
                ref_key = ref_graph[key][0]
                ref_value = self[ref_key]
                if isinstance(ref_value, Config):
                    self[key] = Config()
                    init_assign_dict(self[key], ref_value)
                else:
                    self[key] = ref_value
            elif ref_type[key] == 'string':
                ref_keys = ref_graph[key]
                format_args = []
                for ref_key in ref_keys:
                    ref_value = self[ref_key]
                    format_args.append(ref_value)
                value = self[key]
                value = re.sub('\$\{[^\$\{\}:]*(:[^\$\{\}:]*)?\}', '{\\1}', value)  # noqa
                value = value.format(*format_args)
                self[key] = value
            else:
                raise RuntimeError


###############################################################
# warpper for outside calling
###############################################################


def setup_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('-cfg', dest='config', nargs='+', required=True)
    opt, unknown_args = parser.parse_known_args()
    config = Config(*opt.config)
    config.parse_args(unknown_args)
    config.parse_refs(level=1)
    return config


def print_config():
    print(setup_config(), flush=True)
