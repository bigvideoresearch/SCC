import torch
from runner_master import runner

def load_ckpt_unstrict(network, ckpt, network_name='main'):
    state_dict = torch.load(ckpt, map_location='cuda:{}'.format(torch.cuda.current_device()))
    if 'network' in state_dict:
        state_dict = state_dict['network']
    if network_name in state_dict:
        state_dict = state_dict[network_name]
    network.load_state_dict(state_dict, strict=False)

runner.patch_network_init('load_ckpt_unstrict', load_ckpt_unstrict)

