import torch


old_load = torch.load


def new_load(f, *args, **kwargs):
    if isinstance(f, str) and f.startswith('ae://'):
        import runner_master.runner.distributed as run_dist
        import ae_toolbox.training.helpers as ae_helpers
        rank = run_dist.get_rank()
        f = ae_helpers.distributed_get_resource_by_ref(
            'model', f, rank=rank, resource_save_path='~/auto_evolution_resources',
        )
    return old_load(f, *args, **kwargs)


torch.load = new_load
