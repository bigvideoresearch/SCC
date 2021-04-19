__all__ = ['ListImglist']


import re


class ListImglist:

    def __init__(self, path, root=None):
        self.root = root
        if path.startswith('ae://'):
            import runner_master.runner.distributed as run_dist
            import ae_toolbox.training.helpers as ae_helpers
            rank = run_dist.get_rank()
            path = ae_helpers.distributed_get_resource_by_ref(
                'dataset', path, rank=rank, resource_save_path='~/auto_evolution_resources',
            )
        elif re.fullmatch('(?:([^:]+):)?s3://([^/]+)/?(.*)', path):
            from petrel_tool import Downloader
            downloader = Downloader()
            path = downloader.download(path)
        with open(path) as f:
            self.lines = f.readlines()

    def __getitem__(self, index):
        if self.root is None:
            return self.lines[index]
        else:
            return self.root, self.lines[index]

    def __len__(self):
        return len(self.lines)
