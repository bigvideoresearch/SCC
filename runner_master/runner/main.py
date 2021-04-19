__all__ = ['main']


import os
import sys as _sys
from importlib import import_module
from collections import OrderedDict
from runner_master import runner
def main():
    os.environ['PYTHONUNBUFFERED'] = 'x'  # equivalent to python -u
    config, rank, world_size = runner.setup()

    # dynamically import
    import_packages = config.get('import', [])
    error_msg = OrderedDict()
    if isinstance(import_packages, dict):
        import_packages = [key for key, value in import_packages.items() if value]
    for package in import_packages:
        try:
            import_module(package)
        except Exception as e:
            error_msg[package] = repr(e)
    if len(error_msg) > 0:
        print('raise errors when importing these packages:')
        for package, msg in error_msg.items():
            print('[{}]: {}'.format(package, msg))
        print('the original import arguments are:')
        print(config.get('import', []))
        _sys.exit()

    # dynamically create and run pipeline
    lib = import_module('runner_master.runner.pipelines')
    for i, token in enumerate(config.get('pipeline', 'LearnPipelineV2').split('.')):
        lib = getattr(lib, token)
    pipeline = lib(config=config, rank=rank, world_size=world_size)
    pipeline.run()


if __name__ == '__main__':
    main()
