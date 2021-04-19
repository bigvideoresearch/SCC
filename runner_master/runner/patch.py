from runner_master import runner

def patch(module, name, stuff):
    '''给runner_master.runner打猴子补丁的通用接口
    '''
    if hasattr(module, name):
        raise RuntimeError('\n'.join([
            'Attribute [{}] already in module [{}], monkey patching is not allowed.'.format(name, module),
            'If you insist, use manual code like [module.attribute_name = new_attribute_value].',
        ]))
    if stuff is None:
        def decorator(stuff):
            setattr(module, name, stuff)
            return stuff
        return decorator
    else:
        setattr(module, name, stuff)


def patch_model(name, stuff=None):
    return patch(runner.models, name, stuff)


def patch_super_block(name, stuff=None):
    return patch(runner.models.nas.super_block, name, stuff)


def patch_optimizer(name, stuff=None):
    return patch(runner.optimizers, name, stuff)


def patch_scheduler(name, stuff=None):
    return patch(runner.schedulers, name, stuff)


def patch_imglist(name, stuff=None):
    return patch(runner.data.imglists, name, stuff)


def patch_filereader(name, stuff=None):
    return patch(runner.data.filereaders, name, stuff)


def patch_dataset(name, stuff=None):
    return patch(runner.data.datasets, name, stuff)


def patch_sampler(name, stuff=None):
    return patch(runner.data.samplers, name, stuff)


def patch_transform_line(name, stuff=None):
    return patch(runner.transforms.line, name, stuff)


def patch_transform_image(name, stuff=None):
    return patch(runner.transforms.image, name, stuff)


def patch_transform_extra(name, stuff=None):
    return patch(runner.transforms.extra, name, stuff)

patch_transform_label = patch_transform_extra


def patch_transform_batch(name, stuff=None):
    return patch(runner.transforms.batch, name, stuff)


patch_transform_collate = patch_transform_batch


def patch_param_group(name, stuff=None):
    return patch(runner.optimizers.param_group, name, stuff)


def patch_metric(name, stuff=None):
    return patch(runner.metrics, name, stuff)


def patch_ruler(name, stuff=None):
    return patch(runner.pipelines.rulers, name, stuff)


def patch_pipeline(name, stuff=None):
    return patch(runner.pipelines, name, stuff)


def patch_network_init(name, stuff=None):
    return patch(runner.network_initializers, name, stuff)



