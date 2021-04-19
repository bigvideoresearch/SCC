from .. import models


def get(model_name):
    names = model_name.split('.')
    if model_name.startswith('gluon.'):
        func = getattr(models.gluon, model_name[len('gluon.'):], None)
    else:
        func = getattr(models, model_name, None)
    if func is None:
        raise RuntimeError('model_name [{}] not exists'.format(model_name))
    return func
