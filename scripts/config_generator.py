#!/usr/bin/env python
import sys
import os

from jinja2 import Environment, FileSystemLoader

template = None

def process_dir(path, regenerate_config=False):
    sdfs = [f for f in os.listdir(path) if f[-4:].lower() == '.sdf']
    sub_dirs = {d for d in os.listdir(path) if os.path.isdir(d)}
    if len(sdfs) == 1:
        if regenerate_config or not os.path.isfile('{}/model.config'.format(path)):
            with open('{}/model.config'.format(path), 'w') as cf:
                cf.write(template.render(name=sdfs[0][:-4], file=sdfs[0]))
    elif len(sdfs) > 1:
        for f in sdfs:
            subdir = '{}/{}'.format(path, f[:-4])
            if not os.path.isdir(subdir):
                os.mkdir(subdir)
            os.rename('{}/{}'.format(path, f), '{}/{}'.format(subdir, f))
            process_dir(subdir, regenerate_config)

    for d in sub_dirs:
        process_dir(d, regenerate_config)


if __name__ == '__main__':
    
    template_fn = 'model_config.template' if len(sys.argv) < 2 else sys.argv[1]

    file_loader = FileSystemLoader('.')
    env = Environment(loader=file_loader)

    template = env.get_template(template_fn)

    process_dir('.', False)