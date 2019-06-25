import os
import shutil


def ensure_dir_exists_and_empty(dir):
    assert dir, 'Directory is not specified'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)

def ensure_dir_exists(dir):
    assert dir, 'Directory is not specified'
    if not os.path.exists(dir):
        os.makedirs(dir)