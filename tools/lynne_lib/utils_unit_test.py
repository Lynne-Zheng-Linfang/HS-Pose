import utils
import os


def test_create_folder():
    folder_name = 'create_folder'
    exp_dir = os.path.dirname(__file__)
    folder_path = os.path.abspath(
        os.path.join(
            exp_dir,
            folder_name 
        )
    )
    if os.path.exists(folder_path):
        os.system('rm -rf {}'.format(folder_path))
    utils.create_folder_if_inexist(folder_path)
    print('Creating folder. ==>', os.path.exists(folder_path))
    assert os.path.exists(folder_path), 'Failed to create file' 
    if os.path.exists(folder_path):
        os.system('rm -rf {}'.format(folder_path))

def test_all():
    test_create_folder()

test_all()