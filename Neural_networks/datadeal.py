import shutil
import subprocess
import scipy.io as io
import os
import platform
import tarfile
from pathlib import Path
from tqdm import tqdm

def get_system() -> str:
    return platform.system().lower()

def linux_decompress() -> None:
    if not Path.exists(IMAGENET_DIR):
        raise FileNotFoundError("The directory `./data/imagenet` does not exist.")

    subprocess.run(
        ['source', 'decompress.sh'],
        shell=True,
        check=True,
    )
    return None

def python_decompress() -> None:
    if not Path.exists(IMAGENET_DIR):
        raise FileNotFoundError("The directory `./data/imagenet` does not exist.")
    
    os.chdir(IMAGENET_DIR)


    devkit_tar = IMAGENET_DIR / 'ILSVRC2012_devkit_t12.tar.gz'
    with tarfile.open(devkit_tar, 'r:gz') as tar:
        tar.extractall(IMAGENET_DIR)

    train_tar = IMAGENET_DIR / 'ILSVRC2012_img_train.tar'
    train_dir = IMAGENET_DIR / 'train'
    Path.mkdir(train_dir, exist_ok=True)


    print('Decompressing the train set...')
    with tarfile.open(train_tar, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, unit='file'):
            tar.extract(member=member, path=train_dir)

    total_entries = len(os.listdir(train_dir))
    print('Dealing with the train set...')
    for entry in tqdm(Path.iterdir(train_dir), total=total_entries):
        if entry.suffix == '.tar':
            sub_tar = train_dir / entry
            sub_dir = train_dir / entry.stem
            Path.mkdir(sub_dir, exist_ok=True)
            with tarfile.open(sub_tar, 'r') as tar:
                tar.extractall(sub_dir)
            os.remove(sub_tar)

    val_tar = IMAGENET_DIR / 'ILSVRC2012_img_val.tar'
    val_dir = IMAGENET_DIR / 'val'
    Path.mkdir(val_dir, exist_ok=True)

    print('Decompressing the val set...')
    with tarfile.open(val_tar, 'r') as tar:
        members = tar.getmembers()
        for member in tqdm(members, unit='file'):
            tar.extract(member=member, path=val_dir)

    return None

def move_valimg(val_dir='val', devkit_dir='ILSVRC2012_devkit_t12') -> None:
    val_dir = IMAGENET_DIR / val_dir
    devkit_dir = IMAGENET_DIR / devkit_dir

    synset = io.loadmat(devkit_dir / 'data' / 'meta.mat')
    ground_truth = open(devkit_dir / 'data' / 'ILSVRC2012_validation_ground_truth.txt')
    lines = ground_truth.readlines()
    labels = [int(line[:-1]) for line in lines]
    root, _, filenames = next(Path.walk(val_dir))
    print('Dealing with the val set...')
    for filename in tqdm(filenames):

        val_id = int(filename.split('.')[0].split('_')[-1])
        ILSVRC_ID = labels[val_id-1]
        WIND = synset['synsets'][ILSVRC_ID-1][0][1][0]

        output_dir = root / WIND
        Path.mkdir(output_dir, exist_ok=True)
        shutil.move(root / filename, output_dir / filename)

    return None

if __name__ == '__main__':
    system = get_system()
    FILE_DIR = Path(__file__).parent.resolve()
    IMAGENET_DIR = FILE_DIR / 'data' / 'imagenet'

    if system == 'linux':
        try:
            linux_decompress()
        except Exception as e:
            print(f'Error: {e}, try to use python script to decompress.')
            python_decompress()
    else:
        python_decompress()
    move_valimg()