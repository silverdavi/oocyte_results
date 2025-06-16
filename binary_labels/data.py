
# Standard
import csv
from enum import IntEnum
import json
import os
import tarfile
from typing import Tuple
from urllib.request import urlretrieve
# External
import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchvision.transforms as transforms

A_LO = 0.2
A_HI = 0.1
H_LO = 90.0
H_HI = 130.0
C = 1

def double_sigmoid_score(h: float):
    return (np.tanh(A_LO * (h - H_LO)) - np.tanh(A_HI * (h - H_HI))) * C

H_MX = minimize_scalar(
    lambda x: -double_sigmoid_score(x),
    bracket = (H_LO, H_HI),
    method = 'Brent',
).x
C = 1.0 / double_sigmoid_score(H_MX)

DATA_DIR = 'data'
DATASET_URL = 'https://zenodo.org/record/7912264/files'

DIR_FRAMES = 'frames'
DIR_ANNOTATIONS = 'annotations'
DIR_TIMES = 'times'

DATASET_FILES = {
    'embryo_dataset.tar.gz': DIR_FRAMES,
    'embryo_dataset_annotations.tar.gz': DIR_ANNOTATIONS,
    'embryo_dataset_time_elapsed.tar.gz': DIR_TIMES,
    'embryo_dataset_grades.csv': 'grades.csv',
}

def download_dataset():
    for filename, target in DATASET_FILES.items():
        target_path = f'{DATA_DIR}/{target}'
        if os.path.exists(target_path):
            continue
        source_url = f'{DATASET_URL}/{filename}'
        download_path = f'{DATA_DIR}/{filename}'
        print(f'Downloading {filename}...')
        urlretrieve(source_url, download_path)
        if filename.endswith('.tar.gz'):
            print(f'Extracting {filename}...')
            with tarfile.open(download_path, "r:gz") as tar:
                tar.extractall(path = DATA_DIR)
            os.rename(download_path[:-7], target_path)
            os.remove(download_path)
        else:
            os.rename(download_path, target_path)

def read_phases(embryo_id):
    phase2idx = {}
    with open(f'{DATA_DIR}/{DIR_ANNOTATIONS}/{embryo_id}_phases.csv') as fr:
        cr = csv.reader(fr)
        for row in cr:
            phase2idx[row[0]] = (int(row[1]), int(row[2]))
    return phase2idx

def read_times(embryo_id):
    idx2time = {}
    with open(f'{DATA_DIR}/{DIR_TIMES}/{embryo_id}_timeElapsed.csv') as fr:
        cr = csv.reader(fr)
        next(cr)
        for row in cr:
            if len(row) == 2:
                idx2time[int(row[0])] = round(float(row[1]), 1)
    return idx2time

def get_frame(embryo_id):
    min_run = None
    min_frame = None
    for frame in os.listdir(f'{DATA_DIR}/{DIR_FRAMES}/{embryo_id}/'):
        if not frame.endswith('.jpeg'):
            continue
        run = int(frame.split('_RUN')[-1].split('.')[0])
        if min_run is None or run < min_run:
            min_run = run
            min_frame = frame
    return min_frame

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.485, 0.456, 0.406),  # ImageNet mean
        std=(0.229, 0.224, 0.225)    # ImageNet std
    )
])

class LabelType(IntEnum):
    Binary = 0
    DoubleSigmoid = 1

class EmbryoAnnotations(Dataset):
    
    def __init__(self, label_type: LabelType = LabelType.DoubleSigmoid):
        super(EmbryoAnnotations, self).__init__()
        download_dataset()
        ids = [ f.name for f in os.scandir(f'{DATA_DIR}/{DIR_FRAMES}/') if f.is_dir()]
        files = list(map(get_frame, ids))
        anots = list(map(read_phases, ids))
        times = list(map(read_times, ids))
        print(f'Found {len(ids)} samples.')
        good = [ i for i in range(len(ids)) if 'tB' not in anots[i] or anots[i]['tB'][0] in times[i] ]
        self.ids = [ ids[i] for i in good ]
        self.files = [ files[i] for i in good ]
        self.annotations = [ anots[i] for i in good ]
        self.times = [ times[i] for i in good ]
        self.tbs = [ a.get('tB', -1) for a in self.annotations ]
        print(f'Filtered to {len(self.ids)} samples with valid annotations.')
        if label_type == LabelType.Binary:
            self.labels = [ 1 if 'tB' in anots else 0 for anots in self.annotations ]
        elif label_type == LabelType.DoubleSigmoid:
            self.labels = [ double_sigmoid_score(times[anots['tB'][0]]) if 'tB' in anots else 0.0
                           for anots, times in zip(self.annotations, self.times) ]
        else:
            raise NotImplementedError("Unsupported label type. Use LabelType.DoubleSigmoid or LabelType.Binary.")
        with open(f'logs/labels.json', 'w') as f:
            json.dump(self.labels, f, indent = 4, default = str)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, int]:
        id = self.ids[index]
        filename = self.files[index]
        image = Image.open(f'{DATA_DIR}/{DIR_FRAMES}/{id}/{filename}').convert('RGB')  # Ensure 3-channel images
        image: Tensor = transform(image)
        label = torch.tensor(self.labels[index], dtype = torch.float32)
        return image, label, index
    
    def __len__(self) -> int:
        return len(self.files)

def get_frames(id: str):
    frames = []
    for frame in os.listdir(f'{DATA_DIR}/{DIR_FRAMES}/{id}/'):
        if not frame.endswith('.jpeg'):
            continue
        run = int(frame.split('_RUN')[-1].split('.')[0])
        frames.append((run, frame))
    return sorted(frames)

if __name__ == '__main__':
    ids = [ f.name for f in os.scandir(f'{DATA_DIR}/{DIR_FRAMES}/') if f.is_dir() ]
    id2frames = { id: get_frames(id) for id in ids }
    id2times = { id: read_times(id) for id in ids }
    id2phases = { id: read_phases(id) for id in ids }
    countp1 = 0
    countall = 0
    countclose = 0
    anames = set()
    for id in ids:
        for aname in id2phases[id]:
            anames.add(aname)
        last_index = id2frames[id][-1][0]
        last_phaseidx = max(id2phases[id].values(), key = lambda rng: rng[1])[1]
        last_timeidx = max(id2times[id].keys())
        print(id, last_index, last_phaseidx, last_timeidx)
        countall += 1
        if last_index + 1 == last_phaseidx == last_timeidx:
            countp1 += 1
        if abs(last_index - last_phaseidx) <= 5 and abs(last_index - last_timeidx) <= 5:
            countclose += 1
    print(countall, countp1, countclose)
    print(anames)
    with open('agg.csv', 'w') as f:
        cw = csv.writer(f)
        cw.writerow([
            'id',
            'first_run',
            'last_run',
            'num_runs',
            'first_time_idx',
            'first_time_hpi',
            'last_time_idx',
            'last_time_hpi',
            'first_phase_idx',
            'last_phase_idx',
            *[ f'{n}_first' for n in anames ],
            *[ f'{n}_last' for n in anames ],
        ])
        for id in ids:
            row = [
                id,
                id2frames[id][0][0],
                id2frames[id][-1][0],
                len(id2frames[id]),
                min(id2times[id].keys()),
                id2times[id][min(id2times[id].keys())],
                max(id2times[id].keys()),
                id2times[id][max(id2times[id].keys())],
                min(id2phases[id].values(), key = lambda rng: rng[0])[0],
                max(id2phases[id].values(), key = lambda rng: rng[1])[1],
                *[id2phases[id].get(aname, (None, None))[0] for aname in anames],
                *[id2phases[id].get(aname, (None, None))[1] for aname in anames],
            ]
            cw.writerow(row)
