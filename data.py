"""Dataset/dataloader factories used by the training scripts.

`prepare_train_data`/`prepare_test_data` dispatch on a `dataset` name
substring (e.g. "cifar10", "mnist", "svhn", "cif10"/"cif100" for the
timm-augmented CIFAR variants, "toycar" for MelSpectrogramDataset).
"""

from __future__ import print_function

import torch
import torchaudio
import torchvision
import torchvision.transforms as transforms
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import Dataset, DataLoader
import librosa

import logging
import numpy as np
import glob
import os
import sys

crop_size = 32
padding = 4


def prepare_train_data(dataset='cifar10', batch_size=128,
                       shuffle=True, num_workers=4):
    """Builds the training DataLoader for `dataset` (matched by name
    substring - see module docstring), with that dataset's standard
    augmentation/normalization transform.

    Output: a torch.utils.data.DataLoader, or None if `dataset` matches
    nothing."""

    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'cif100' in dataset:
        transform_train_list = []

        # Random crop + horizontal flip
        transform_train_list.append(transforms.RandomCrop(32, padding=4))
        transform_train_list.append(transforms.RandomHorizontalFlip())

        # Optional color jitter
        color_jitter = 0.4
        transform_train_list.append(transforms.ColorJitter(
            brightness=color_jitter, contrast=color_jitter, 
            saturation=color_jitter, hue=color_jitter
        ))
        
        # Auto augment
        auto_augment = 'rand-m9-mstd0.5-inc1'
        transform_train_list.append(rand_augment_transform(
            config_str=auto_augment,
            hparams={}
        ))
        
        # To tensor and normalize
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform_train_list.append(transforms.ToTensor())
        transform_train_list.append(transforms.Normalize(mean, std))

        # Random erasing
        re_prob = 0.25
        transform_train_list.append(transforms.RandomErasing(
            p=re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'
        ))


        transform_train = transforms.Compose(transform_train_list)
        trainset = torchvision.datasets.__dict__["CIFAR100"](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)        
    elif 'cif10' in dataset:
        transform_train_list = []

        # Random crop + horizontal flip
        transform_train_list.append(transforms.RandomCrop(32, padding=4))
        transform_train_list.append(transforms.RandomHorizontalFlip())

        # Optional color jitter
        color_jitter = 0.4
        transform_train_list.append(transforms.ColorJitter(
            brightness=color_jitter, contrast=color_jitter, 
            saturation=color_jitter, hue=color_jitter
        ))
        
        # Auto augment
        auto_augment = 'rand-m9-mstd0.5-inc1'
        transform_train_list.append(rand_augment_transform(
            config_str=auto_augment,
            hparams={}
        ))
        
        # To tensor and normalize
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform_train_list.append(transforms.ToTensor())
        transform_train_list.append(transforms.Normalize(mean, std))

        # Random erasing
        re_prob = 0.25
        transform_train_list.append(transforms.RandomErasing(
            p=re_prob, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random'
        ))


        transform_train = transforms.Compose(transform_train_list)
        trainset = torchvision.datasets.__dict__["CIFAR10"](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)  
               
    elif 'svhn' in dataset:
        transform_train =transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4377, 0.4438, 0.4728),
                                         (0.1980, 0.2010, 0.1970)),
                ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300,  0.4284, 0.4427),
                                 (0.1963,  0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data',
            split='extra',
            download=True,
            transform = transform_extra
        )

        total_data =  torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
        
    elif 'mnist' in dataset:
        transform_train = transforms.Compose([
            transforms.ToTensor()
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/tmp/data', train=True, download=True, transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers)
    elif 'toycar' in dataset:
        dataset = MelSpectrogramDataset(
            target_dir = "/data/zkokhazad/adaptive_networks/DFS_LRF_DynamicMinRank_MaxPoolDS/Zari_Anomaly_Detection/dev_data/ToyCar/train",
            n_mels=128,
            frames=5,
            n_fft=1024,
            hop_length=512,
            power=2.0
        )
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        

    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar10', batch_size=128,
                      shuffle=False, num_workers=4):
    """Builds the test/eval DataLoader for `dataset` (matched by name
    substring - see module docstring), with that dataset's standard
    (non-augmenting) normalization transform.

    Output: a torch.utils.data.DataLoader, or None if `dataset` matches
    nothing."""

    if 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    elif 'cif100' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        testset = torchvision.datasets.CIFAR100(
            root='/tmp/data', train=False, download=True, transform=transform_test
        )

        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )    
    elif 'cif10' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        testset = torchvision.datasets.CIFAR10(
            root='/tmp/data', train=False, download=True, transform=transform_test
        )

        test_loader = DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )        
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.4524,  0.4525,  0.4690),
                                         (0.2194,  0.2266,  0.2285)),
                ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
                                               root='/tmp/data',
                                               split='test',
                                               download=True,
                                               transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
        
    elif 'mnist' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](root='/tmp/data',
                                               train=False,
                                               download=True,
                                               transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers)
    else:
        test_loader = None
    return test_loader


    
class MelSpectrogramDataset(Dataset):
    """Loads every .wav file in `target_dir`, converts each to a
    log-mel-spectrogram, and slices it into fixed-size overlapping frames
    (each `n_mels * frames` long) as the dataset's samples.
    `__getitem__` returns (sample, dummy label 0) - there are no real
    labels."""
    def __init__(self, target_dir, n_mels=128, frames=5, n_fft=1024, hop_length=512, power=2.0, transform=None):
        self.target_dir = target_dir

        self.n_mels = n_mels
        self.frames = frames
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power

        # Get list of files
        self.file_list = sorted(glob.glob(os.path.join(target_dir, "*.wav")))
        if len(self.file_list) == 0:
            raise RuntimeError(f"No .wav files found in {target_dir}")

        # Precompute all vectors from all files and concatenate
        all_vectors = []
        for file_path in self.file_list:
            vectors = self.file_to_vector_array(file_path)
            if vectors.size > 0:
                all_vectors.append(vectors)
        if len(all_vectors) == 0:
            raise RuntimeError("No valid feature vectors in dataset")
        
        # Vertically stack all vectors: shape (total_samples, dims)
        self.dataset = np.vstack(all_vectors)
        self.dataset = torch.tensor(self.dataset, dtype=torch.float32)


    def file_load(self, wav_name, mono=False):
        """Loads a .wav file. Output: (waveform array, sample rate)."""
        try:
            return librosa.load(wav_name, sr=None, mono=mono)
        except Exception as e:
            print(f"file_broken or not exists!! : {wav_name}, error: {e}")
            raise

    def file_to_vector_array(self, file_name):
        """Converts one .wav file into its overlapping-frame feature
        vectors. Output: a (num_frames, n_mels * frames) numpy array
        (empty if the file is too short for even one frame)."""
        dims = self.n_mels * self.frames
        y, sr = self.file_load(file_name)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=self.power)
        log_mel_spectrogram = 20.0 / self.power * np.log10(mel_spectrogram + np.finfo(float).eps)
        log_mel_spectrogram = log_mel_spectrogram[:, 50:250]
        vector_array_size = log_mel_spectrogram.shape[1] - self.frames + 1
        if vector_array_size < 1:
            return np.empty((0, dims))
        vector_array = np.zeros((vector_array_size, dims))
        for t in range(self.frames):
            vector_array[:, self.n_mels * t: self.n_mels * (t + 1)] = log_mel_spectrogram[:, t:t+vector_array_size].T
        return vector_array

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, idx):
        return self.dataset[idx], torch.tensor(0)  # dummy label
