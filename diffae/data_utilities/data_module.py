import os
from pathlib import Path
import sys

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader

from diffae.data_utilities.io import load_audio, log_mel_spectrogram

class MusicDataset(Dataset):
    def __init__(self,
                 split: str,
                 dataset_path: str,
                 sample_rate: int = 16_000,):
        # Store split
        assert split in ["train", "val", "test"]
        self.split = split

        path = Path(dataset_path)
        self.data_dir = path.parent.absolute()
        
        # Read data
        with open(dataset_path, "r") as f:
            lines = f.readlines()
        if split == "train":
            self.data = lines[:int(0.9*len(lines))]
        elif split == "val" or split == "test":
            self.data = lines[int(0.9*len(lines)):]
        else:
            raise ValueError()

        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.data)

    def _random_cut(self, audio, max_length: int = 10):
        """ audio: audio waveform
            max_length: maximum length of the cut in seconds
        """
        max_length_sample = self.sample_rate*max_length
        if len(audio) < max_length_sample:
            return None
        start = np.random.randint(0, len(audio) - max_length_sample)
        return audio[start : start + max_length_sample]

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        audio_path = os.path.join(self.data_dir, self.data[idx]).strip()

        try:
            audio = load_audio(audio_path)
            audio = self._random_cut(audio)
            mel = log_mel_spectrogram(audio)
        except Exception as e:
            print("Error processing audio file:", audio_path)
            return self.__getitem__(idx + 1)
        
        # Return if something is not ok
        if torch.isnan(mel).any():
            return self.__getitem__(idx + 1)

        # Spec augment
        return {
            "mel": mel,
        }


class MusicDatamodule(pl.LightningDataModule):
    def __init__(self,
                 dataset_path,
                 batch_size,
                 n_workers=4) -> None:
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.n_workers = n_workers

    def train_dataloader(self):
        return DataLoader(
            MusicDataset(split="train",
                        dataset_path=self.dataset_path,),
                        n_workers=self.n_workers,
                        batch_size=self.batch_size,
                        shuffle=True,)

    def val_dataloader(self):
        return DataLoader(
            MusicDataset(split="val",
                        dataset_path=self.dataset_path,),
                        n_workers=self.n_workers,
                        batch_size=self.batch_size,
                        shuffle=True,)

    def test_dataloader(self):
        return self.val_dataloader()  # TODO(Oleguer): Test dataloader?


if __name__ == "__main__":
    # Test
    dataset = MusicDataset(split="train",
                           dataset_path="/home/mary/code/diffusion_autoencoder/data/data.lst",
                           sample_rate=16_000)
    dataloader = DataLoader(dataset,
                            batch_size=2,
                            shuffle=True,
                            num_workers=4)
    for batch in dataloader:
        print(batch["mel"].shape)
        break