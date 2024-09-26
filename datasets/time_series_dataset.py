import torch
from torch.utils.data import Dataset
import mne
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, data_source="toy", eeg_file=None, tmin=-0.05, tmax=0.22, event_id=None, preload=False, context_length=29, prediction_length=70, seq_len=100, num_samples=6000):
        """
        A custom dataset class to handle both toy data and EEG data.
        
        Parameters:
        - data_source (str): "toy" or "eeg" to specify the data source.
        - eeg_file (str): Path to the EEG file (.set) if data_source is "eeg".
        - tmin, tmax (float): Time window for EEG epochs.
        - event_id (dict): Event ID mapping for EEG data.
        - preload (bool): Whether to preload the EEG data.
        - context_length (int): Context window for time series.
        - prediction_length (int): Prediction window for time series.
        - seq_len (int): Sequence length for toy data.
        - num_samples (int): Number of samples for toy data.
        """
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.seq_len = seq_len
        
        if data_source == "toy":
            self.data = self.generate_toy_data(num_samples=num_samples)
        elif data_source == "eeg":
            if eeg_file is None:
                raise ValueError("eeg_file path must be provided for EEG data")
            self.data = self.load_eeg_data(eeg_file, tmin, tmax, event_id, preload)
        else:
            raise ValueError("data_source must be 'toy' or 'eeg'")

    def generate_toy_data(self, num_samples=6000):
        """
        Generates synthetic toy data.
        """
        def gaussian(t, mean, std):
            return 1/np.sqrt(2*np.pi*std**2)*np.exp(-(t-mean)**2/(2*std**2))

        data = []
        for _ in range(num_samples):
            phase_shift = np.random.randn(1)
            t = np.linspace(0, 5, self.seq_len) + phase_shift

            pulse1 = gaussian(t, 1, 0.3) + np.random.randn(self.seq_len)
            pulse2 = gaussian(t, 2, 0.3) + np.random.randn(self.seq_len)

            sample = np.stack([pulse1, pulse2], axis=1)
            data.append(sample)
        return np.array(data)

    def load_eeg_data(self, eeg_file, tmin, tmax, event_id, preload):
        """
        Loads and processes EEG data using MNE.
        """
        raw = mne.io.read_raw_eeglab(eeg_file, preload=preload)
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=preload)
        
        # Get the epoched data
        data = epochs.get_data()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        past_values = self.data[idx, :-self.prediction_length, :]
        future_values = self.data[idx, -self.prediction_length:, :]
        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(future_values, dtype=torch.float32)
