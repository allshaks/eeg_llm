import torch
from torch.utils.data import Dataset
import mne
import numpy as np

class TimeSeriesDataset(Dataset):
    def __init__(self, config):
        """
        A custom dataset class to handle both toy data and EEG data.
        
        Parameters:
        - config (dict): Configuration dictionary containing parameters for datasets.
            Keys:
            - data_source (str): "toy" or "eeg" to specify the data source.
            - eeg_file (str): Path to the EEG file (.set) if data_source is "eeg".
            - tmin, tmax (float): Time window for EEG epochs.
            - preload (bool): Whether to preload the EEG data.
            - context_length (int): Context window for time series.
            - prediction_length (int): Prediction window for time series.
            - seq_len (int): Sequence length for toy data.
            - num_obsv (int): Number of samples for toy data.
            - noise (int): Level of noise applied to the toy data. 
            - mean1 (float): Mean of the first pulse.
            - mean2 (float): Mean of the second pulse.
            - std1 (float): Standard deviation of the fist pulse.
            - std2 (float): Standard deviation of the second pulse. 
        """
        
        self.config = config 

        if config.get('data_source') == "toy":
            self.data = self.generate_toy_data(
                num_obsv=config.get('num_obsv'), 
                seq_len=config.get('seq_len'), 
                noise=config.get('noise'), 
                mean1=config.get('mean1'),
                mean2=config.get('mean2'), 
                std1=config.get('std1'), 
                std2=config.get('std2')
                )
            
        elif config.get('data_source') == "eeg":

            if config.get('eeg_file') is None:
                raise ValueError("eeg_file path must be provided for EEG data")
            self.data = self.load_eeg_data(config.get('eeg_file'), config.get('tmin'), config.get('tmax'), config.get('preload'))

        else:
            raise ValueError("data_source must be 'toy' or 'eeg'")


    def generate_toy_data(self, num_obsv, seq_len, noise, mean1, mean2, std1, std2):
        """
        Generates synthetic toy data.
        """
        def gaussian(t, mean, std):
            return 1/np.sqrt(2*np.pi*std**2)*np.exp(-(t-mean)**2/(2*std**2))

        data = []
        for _ in range(num_obsv):
            phase_shift = np.random.randn(1)
            t = np.linspace(0, 5, seq_len) + phase_shift

            pulse1 = gaussian(t, mean1, std1) + noise * np.random.randn(seq_len)
            pulse2 = gaussian(t, mean2, std2) + noise * np.random.randn(seq_len)

            sample = np.stack([pulse1, pulse2], axis=1)
            data.append(sample)
        return np.array(data)


    def load_eeg_data(self, eeg_file, tmin, tmax, preload):
        """
        Loads and processes EEG data using MNE.
        """
        raw = mne.io.read_raw_eeglab(eeg_file, preload=preload)
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(raw, events, event_id, tmin, tmax, baseline=None, preload=preload)
        
        # Get the epoched data
        data_original = epochs.get_data()
        data = np.transpose(data_original, (0, 2, 1))

        return data


    def __len__(self):
        """
        Returns the length of the data set
        """

        return len(self.data)


    def __getitem__(self, idx):
        """
        Returns a sample of the data set 
        """

        return np.array(self.data[idx], dtype=np.float32)
    

    def get_average_data(self):
        """
        Returns the average data for both toy data and EEG data.
        """
        
        if self.config.get('data_source') == "toy":
            # Calculate the average across all samples for toy data
            avg_data = np.mean(self.data, axis=0)

        elif self.config.get('data_source') == "eeg":
            # For EEG data, use MNE's averaging function
            raw = mne.io.read_raw_eeglab(self.config.get('eeg_file'), preload=self.config.get('preload'))
            events, event_id = mne.events_from_annotations(raw)
            epochs = mne.Epochs(raw, events, event_id, self.config.get('tmin'), self.config.get('tmax'), baseline=None, preload=self.config.get('preload'))
            avg_data_original = epochs.average().data
            avg_data = np.transpose(avg_data_original, (1, 0))
        return avg_data
    

    def get_past_and_future(self, idx):
        """
        Splits the data at the given index into past and future values.
        
        Parameters:
        - idx (int): Index of the data sample.

        Returns:
        - past_values (torch.Tensor): The past segment of the data.
        - future_values (torch.Tensor): The future segment of the data.
        """
        past_values = self.data[idx, :-self.prediction_length, :]
        future_values = self.data[idx, -self.prediction_length:, :]
        return torch.tensor(past_values, dtype=torch.float32), torch.tensor(future_values, dtype=torch.float32)s
