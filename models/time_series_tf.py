import torch
import torch.nn as nn
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerForPrediction

class TimeSeriesTransformer(nn.Module):
    def __init__(self, config):
        super(TimeSeriesTransformer, self).__init__()
        self.config = config

        # Define the configuration for the transformer model
        transformer_config = TimeSeriesTransformerConfig(
            prediction_length=config['prediction_length'],
            context_length=config['context_length'],
            num_time_features=config['num_time_features'],
            encoder_layers=config['encoder_layers'],
            decoder_layers=config['decoder_layers'],
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            input_size=config['input_size'],
            lags_sequence=config['lags_sequence'],
            num_parallel_samples=config['num_parallel_samples'],
            distribution_output=config['distribution_output']
        )

        # Initialize the TimeSeriesTransformer model
        self.model = TimeSeriesTransformerForPrediction(transformer_config)

    def forward(self, past_values, past_time_features, past_observed_mask, future_values, future_time_features, future_observed_mask, return_dict):
        """
        Perform the forward pass through the transformer model.

        Parameters:
        - past_values: Tensor containing past time series data.
        - past_time_features: Tensor of time features associated with the past.
        - past_observed_mask: Binary mask indicating observed data points in the past.
        - future_values: Tensor containnig future time series data.
        - future_time_features: Tensor of time features associated with the future.
        - future_observed_mask: Binary mask indcatinig observed data points in the future. 

        Returns:
        - Predicted future values.
        """
        output = self.model(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            future_values=future_values,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
            return_dict=return_dict
            )

        return output
    
