import torch as th
from gymnasium import spaces
from torch import nn
from typing import Dict, List, Optional

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Dict,
        drop_keys: Optional[List[str]] = None,
        cnn_output_dim: int = 256,
        custom_extractors: Optional[Dict[str, nn.Module]] = None,
        custom_output_dims: Optional[Dict[str, int]] = None,
        normalized_image: bool = False,
        share_cnn: bool = False,
    ) -> None:
        super().__init__(observation_space, features_dim=1)

        if not isinstance(observation_space, spaces.Dict):
            raise ValueError(
                "CustomCombinedExtractor only works with Dict observation spaces."
            )

        self.dropped_keys = set(drop_keys or [])
        obs_keys = set(observation_space.spaces.keys())

        invalid_drops = self.dropped_keys - obs_keys
        if invalid_drops:
            raise ValueError(f"drop_keys contains invalid keys: {invalid_drops}")

        custom_extractors = custom_extractors or {}
        custom_output_dims = custom_output_dims or {}

        extractors: Dict[str, nn.Module] = {}
        total_dim = 0
        shared_cnn = None

        for key, subspace in observation_space.spaces.items():
            if key in self.dropped_keys:
                continue

            if key in custom_extractors:
                extractors[key] = custom_extractors[key]
                if hasattr(custom_extractors[key], "features_dim"):
                    dim = custom_extractors[key].features_dim
                elif key in custom_output_dims:
                    dim = custom_output_dims[key]
                else:
                    raise ValueError(
                        f"Missing dimension info for custom extractor of key '{key}'"
                    )

            elif is_image_space(subspace, normalized_image=normalized_image):
                dim = custom_output_dims.get(key, cnn_output_dim)
                if share_cnn and shared_cnn is not None:
                    extractors[key] = shared_cnn
                else:
                    cnn = NatureCNN(
                        subspace, features_dim=dim, normalized_image=normalized_image
                    )
                    extractors[key] = cnn
                    if share_cnn and shared_cnn is None:
                        shared_cnn = cnn
            else:
                extractors[key] = nn.Flatten()
                dim = get_flattened_obs_dim(subspace)

            total_dim += dim

        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_dim

    def forward(self, observations: TensorDict) -> th.Tensor:
        encoded_tensor_list = [
            extractor(observations[key]) for key, extractor in self.extractors.items()
        ]
        return th.cat(encoded_tensor_list, dim=1)
