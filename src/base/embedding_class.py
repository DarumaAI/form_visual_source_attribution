from abc import ABC, abstractmethod
from typing import Any, List, Union

import torch
from PIL import Image


class MultimodalEmbeddingModel(ABC):
    """
    General class for multimodal embedding models.
    Subclass this and implement the encode method for your specific model.
    """

    def __init__(self, model_name: str, device: str = "cpu", **kwargs):
        """
        Args:
            model_name (str): Name or path of the embedding model.
            device (str): Device to run the model on ("cpu" or "cuda").
            **kwargs: Additional model-specific parameters.
        """
        self.model_name = model_name
        self.device = device
        self.model = self._load_model(model_name, device, **kwargs)

    @abstractmethod
    def _load_model(self, model_name: str, device: str, **kwargs) -> Any:
        """
        Loads the embedding model.
        """
        pass

    @abstractmethod
    def encode(
        self, inputs: List[Union[str, Image.Image]], modality: str = "text", **kwargs
    ) -> torch.Tensor:
        """
        Encodes the input(s) into embeddings.

        Args:
            inputs: Input data (text, image, audio, etc.).
            modality (str): The modality of the input ("text", "image", etc.).
            **kwargs: Additional encoding parameters.

        Returns:
            Embedding(s) as numpy array, tensor, or list.
        """
        pass
