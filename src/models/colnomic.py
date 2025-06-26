from typing import List, Union

import torch
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
from PIL import Image
from transformers import BatchFeature
from transformers.utils.import_utils import is_flash_attn_2_available

from ..base.embedding_class import MultimodalEmbeddingModel


class Conomic(MultimodalEmbeddingModel):
    """
    General class for multimodal embedding models.
    Subclass this and implement the encode method for your specific model.
    """

    def _load_model(self, model_name: str, device: str, **kwargs) -> None:
        """
        Loads the embedding model.
        """
        if model_name not in [
            "nomic-ai/colnomic-embed-multimodal-3b",
            "nomic-ai/colnomic-embed-multimodal-7b",
        ]:
            raise ValueError(f"Model {model_name} is not supported. ")
        self.device = device
        self.model = ColQwen2_5.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # or "mps" if on Apple Silicon
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else None
            ),
        ).eval()
        self.processor = ColQwen2_5_Processor.from_pretrained(model_name)

    def encode(
        self, inputs: List[Union[str, Image.Image]], modality: str = "text", **kwargs
    ) -> torch.Tensor:
        """
        Encodes the input(s) into embeddings.

        Args:
            inputs: Input data (text, image, audio, etc.).
            modality (str): The modality of the input ("text", "image").
            **kwargs: Additional encoding parameters.

        Returns:
            Embedding(s) as numpy array, tensor, or list.
        """
        batch_size = kwargs.get("batch_size", 1)
        if modality == "text":
            processed_inputs = self.processor.process_queries(inputs).to(
                self.model.device
            )
        elif modality == "image":
            processed_inputs = self.processor.process_images(inputs).to(
                self.model.device
            )
        else:
            raise ValueError(
                f"Modality {modality} is not supported. Use 'text' or 'image'."
            )

        all_embeddings = torch.empty(
            (0, self.model.config.hidden_size), device=self.model.device
        )
        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch = processed_inputs[i : i + batch_size]
                embeddings = self.model(**batch)
                all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)

        return all_embeddings

    def compute_score(self, query_embeddings, image_embeddings):
        """
        Computes the similarity score between query and image embeddings.

        Args:
            query_embeddings: Embeddings for the query.
            image_embeddings: Embeddings for the images.

        Returns:
            Similarity scores as a tensor.
        """
        return self.processor.score_multi_vector(query_embeddings, image_embeddings)
