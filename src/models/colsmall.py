from typing import List, Union

import torch
from colpali_engine.models import ColIdefics3, ColIdefics3Processor
from PIL import Image
from transformers import BatchFeature
from transformers.utils.import_utils import is_flash_attn_2_available

from ..base.embedding_class import MultimodalEmbeddingModel


class Colnomic(MultimodalEmbeddingModel):
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
            "vidore/colqwen2.5-v0.2",
        ]:
            raise ValueError(f"Model {model_name} is not supported. ")
        self.device = device
        self.model = ColIdefics3.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map=device,  # or "mps" if on Apple Silicon
            attn_implementation=(
                "flash_attention_2" if is_flash_attn_2_available() else None
            ),
        ).eval()
        self.processor = ColIdefics3Processor.from_pretrained(model_name)

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

        all_embeddings = torch.empty(
            (0, self.model.config.hidden_size), device=self.model.device
        )

        with torch.no_grad():
            for i in range(0, len(inputs), batch_size):
                batch_inputs = inputs[i : i + batch_size]
                if modality == "text":
                    processed_inputs = self.processor.process_queries(batch_inputs).to(
                        self.model.device
                    )
                elif modality == "image":
                    processed_inputs = self.processor.process_images(batch_inputs).to(
                        self.model.device
                    )
                else:
                    raise ValueError(
                        f"Modality {modality} is not supported. Use 'text' or 'image'."
                    )
                embeddings = self.model(**processed_inputs)

                all_embeddings = torch.cat([all_embeddings, embeddings], dim=0)

        return all_embeddings

    def compute_score(self, query_embeddings, image_embeddings) -> torch.Tensor:
        """
        Computes the similarity score between query and image embeddings.

        Args:
            query_embeddings: Embeddings for the query.
            image_embeddings: Embeddings for the images.

        Returns:
            Similarity scores as a tensor.
        """
        return self.processor.score_multi_vector(
            list(torch.unbind(query_embeddings)), list(torch.unbind(image_embeddings))
        )
