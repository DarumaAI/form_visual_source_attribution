from ..base.embedding_class import MultimodalEmbeddingModel
from .colnomic import Colnomic

MODEL_MAP = {
    "nomic-ai/colnomic-embed-multimodal-3b": Colnomic,
}


def automodel(
    model_name: str, device: str = "cpu", **kwargs
) -> MultimodalEmbeddingModel:
    model_class = MODEL_MAP.get(model_name)
    if not model_class:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_class(model_name=model_name, device=device, **kwargs)
