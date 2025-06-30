from ..base.embedding_class import MultimodalEmbeddingModel
from .colnomic import Colnomic
from .colpali import Colpali
from .colsmall import Colsmall

MODEL_MAP = {
    "nomic-ai/colnomic-embed-multimodal-3b": Colnomic,
    "nomic-ai/colnomic-embed-multimodal-7b": Colnomic,
    "vidore/colqwen2.5-v0.2": Colnomic,
    "vidore/colpali-v1.3": Colpali,
    "vidore/colSmol-256M": Colsmall,
    "vidore/colSmol-500M": Colsmall,
}


def automodel(
    model_name: str, device: str = "cpu", **kwargs
) -> MultimodalEmbeddingModel:
    model_class = MODEL_MAP.get(model_name)
    if not model_class:
        raise ValueError(f"Model '{model_name}' is not supported.")
    return model_class(model_name=model_name, device=device, **kwargs)
