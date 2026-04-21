"""Embedding registry.

Single source of truth for which embedding heads to run the analysis on.
Each entry points to one npz file + one array key inside it. Downstream
scripts (step2, step4, R/dbrda.R) iterate over this list.

Files live under `embeddings/<model>/<model>_contempart.npz` and are
shipped via the project's Zenodo record.
"""

from dataclasses import dataclass
from pathlib import Path

EMBEDDINGS_DIR = Path("embeddings")


@dataclass(frozen=True)
class Head:
    """One embedding head (a single feature vector per image)."""

    name: str             # short id used in filenames / CSVs
    display: str          # human-readable name for README tables
    file: str             # path relative to EMBEDDINGS_DIR
    key: str              # array key inside the npz
    dim: int
    role: str             # content | style | appearance | general | duplicate
    backbone: str         # source paper / model family
    normalized: bool      # True if L2-normalized at save time

    @property
    def path(self) -> Path:
        return EMBEDDINGS_DIR / self.file


HEADS: list[Head] = [
    Head("clip_l", "CLIP-L", "clip_openai/clip_openai_contempart.npz",
         "embeddings", 768, "content", "CLIP ViT-L/14 (OpenAI)", True),
    Head("clip_b32", "CLIP-B32", "clip_openai_b32/clip_openai_b32_contempart.npz",
         "embeddings", 512, "content", "CLIP ViT-B/32 (OpenAI)", True),
    Head("openclip_l", "OpenCLIP-L", "clip_openclip/clip_openclip_contempart.npz",
         "embeddings", 768, "content", "OpenCLIP ViT-L/14 (LAION-2B)", True),
    Head("dinov2_cls", "DINOv2-CLS", "dinov2/dinov2_contempart.npz",
         "cls_embeddings", 1024, "general", "DINOv2 ViT-L/14", True),
    Head("dinov2_l12", "DINOv2-L12", "dinov2/dinov2_contempart.npz",
         "layer12_embeddings", 1024, "general", "DINOv2 ViT-L/14", True),
    Head("dinov2_style", "DINOv2-Style", "dinov2/dinov2_contempart.npz",
         "embeddings", 1024, "style", "DINOv2 ViT-L/14", True),
    Head("dinov2_gram", "DINOv2-Gram", "dinov2/dinov2_contempart.npz",
         "gram_embeddings", 1024, "style", "DINOv2 ViT-L/14", True),
    Head("csd_content", "CSD-Content", "csd/csd_contempart.npz",
         "content_embeddings", 768, "content", "CSD ViT-L/14", True),
    Head("csd_style", "CSD-Style", "csd/csd_contempart.npz",
         "embeddings", 768, "style", "CSD ViT-L/14", True),
    Head("scflow_content", "SCFlow-Content", "scflow/scflow_contempart.npz",
         "content_embeddings", 768, "content", "SCFlow (ICCV 2025)", False),
    Head("scflow_style", "SCFlow-Style", "scflow/scflow_contempart.npz",
         "embeddings", 768, "style", "SCFlow (ICCV 2025)", False),
    Head("goya_content", "GOYA-Content", "goya/goya_contempart.npz",
         "content_embeddings", 2048, "content", "GOYA MLP (on CLIP-B32)", False),
    Head("goya_style", "GOYA-Style", "goya/goya_contempart.npz",
         "style_embeddings", 2048, "style", "GOYA MLP (on CLIP-B32)", False),
    Head("kim_c", "KIM-C", "sd2_kim/sd2_kim_contempart.npz",
         "embeddings", 1024, "content", "OpenCLIP ViT-H/14 (LAION-2B)", True),
    Head("kim_a", "KIM-A", "sd2_kim/sd2_kim_contempart.npz",
         "a_vectors", 16384, "appearance", "SD 2.1 VAE", False),
    Head("sscd", "SSCD", "sscd/sscd_contempart.npz",
         "embeddings", 512, "duplicate", "SSCD ResNet50", True),
]


HEADS_BY_NAME: dict[str, Head] = {h.name: h for h in HEADS}


def get(name: str) -> Head:
    return HEADS_BY_NAME[name]
