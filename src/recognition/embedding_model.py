"""Generate FaceNet embeddings for fur-seal faces."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import cv2
import numpy as np
from sklearn.decomposition import PCA

from src.recognition.face_extractor import FacePatch

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class FaceEmbedding:
    """Embedding vector metadata container."""

    face: FacePatch
    embedding: torch.Tensor


class FacePatchDataset(Dataset[torch.Tensor]):
    """PyTorch dataset for already cropped face patches."""

    def __init__(self, patches: Sequence[FacePatch], transform: transforms.Compose) -> None:
        self.patches = list(patches)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, FacePatch]:
        patch = self.patches[index]
        image = Image.open(patch.patch_path).convert("RGB")
        return self.transform(image), patch


class FaceEmbedder:
    """Compute FaceNet embeddings for a batch of face patches."""

    def __init__(
        self,
        backbone: str = "facenet-inception",
        device: str = "cuda",
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        self.backbone = backbone
        self.batch_size = batch_size
        self.num_workers = num_workers

        if self.backbone == "facenet-inception":
            self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
            self.model = InceptionResnetV1(pretrained="vggface2", classify=False)
            self.model.eval()
            self.model.to(self.device)
            self.transform = transforms.Compose(
                [
                    transforms.Resize((160, 160)),
                    transforms.ToTensor(),
                    fixed_image_standardization,
                ]
            )
        elif self.backbone == "color-hist":
            self.device = torch.device("cpu")
            self.model = None
            self.transform = None
        elif self.backbone == "hog-pca":
            # CPU-only classical feature pipeline
            self.device = torch.device("cpu")
            self.model = None
            self.transform = None
            # Configure an OpenCV HOG descriptor for square faces
            win_size = (128, 128)
            block_size = (16, 16)
            block_stride = (8, 8)
            cell_size = (8, 8)
            nbins = 9
            self.hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
            self.hog_win_size = win_size
        else:
            raise ValueError(f"Unsupported recognition backbone: {self.backbone}")

    @torch.inference_mode()
    def encode(self, patches: Sequence[FacePatch]) -> List[FaceEmbedding]:
        if not patches:
            return []

        if self.backbone == "color-hist":
            return [FaceEmbedding(face=patch, embedding=self._encode_histogram(patch)) for patch in patches]
        if self.backbone == "hog-pca":
            return self._encode_hog_pca(patches)

        dataset = FacePatchDataset(patches, self.transform)

        def _collate(batch: Iterable[tuple[torch.Tensor, FacePatch]]) -> tuple[torch.Tensor, List[FacePatch]]:
            images, meta = zip(*batch)
            return torch.stack(list(images), dim=0), list(meta)

        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.device.type == "cuda",
            shuffle=False,
            collate_fn=_collate,
        )

        embeddings: List[FaceEmbedding] = []
        for batch, batch_meta in loader:
            batch = batch.to(self.device, non_blocking=True)
            features = self.model(batch)
            for vector, meta in zip(features.cpu(), batch_meta):
                embeddings.append(FaceEmbedding(face=meta, embedding=vector.detach().clone()))
        return embeddings

    def _encode_histogram(self, patch: FacePatch) -> torch.Tensor:
        image = cv2.imread(str(patch.patch_path))
        if image is None:
            raise FileNotFoundError(f"Face patch missing: {patch.patch_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bins = 32
        hist_parts = []
        for channel in range(3):
            channel_hist = cv2.calcHist([image], [channel], None, [bins], [0, 256])
            hist_parts.append(channel_hist.squeeze())
        hist = np.concatenate(hist_parts).astype("float32")
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist /= norm
        return torch.from_numpy(hist)

    def _encode_hog_pca(self, patches: Sequence[FacePatch]) -> List[FaceEmbedding]:
        # Compute HOG for all patches first
        feats: List[np.ndarray] = []
        valid_meta: List[FacePatch] = []
        for patch in patches:
            vec = self._compute_hog(str(patch.patch_path))
            if vec is None:
                continue
            feats.append(vec)
            valid_meta.append(patch)
        if not feats:
            return []

        X = np.stack(feats).astype("float32")
        # PCA to min(n_samples, 64) dims for compactness (no whitening to avoid collapsing distances)
        n_components = int(min(64, X.shape[0], X.shape[1]))
        if n_components >= 2:
            pca = PCA(n_components=n_components, whiten=False, random_state=1337)
            X_proj = pca.fit_transform(X).astype("float32")
        else:
            X_proj = X
        # L2 normalize after PCA for cosine-friendly space
        X_proj = X_proj / (np.linalg.norm(X_proj, axis=1, keepdims=True) + 1e-8)

        embeddings: List[FaceEmbedding] = []
        for vec, meta in zip(X_proj, valid_meta):
            embeddings.append(FaceEmbedding(face=meta, embedding=torch.from_numpy(vec)))
        return embeddings

    def _compute_hog(self, img_path: str) -> np.ndarray | None:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        resized = cv2.resize(img, self.hog_win_size, interpolation=cv2.INTER_LINEAR)
        # OpenCV HOG expects specific window size
        desc = self.hog.compute(resized)
        if desc is None:
            return None
        return desc.reshape(-1)
