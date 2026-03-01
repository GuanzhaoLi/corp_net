"""
Route B: Qwen2-VL-2B vision encoder (frozen) + temporal mean pool over 5 images + regression head.
Requires: pip install transformers>=4.45 accelerate
"""
import torch
import torch.nn as nn
import numpy as np

# Optional: only import when using Qwen2-VL
def _load_qwen2vl_components(device, dtype=torch.float32):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        torch_dtype=dtype,
        device_map=None,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = model.to(device)
    return model, processor


class Qwen2VLYieldModel(nn.Module):
    """
    Freeze Qwen2-VL, use vision encoder output -> mean over 5 images -> regression head -> yield.
    Input: batch of (B, 5, 3, H, W) images in [0, 1], float32.
    """
    def __init__(self, num_frames=5, hidden_size=None, dropout=0.1, device=None, dtype=torch.float32):
        super().__init__()
        self.num_frames = num_frames
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        model, processor = _load_qwen2vl_components(self.device, dtype)
        self.vlm = model
        self.processor = processor
        for p in self.vlm.parameters():
            p.requires_grad = False
        # Vision encoder output dim (merger output = text hidden_size)
        self.vision_hidden_size = getattr(
            self.vlm.config, "hidden_size",
            getattr(self.vlm.config.text_config, "hidden_size", 2048)
        )
        if hidden_size is None:
            hidden_size = self.vision_hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.head = self.head.to(self.device, dtype=torch.float32)

    def _images_to_processor_input(self, images):
        """images (B*5, 3, H, W) float [0,1] -> list of numpy (H,W,3) [0,255] for processor."""
        B5, C, H, W = images.shape
        out = []
        for i in range(B5):
            img = images[i].cpu().float()
            if img.max() <= 1.0:
                img = img * 255.0
            img = img.permute(1, 2, 0).numpy().astype(np.uint8)
            out.append(img)
        return out

    def _get_vision_features(self, pixel_values, image_grid_thw):
        """Run vision encoder and return one vector per image (mean pool over patches)."""
        with torch.no_grad():
            vision_outputs = self.vlm.model.visual(
                pixel_values.to(self.device, dtype=self.dtype),
                image_grid_thw.to(self.device),
            )
        # vision_outputs: (total_tokens, hidden_size)
        # Split by image using image_grid_thw. Merger does spatial_merge_size=2 -> 4 patches -> 1 token.
        grid = image_grid_thw.cpu()
        n_images = grid.shape[0]
        tokens_per_image = (grid[:, 0] * (grid[:, 1] // 2) * (grid[:, 2] // 2)).tolist()
        if sum(tokens_per_image) != vision_outputs.shape[0]:
            tokens_per_image = (grid[:, 0] * grid[:, 1] * grid[:, 2]).tolist()
            # try no merge div
            if sum(tokens_per_image) != vision_outputs.shape[0]:
                tokens_per_image = [vision_outputs.shape[0] // n_images] * n_images
        start = 0
        feats = []
        for L in tokens_per_image:
            end = start + L
            feats.append(vision_outputs[start:end].mean(dim=0))
            start = end
        return torch.stack(feats, dim=0)

    def forward(self, images, return_features=False):
        """
        images: (B, 5, 3, H, W) float [0, 1]
        """
        B, T, C, H, W = images.shape
        assert T == self.num_frames
        images_flat = images.reshape(B * T, C, H, W)
        list_imgs = self._images_to_processor_input(images_flat)
        # Processor: list of images -> pixel_values, image_grid_thw
        from transformers.image_utils import to_numpy_array
        processed = self.processor.image_processor.preprocess(
            list_imgs,
            return_tensors="pt",
            do_rescale=True,
            do_normalize=True,
        )
        pixel_values = processed["pixel_values"].to(self.device, dtype=self.dtype)
        image_grid_thw = processed["image_grid_thw"].to(self.device)
        # (B*5, hidden_size)
        feats = self._get_vision_features(pixel_values, image_grid_thw)
        # (B, 5, hidden_size) -> (B, hidden_size)
        feats = feats.reshape(B, T, -1).mean(dim=1)
        out = self.head(feats.float()).squeeze(-1)
        if return_features:
            return out, feats
        return out


def build_vlm_yield_model(num_frames=5, dropout=0.1, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return Qwen2VLYieldModel(num_frames=num_frames, dropout=dropout, device=device)
