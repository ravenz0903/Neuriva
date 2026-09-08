"""
model_service.py — Pretrained U-Net Inference Engine with Safe Deserialization
=============================================================================
Manages HuggingFace model caching, safe weights loading (weights_only=True),
architecture compatibility verification, and seamless fallback to vision_engine.
"""

import os
import logging
import numpy as np

logger = logging.getLogger("cerebroaspects.model")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — model inference disabled, using algorithmic fallback")

HUGGINGFACE_REPO = "herutriana44/acute-ischemic-stroke-unet"
MODEL_FILENAME = "best_unet.pt"
FINETUNED_FILENAME = "best_finetuned_unet.pt"
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_INPUT_SIZE = (256, 256)
THRESHOLD = 0.5

_model = None
_device = None
_model_status = "not_loaded"


def get_device():
    """Detect best available device."""
    global _device
    if not TORCH_AVAILABLE:
        _device = "cpu"
        return _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {_device}")
    return _device


def ensure_model_download() -> str:
    """Download model weights from HuggingFace Hub if not cached."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    local_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
    if os.path.isfile(local_path):
        return local_path

    try:
        from huggingface_hub import hf_hub_download
        logger.info(f"Downloading {MODEL_FILENAME} from {HUGGINGFACE_REPO}...")
        return hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            filename=MODEL_FILENAME,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
        )
    except Exception as e:
        logger.warning(f"Failed to download primary model: {e}")
        try:
            from huggingface_hub import hf_hub_download
            alt_path = os.path.join(MODEL_DIR, FINETUNED_FILENAME)
            if os.path.isfile(alt_path):
                return alt_path
            return hf_hub_download(
                repo_id=HUGGINGFACE_REPO,
                filename=FINETUNED_FILENAME,
                local_dir=MODEL_DIR,
                local_dir_use_symlinks=False,
            )
        except Exception as e2:
            logger.warning(f"Failed to download finetuned model: {e2}")
            return None


def _build_unet_architecture():
    """Build candidate U-Net architecture."""
    try:
        import segmentation_models_pytorch as smp
        return smp.Unet(encoder_name="resnet34", in_channels=1, classes=1)
    except ImportError:
        pass

    import torch.nn as nn

    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x): return self.conv(x)

    class VanillaUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = DoubleConv(1, 64)
            self.enc2 = DoubleConv(64, 128)
            self.enc3 = DoubleConv(128, 256)
            self.enc4 = DoubleConv(256, 512)
            self.bottleneck = DoubleConv(512, 1024)
            self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
            self.dec4 = DoubleConv(1024, 512)
            self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
            self.dec3 = DoubleConv(512, 256)
            self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
            self.dec2 = DoubleConv(256, 128)
            self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec1 = DoubleConv(128, 64)
            self.final = nn.Conv2d(64, 1, 1)
            self.pool = nn.MaxPool2d(2)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
            d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
            d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
            return self.final(d1)

    return VanillaUNet()


def load_model():
    """
    Safely loads U-Net with weights_only=True and verifies architecture match.
    Downgrades to fallback if missing keys exceed 10%.
    """
    global _model, _model_status

    if not TORCH_AVAILABLE:
        _model_status = "fallback"
        return

    checkpoint_path = ensure_model_download()
    if checkpoint_path is None:
        _model_status = "fallback"
        return

    try:
        import torch
        device = get_device()

        # Defect #5 fix: weights_only=True to eliminate RCE vulnerability
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        except Exception as e:
            logger.warning(f"weights_only=True load rejected checkpoint: {e}. Engaging fallback.")
            _model_status = "fallback"
            return

        if isinstance(checkpoint, dict):
            state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
            model = _build_unet_architecture()

            # Defect #4 fix: detect incompatible architecture keys
            incompatible = model.load_state_dict(state_dict, strict=False)
            total_keys = len(list(model.state_dict().keys()))
            missing_count = len(incompatible.missing_keys)

            if total_keys > 0 and (missing_count / total_keys) > 0.10:
                logger.error(f"Model architecture mismatch: {missing_count}/{total_keys} missing keys. Engaging fallback.")
                _model_status = "fallback"
                return

            model = model.to(device)
            model.eval()
            _model = model
            _model_status = "loaded"
            logger.info("Pretrained U-Net loaded successfully")
        else:
            logger.warning("Checkpoint was not a state dict. Engaging fallback.")
            _model_status = "fallback"

    except Exception as e:
        logger.error(f"Error initializing model: {e}")
        _model_status = "fallback"


def predict_stroke_mask(scan_gray: np.ndarray) -> np.ndarray:
    """Runs U-Net inference if loaded; otherwise returns None to trigger vision_engine differencing."""
    global _model, _model_status

    if _model_status == "not_loaded":
        load_model()

    if _model_status != "loaded" or _model is None:
        return None

    import torch
    import cv2

    device = get_device()
    orig_h, orig_w = scan_gray.shape
    resized = cv2.resize(scan_gray, MODEL_INPUT_SIZE, interpolation=cv2.INTER_LINEAR)
    inp = torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)

    try:
        with torch.no_grad():
            out = _model(inp)
            prob = torch.sigmoid(out).squeeze().cpu().numpy()
            mask = (prob >= THRESHOLD).astype(np.uint8) * 255
            return cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return None


def get_status() -> dict:
    return {
        "model_status": _model_status,
        "device": str(_device) if _device else "cpu",
        "huggingface_repo": HUGGINGFACE_REPO,
        "model_file": MODEL_FILENAME,
    }
