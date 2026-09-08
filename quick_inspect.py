import torch

checkpoint_path = "models/best_unet.pt"

try:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    print("Type of checkpoint:", type(checkpoint))
    if isinstance(checkpoint, dict):
        print("Keys found in checkpoint:", list(checkpoint.keys())[:10])
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))
        first_few_layers = list(state_dict.keys())[:5]
        print("First layer names:", first_few_layers)
        if any("encoder" in k for k in first_few_layers):
            print("💡 Likely from 'segmentation_models_pytorch' (SMP) library!")
        elif any("conv1" in k for k in first_few_layers):
            print("💡 Likely a standard torchvision ResNet or custom basic U-Net.")
    else:
        print("Direct model object loaded! Architecture class is embedded.")
except Exception as e:
    print(f"Error loading checkpoint: {e}")
