import torch
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator

def load_sam_model(checkpoint_path="mobile_sam.pt"):
    model_type = "vit_t"
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
    return SamAutomaticMaskGenerator(sam)

