import torch
import os
import numpy as np
import folder_paths
import comfy.model_management
from comfy.clip_vision import load_torch_file, clip_preprocess

def get_model_path(model_name):
    try:
        path = folder_paths.get_folder_paths(model_name)[0]
    except:
        path = os.path.join(folder_paths.models_dir, model_name)
    return path

class ClipVisionStyleLoader:
    @classmethod
    def INPUT_TYPES(cls):
        crop_methods = ["none", "center", "mask"]
        
        return {"required": 
                   {"clip_vision": (folder_paths.get_filename_list("clip_vision"), ),
                    "style_model": (folder_paths.get_filename_list("style_models"), ),
                    "image": ("IMAGE",),
                    "crop_method": (crop_methods, {"default": "none"}),
                   },
                "optional": {
                    "mask": ("MASK", ),
                }
               }
    
    RETURN_TYPES = ("IMAGE", "STYLE_MODEL", "CLIP_VISION_OUTPUT")
    RETURN_NAMES = ("IMAGE", "STYLE_MODEL", "CLIP_VISION_OUTPUT")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "process_image"
    CATEGORY = "🧪AILab/⚛️ReduxFineTune"
    
    def crop_center(self, image, target_size=224):
        h, w = image.shape[1], image.shape[2]
        crop_size = min(h, w)
        y = (h - crop_size) // 2
        x = (w - crop_size) // 2
        cropped = image[:, y:y+crop_size, x:x+crop_size, :]
        return cropped
    
    def crop_mask(self, image, mask, target_size=None):
        if mask.shape[1:3] != image.shape[1:3]:
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(1), 
                size=(image.shape[1], image.shape[2]), 
                mode='bilinear', 
                antialias=False
            ).squeeze(1)
        
        if mask.device.type != 'cpu':
            mask_np = mask[0].cpu().numpy()
        else:
            mask_np = mask[0].numpy()
            
        y_indices, x_indices = np.where(mask_np > 0.05)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return self.crop_center(image)
            
        y1, y2 = np.min(y_indices), np.max(y_indices)
        x1, x2 = np.min(x_indices), np.max(x_indices)
        
        cropped = image[:, y1:y2, x1:x2, :]
        return cropped
    
    def process_image(self, clip_vision, style_model, image, crop_method, mask=None):
        # Load CLIP model
        clip_path = folder_paths.get_full_path("clip_vision", clip_vision)
        clip_vision = comfy.clip_vision.load(clip_path)
        if clip_vision is None:
            raise Exception(f"CLIP Vision model {clip_vision} not found")

        # Load style model
        style_model_path = folder_paths.get_full_path("style_models", style_model)
        style_model = comfy.sd.load_style_model(style_model_path)

        # Process image based on crop method
        if crop_method == "none":
            cropped_image = image
            crop_flag = False
        elif crop_method == "center":
            cropped_image = self.crop_center(image)
            crop_flag = True
        elif crop_method == "mask" and mask is not None:
            cropped_image = self.crop_mask(image, mask)
            crop_flag = True
        else:
            cropped_image = image
            crop_flag = False

        # Clone image for CLIP processing
        clip_image = cropped_image.clone()

        # Encode with CLIP
        try:
            out = clip_vision.encode_image(clip_image, crop=crop_flag)
        except:
            out = clip_vision.encode_image(clip_image)
            
        return [cropped_image], style_model, out

NODE_CLASS_MAPPINGS = {
    "ClipVisionStyleLoader": ClipVisionStyleLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClipVisionStyleLoader": "CLIP Vision + Style Loader",
}