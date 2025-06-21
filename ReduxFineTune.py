import torch
import math
import torch.nn.functional as F
from functools import lru_cache
import gc

FEATURE_DIM = 4096

# Fusion mode list for both nodes
FUSION_MODES = [
    "Mix", "Enhance", "Sharpen", "AdaIN", "Residual", "Max", "Min", "Random", "FrequencyMix",
    "Multiply", "AttnBias"
]

# Fusion mode tooltips for display
FUSION_MODE_TOOLTIPS = {
    "Mix": "Mix - Linear blend, mainly uses the reference image, suitable for style transfer and light style fusion.",
    "Enhance": "Enhance - Feature enhancement, improves details and realism, suitable for stronger style influence.",
    "Sharpen": "Sharpen - Frequency enhancement, highlights high-frequency details, makes the image sharper.",
    "AdaIN": "AdaIN - Adaptive Instance Normalization, automatically matches the mean and variance of style and content features, suitable for style transfer.",
    "Residual": "Residual - Residual blend, fuses the difference between style and content, suitable for balancing both.",
    "Max": "Max - Element-wise maximum, keeps the stronger part of style and content features, suitable for highlighting prominent features.",
    "Min": "Min - Element-wise minimum, keeps the weaker part of style and content features, suitable for soft fusion.",
    "Random": "Random - Random blend, randomly selects style or content features, suitable for generating diverse effects.",
    "FrequencyMix": "FrequencyMix - Frequency domain blend, low frequency from image, high frequency from text, suitable for structure and detail separation fusion.",
    "Multiply": "Multiply - Directly multiplies the style feature by the strength, similar to StyleModelApply's multiply.",
    "AttnBias": "AttnBias - Adds a log(strength) bias to the style feature, similar to StyleModelApply's attn_bias."
}
FUSION_MODE_TOOLTIP_TEXT = "\n".join([FUSION_MODE_TOOLTIPS[m] for m in FUSION_MODES])

# Combined simple fusion operations into a single function
def _basic_fusion(image_feat, text_feat, strength, region_size, mode):
    text_part = text_feat[..., :region_size]
    
    if mode == "mix":
        alpha = min(strength / 10.0, 1.0)
        return (1 - alpha) * image_feat + alpha * text_part
    elif mode == "enhance":
        return image_feat + strength * text_part
    elif mode == "residual":
        return image_feat + strength * (text_part - image_feat)
    elif mode == "max":
        return torch.max(image_feat, text_part)
    elif mode == "min":
        return torch.min(image_feat, text_part)
    elif mode == "multiply":
        return image_feat * strength
    elif mode == "attnbias":
        attn_bias = torch.log(torch.tensor([strength], dtype=image_feat.dtype, device=image_feat.device))
        return image_feat + attn_bias
    elif mode == "random":
        mask = torch.rand_like(image_feat)
        return mask * image_feat + (1 - mask) * text_part
    
    return image_feat

@lru_cache(maxsize=16)
def get_sharpen_kernel(channels, device, dtype):
    kernel = torch.tensor([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=dtype, device=device)
    return kernel.expand(channels, 1, 3, 3)

def _sharpen_fusion(image_feat, text_feat, strength, region_size):
    feat = image_feat
    if feat.dim() == 4:
        kernel = get_sharpen_kernel(feat.shape[1], feat.device, feat.dtype)
        high_freq = F.conv2d(feat, kernel, padding=1, groups=feat.shape[1])
        return feat + strength * high_freq
    return feat

def _adain_fusion(image_feat, text_feat, strength, region_size):
    img_mean = image_feat.mean(dim=-1, keepdim=True)
    img_std = image_feat.std(dim=-1, keepdim=True)
    txt_mean = text_feat[..., :region_size].mean(dim=-1, keepdim=True)
    txt_std = text_feat[..., :region_size].std(dim=-1, keepdim=True)
    return txt_std * (image_feat - img_mean) / (img_std + 1e-6) + txt_mean

@lru_cache(maxsize=32)
def _get_fft_params(freq_dim, device, dtype):
    split = max(1, int(freq_dim * 0.5))
    return split, torch.ones((freq_dim,), device=device, dtype=dtype)

def _frequency_mix_fusion(image_feat, text_feat, strength, region_size):
    image_feat = image_feat.float()
    text_feat_part = text_feat[..., :region_size].float()
    
    img_fft = torch.fft.rfft(image_feat, dim=-1)
    img_magnitudes = torch.abs(img_fft)
    img_phases = torch.angle(img_fft)
    txt_fft = torch.fft.rfft(text_feat_part, dim=-1)
    txt_magnitudes = torch.abs(txt_fft)
    freq_dim = img_fft.shape[-1]
    
    split, _ = _get_fft_params(freq_dim, img_magnitudes.device, img_magnitudes.dtype)
    
    mixed_magnitudes = img_magnitudes.clone() if freq_dim > 1 and txt_magnitudes.shape[-1] >= freq_dim else img_magnitudes
    
    if freq_dim > 1 and txt_magnitudes.shape[-1] >= freq_dim:
        scale_factor = strength * 0.5 + 0.5
        mixed_magnitudes[..., split:] = txt_magnitudes[..., split:] * scale_factor
        
    mixed_fft = mixed_magnitudes * torch.exp(1j * img_phases)
    result = torch.fft.irfft(mixed_fft, n=region_size, dim=-1)
    
    result_mean = result.mean(dim=-1, keepdim=True)
    result_std = result.std(dim=-1, keepdim=True) + 1e-6
    result_norm = (result - result_mean) / result_std
    
    mix_ratio = min(max(strength, 0.0), 1.0)
    return (1.0 - mix_ratio) * image_feat + mix_ratio * result_norm.to(image_feat.dtype)

@lru_cache(maxsize=64)
def _get_fusion_strength(mode, strength):
    if mode == "Mix":
        return min(strength / 10.0, 1.0)
    elif mode == "FrequencyMix":
        return min(max(strength, 0.0), 1.0)
    return strength

def fuse_features(image_feat, text_feat, mode: str, strength: float, region_size: int = -1):
    if region_size < 0:
        region_size = image_feat.shape[-1]
    
    if mode == "Mix":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "mix")
    elif mode == "Enhance":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "enhance")
    elif mode == "AdaIN":
        return _adain_fusion(image_feat, text_feat, strength, region_size)
    elif mode == "Residual":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "residual")
    elif mode == "Max":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "max")
    elif mode == "Min":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "min")
    elif mode == "Random":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "random")
    elif mode == "FrequencyMix":
        return _frequency_mix_fusion(image_feat, text_feat, strength, region_size)
    elif mode == "Multiply":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "multiply")
    elif mode == "AttnBias":
        return _basic_fusion(image_feat, text_feat, strength, region_size, "attnbias")
    elif mode == "Sharpen":
        return _sharpen_fusion(image_feat, text_feat, strength, region_size)
    
    return image_feat

class ReduxFineTune:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision_output": ("CLIP_VISION_OUTPUT",),
                "fusion_mode": (FUSION_MODES, {"default": "Mix", "tooltip": FUSION_MODE_TOOLTIP_TEXT}),
                "fusion_strength": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1, "display": "slider", "tooltip": "Fusion strength, higher values apply more style from reference image. 0.1-0.5: subtle effect, 1.0: balanced, 1.5-2.0: strong effect."}),
                "SUPER_REDUX": ("BOOLEAN", {"default": False, "tooltip": "Enable Super Redux mode for extra enhancement."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING","STYLE_MODEL","CLIP_VISION_OUTPUT")
    FUNCTION = "apply_style"
    CATEGORY = "🧪AILab/⚛️ReduxFineTune"

    @staticmethod
    @torch.no_grad()
    def _apply_once(conditioning, style_model, clip_vision_output, fusion_mode, strength):
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1).unsqueeze(dim=0)
        
        c_out = []
        for t in conditioning:
            txt_feat = t[0].mean(dim=1)
            fused_cond = fuse_features(cond, txt_feat, fusion_mode, strength)
            n = [torch.cat((t[0], fused_cond), dim=1), t[1].copy()]
            c_out.append(n)
            
        del cond, txt_feat, fused_cond
        return (c_out,)

    def apply_style(self, conditioning, style_model, clip_vision_output, fusion_mode="Mix", fusion_strength=1.0, SUPER_REDUX=False):
        result = self._apply_once(conditioning, style_model, clip_vision_output, fusion_mode, fusion_strength)
        
        if SUPER_REDUX:
            result = self._apply_once(result[0], style_model, clip_vision_output, fusion_mode, fusion_strength)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        return result[0], style_model, clip_vision_output

class ReduxFineTuneAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "style_model": ("STYLE_MODEL",),
                "clip_vision": ("CLIP_VISION",),
                "image": ("IMAGE",),
                "crop": (["center", "mask_area", "none"], {
                    "default": "none",
                    "tooltip": "Crop mode: center - center crop, mask_area - crop to mask, none - no crop."
                }),
                "fusion_mode": (FUSION_MODES, {
                    "default": "Mix",
                    "tooltip": FUSION_MODE_TOOLTIP_TEXT
                }),
            },
            "optional": {
                "mask": ("MASK",),
                "style_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Overall style strength, higher means more like reference image."}),
                "color_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Color transfer strength. 0 means off, >0 means on with intensity."}),
                "content_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Content transfer strength. 0 means off, >0 means on with intensity."}),
                "structure_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Structure transfer strength. 0 means off, >0 means on with intensity."}),
                "texture_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 10.0, "step": 0.01, "tooltip": "Texture transfer strength. 0 means off, >0 means on with intensity."}),
                "prompt_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "tooltip": "Prompt (text) feature strength. Higher means more text influence."}),
                "feature_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Random noise strength for feature fusion. Can help fix or vary details."}),
                "feature_resolution": ("INT", {"default": 16, "min": 1, "max": 64, "step": 1, "tooltip": "Feature resolution for fusion. Higher means finer spatial detail."}),
                "SUPER_REDUX": ("BOOLEAN", {"default": False, "tooltip": "Enable Super Redux mode for extra enhancement and quality."}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STYLE_MODEL", "CLIP_VISION_OUTPUT", "IMAGE", "MASK")
    FUNCTION = "apply_style"
    CATEGORY = "🧪AILab/⚛️ReduxFineTune"

    @staticmethod
    @lru_cache(maxsize=8)
    def _get_interpolated_mask(mask, size, device):
        if mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif mask.dim() == 3:
            mask = mask.unsqueeze(1)
            
        return F.interpolate(
            mask.to(device),
            size=(size, size),
            mode='bilinear',
            align_corners=False
        ).flatten(1).unsqueeze(-1)

    def crop_to_mask_area(self, image, mask):
        if len(image.shape) == 4:
            image = image.squeeze(0)
        if len(mask.shape) == 3:
            mask = mask.squeeze(0)
            
        nonzero_coords = torch.nonzero(mask)
        if len(nonzero_coords) == 0:
            return image, mask
            
        top = nonzero_coords[:, 0].min().item()
        bottom = nonzero_coords[:, 0].max().item() + 1
        left = nonzero_coords[:, 1].min().item()
        right = nonzero_coords[:, 1].max().item() + 1
        
        cropped_image = image[top:bottom, left:right]
        cropped_mask = mask[top:bottom, left:right]
        
        cropped_image = cropped_image.unsqueeze(0)
        cropped_mask = cropped_mask.unsqueeze(0)
        return cropped_image, cropped_mask

    def encode_image_with_retry(self, clip_vision, image, crop_flag=False, max_retries=2):
        for attempt in range(max_retries):
            try:
                return clip_vision.encode_image(image, crop=crop_flag)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
        
    def apply_style(self, conditioning, style_model, clip_vision, image, crop="none", fusion_mode="Mix", 
                   style_strength=1.0, color_strength=0.0, content_strength=0.0, structure_strength=0.0, 
                   texture_strength=0.0, prompt_strength=1.0, feature_noise=0.0, feature_resolution=16, 
                   mask=None, SUPER_REDUX=False):
        
        @torch.no_grad()
        def _apply_once(conditioning, style_model, clip_vision, image, crop, fusion_mode, style_strength, 
                       color_strength, content_strength, structure_strength, texture_strength, 
                       prompt_strength, feature_noise, feature_resolution, mask):
            processed_image = image
            crop_flag = False
            if crop == "center":
                crop_flag = True
            elif crop == "mask_area" and mask is not None:
                processed_image, mask = self.crop_to_mask_area(processed_image, mask)
                
            clip_vision_output = self.encode_image_with_retry(clip_vision, processed_image, crop_flag)
            
            cond = style_model.get_cond(clip_vision_output)
            B, HW, C = cond.shape
            H = W = int(math.sqrt(HW))
            
            cond = cond.reshape(B, H, W, C)
            
            new_H = H * feature_resolution // 16
            new_W = W * feature_resolution // 16
            
            cond = F.interpolate(
                cond.permute(0, 3, 1, 2),
                size=(new_H, new_W),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1).reshape(B, -1, C).flatten(start_dim=0, end_dim=1)
            
            c_out = []
            feature_size = cond.shape[-1]
            splits = feature_size // 5
            
            for t in conditioning:
                txt, keys = t
                keys = keys.copy()
                
                if prompt_strength != 1.0:
                    txt = txt * (prompt_strength ** 3)
                    if prompt_strength > 1.0:
                        txt = txt.repeat(1, 2, 1)
                
                image_features = {
                    'style': cond[..., :splits],
                    'color': cond[..., splits:splits*2],
                    'content': cond[..., splits*2:splits*3],
                    'structure': cond[..., splits*3:splits*4],
                    'texture': cond[..., splits*4:]
                }
                
                text_features = txt.mean(dim=1)
                
                strengths = {
                    'style': style_strength,
                    'color': color_strength,
                    'content': content_strength,
                    'structure': structure_strength,
                    'texture': texture_strength
                }
                
                fused_features = {}
                for key, strength in strengths.items():
                    if strength > 0:
                        region_size = image_features[key].shape[-1]
                        fused = fuse_features(image_features[key], text_features, fusion_mode, strength, region_size)
                        fused_features[key] = fused
                    else:
                        fused_features[key] = image_features[key]
                
                combined_cond = torch.cat([
                    fused_features['style'],
                    fused_features['color'],
                    fused_features['content'],
                    fused_features['structure'],
                    fused_features['texture']
                ], dim=-1)
                
                del fused_features
                
                if combined_cond.shape[0] != txt.shape[0]:
                    combined_cond = combined_cond.expand(txt.shape[0], -1, -1)
                
                if mask is not None:
                    feature_size = int(math.sqrt(combined_cond.shape[1]))
                    processed_mask = self._get_interpolated_mask(mask, feature_size, combined_cond.device)
                    
                    if txt.shape[1] != combined_cond.shape[1]:
                        txt_expanded = txt.mean(dim=1, keepdim=True).expand(-1, combined_cond.shape[1], -1)
                    else:
                        txt_expanded = txt
                    
                    combined_cond = combined_cond * processed_mask + txt_expanded * (1 - processed_mask)
                    del txt_expanded
                
                if feature_noise > 0:
                    noise = torch.randn_like(combined_cond)
                    noise = (noise - noise.mean()) / (noise.std() + 1e-8)
                    
                    combined_cond = torch.lerp(combined_cond, noise, feature_noise)
                    combined_cond = combined_cond * (1.0 + feature_noise)
                    del noise
                
                c_out.append([torch.cat((txt, combined_cond), dim=1), keys])
                del combined_cond
            
            del cond, image_features
            return (c_out,)

        result = _apply_once(conditioning, style_model, clip_vision, image, crop, fusion_mode, 
                            style_strength, color_strength, content_strength, structure_strength, 
                            texture_strength, prompt_strength, feature_noise, feature_resolution, mask)
        
        if SUPER_REDUX:
            enhanced_strength = min(style_strength * 1.2, 2.0)
            enhanced_color = min(color_strength * 1.1, 10.0) if color_strength > 0 else color_strength
            enhanced_content = min(content_strength * 1.1, 10.0) if content_strength > 0 else content_strength
            enhanced_structure = min(structure_strength * 1.1, 10.0) if structure_strength > 0 else structure_strength
            enhanced_texture = min(texture_strength * 1.1, 10.0) if texture_strength > 0 else texture_strength
            
            result = _apply_once(result[0], style_model, clip_vision, image, crop, fusion_mode, 
                                enhanced_strength, enhanced_color, enhanced_content, enhanced_structure, 
                                enhanced_texture, prompt_strength, feature_noise, feature_resolution, mask)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        processed_image = image
        crop_flag = False
        if crop == "center":
            crop_flag = True
        elif crop == "mask_area" and mask is not None:
            processed_image, mask = self.crop_to_mask_area(processed_image, mask)
            
        clip_vision_output = self.encode_image_with_retry(clip_vision, processed_image, crop_flag)
        
        return result[0], style_model, clip_vision_output, image, mask if mask is not None else torch.zeros((1, image.shape[1], image.shape[2]), device=image.device)

NODE_CLASS_MAPPINGS = {
    "ReduxFineTune": ReduxFineTune,
    "ReduxFineTuneAdvanced": ReduxFineTuneAdvanced,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReduxFineTune": "Flux Redux Style Fine Tune (Simple)",
    "ReduxFineTuneAdvanced": "Flux Redux Style Fine Tune (Advanced)",
} 