# ComfyUI-ComfyUI-ReduxFineTune Update Log

## V1.2.0 (2025/06/21)
  * Performance Optimization:
    - Improved code efficiency and reduced memory usage
    - Enhanced processing speed for all fusion modes
    - Better handling of large images and batch processing
  * Enhanced SUPER_REDUX mode:
    - Improved quality by using enhanced parameters
    - Better memory management for stable performance
  * Modified ReduxFineTuneAdvanced:
    - Updated return types to match ReduxFineTune: `(CONDITIONING, STYLE_MODEL, CLIP_VISION_OUTPUT, IMAGE, MASK)`
    - Added image and mask pass-through for better workflow integration ([user request](https://github.com/1038lab/ComfyUI-ReduxFineTune/issues/4))
    - Fixed issue related to image quality reduction in SUPER_REDUX mode

![image](https://github.com/user-attachments/assets/6c61c02c-200c-4171-a5ac-1622682fe1df)

## V1.1.0 (2025/05/04)  
  * Added new node `ClipVisionStyleLoader` in category `🧪AILab/⚛️ReduxFineTune`  
    - Integrates CLIP Vision model and style model with unified image cropping functionality for streamlined processing.
  * Added Custom Node theme

![Clipstyleloader](https://github.com/user-attachments/assets/ea1828c5-42cf-46de-b67c-479090f323c7)

---

Feel free to open an issue if you encounter any problems or have suggestions.  
