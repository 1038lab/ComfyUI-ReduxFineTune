# ComfyUI-ReduxFineTune

## Introduction

ComfyUI-ReduxFineTune is a custom node for ComfyUI that enables advanced style fine-tuning using the Flux Redux approach. It offers multiple unified fusion modes for precise and consistent control over style transfer, allowing users to fine-tune image styles with high flexibility and detail.

$$\textcolor{red}{\Huge \text{If this custom node helps you or you like my work, please give me ⭐ on this repo!}}$$  
$$\textcolor{red}{\Huge \text{It's a great encouragement for my efforts!}}$$

## News & Updates
- **2025/05/04**: Update ComfyUI-RMBG to **v1.1.0** ( [update.md](https://github.com/1038lab/ComfyUII-ReduxFineTune/blob/main/update.md#v110-20250504) )

**Key Features:**
- Multiple fusion modes for a variety of style transfer needs, from subtle to extreme.
- Precise fusion strength control via sliders for fine-grained adjustment.
- Both simple and advanced nodes available to suit different user needs.
- Easy to extend and maintain: all fusion modes and algorithms are centrally managed.
- **SUPER REDUX** is a unique enhancement feature of this node. When enabled, significantly boosting style strength and detail richness—ideal for users seeking the most dramatic style effects. 

![ReducFineTune_Nodes](https://github.com/user-attachments/assets/a00b2261-098b-4518-864f-24e8598856b1)

## Nodes Overview

![ComfyUI-ReduxFineTune_](https://github.com/user-attachments/assets/543a2748-85fa-4fee-ac82-1630e58d599d)

### ReduxFineTune

The **ReduxFineTune** node provides a simple and user-friendly tool for style fusion, requiring only the core parameters to efficiently blend text and image features. It features a precise fusion strength slider (0.1-2.0) for fine control over the style intensity. This node is ideal for most standard style fine-tuning scenarios.

![reduxfinetune_sample2](https://github.com/user-attachments/assets/880648ac-0a96-4415-aa83-ea4e1f35850c)

### ReduxFineTuneAdvanced
The **ReduxFineTuneAdvanced** node is designed for advanced users, offering support for cropping, masking, region-wise fusion, and fine-grained control over various style strengths and details. This node is suitable for scenarios that demand higher customization and control over style transfer.

Both nodes support all unified fusion modes and can enable the SUPER REDUX enhancement mode, making them suitable for everything from quick results to highly detailed style control.

**Function:**
Fine-tune style by fusing text and image features with multiple blend modes and strengths.

## Fusion Modes (Blend Modes)

All fusion modes (blend modes) are now globally unified in the codebase. Both the advanced and simple nodes use the same set of fusion modes, and their behavior is guaranteed to be consistent.

**Available Fusion Modes:**
| Fusion Mode    | Description                                                                                                                         |
|----------------|-------------------------------------------------------------------------------------------------------------------------------------|
| **Mix**            | Linear blend. Mainly uses the reference image. Suitable for style transfer and light style fusion.                                  |
| **Enhance**        | Feature enhancement. Improves details and realism. Suitable for stronger style influence.                                           |
| **Sharpen**        | Frequency enhancement. Highlights high-frequency details, making the image sharper.                                                 |
| **AdaIN**          | Adaptive Instance Normalization. Automatically matches the mean and variance of style and content features. Suitable for style transfer. |
| **Residual**       | Residual blend. Fuses the difference between style and content. Suitable for balancing both.                                        |
| **Max**            | Element-wise maximum. Keeps the stronger part of style and content features. Suitable for highlighting prominent features.          |
| **Min**            | Element-wise minimum. Keeps the weaker part of style and content features. Suitable for soft fusion.                                |
| **Random**         | Random blend. Randomly selects style or content features. Suitable for generating diverse effects.                                  |
| **FrequencyMix**   | Frequency domain blend. Low frequency from image, high frequency from text. Suitable for structure and detail separation fusion.    |

**How to extend:**
To add or modify a fusion mode, simply update the `FUSION_MODES` list and the `fuse_features` function in the code. All nodes will automatically use the new or updated mode.

## Installation

1. Clone this repository to your ComfyUI custom_nodes folder:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/1038lab/ComfyUI-ReduxFineTune.git
```

2. Install the required dependencies:

```bash
/ComfyUI/python_embeded/python -m pip install -r requirements.txt
```

## Usage

After installation, you can use the ReduxFineTune node in your ComfyUI workflow. Drag and drop the node into your canvas and configure the input parameters as needed.

### Simple Node Parameters
- **conditioning**: Input conditioning from a previous node
- **style_model**: The style model to use (CLIP Vision L, etc.)
- **clip_vision_output**: Output from CLIP Vision node
- **fusion_mode**: Select one of the available fusion modes
- **fusion_strength**: Adjust the strength of fusion (0.1-2.0)
- **SUPER_REDUX**: Enable for extra enhancement (toggle)

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any changes or improvements.

## License

This project is licensed under the GPL-3.0 License - see the LICENSE file for details.

## Acknowledgments

Thanks to all contributors and users who have supported the development of ComfyUI-ReduxFineTune.
