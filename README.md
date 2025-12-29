<div align="center">
<h1>Image2ID: Image Generation in Seconds</h1>

</div>

Image2ID is a new state-of-the-art tuning-free method to achieve ID-Preserving generation with only single image, supporting various downstream tasks.

<img src='assets/applications.png'>

## Demos

### Stylized Synthesis

<p align="center">
  <img src="assets/StylizedSynthesis.png">
</p>

### Kolors Version

We have adapted Image2ID for [Kolors](https://huggingface.co/Kwai-Kolors/Kolors-diffusers). Leveraging Kolors' robust text generation capabilities 👍👍👍, Image2ID can be integrated with Kolors to simultaneously generate **ID** and **text**.


| demo | demo | demo |
|:-----:|:-----:|:-----:|
<img src="./assets/kolor/demo_1.jpg" >|<img src="./assets/kolor/demo_2.jpg" >|<img src="./assets/kolor/demo_3.jpg" >|



## Download

You can directly download the model from [Huggingface](https://huggingface.co/ImageX/Image2ID).
You also can download the model in python script:

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="ImageX/Image2ID", filename="ControlNetModel/config.json", local_dir="./checkpoints")
hf_hub_download(repo_id="ImageX/Image2ID", filename="ControlNetModel/diffusion_pytorch_model.safetensors", local_dir="./checkpoints")
hf_hub_download(repo_id="ImageX/Image2ID", filename="ip-adapter.bin", local_dir="./checkpoints")
```

Or run the following command to download all models:

```python
pip install -r gradio_demo/requirements.txt
python gradio_demo/download_models.py
```

If you cannot access to Huggingface, you can use [hf-mirror](https://hf-mirror.com/) to download models.
```python
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download InstantX/InstantID --local-dir checkpoints --local-dir-use-symlinks False
```

For face encoder, you need to manually download via this [URL](https://github.com/deepinsight/insightface/issues/1896#issuecomment-1023867304) to `models/antelopev2` as the default link is invalid. Once you have prepared all models, the folder tree should be like:

```
  .
  ├── models
  ├── checkpoints
  ├── ip_adapter
  ├── pipeline_stable_diffusion_xl_instantid.py
  └── README.md
```

## Usage

If you want to reproduce results in the paper, please refer to the code in [infer_full.py](infer_full.py). If you want to compare the results with other methods, even without using depth-controlnet, it is recommended that you use this code. 

If you are pursuing better results, it is recommended to follow [InstantID-Rome](https://github.com/instantX-research/InstantID-Rome).

The following code👇 comes from [infer.py](infer.py). If you want to quickly experience InstantID, please refer to the code in [infer.py](infer.py). 



```python
# !pip install opencv-python transformers accelerate insightface
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel

import cv2
import torch
import numpy as np
from PIL import Image

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

# prepare 'antelopev2' under ./models
app = FaceAnalysis(name='antelopev2', root='./', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# prepare models under ./checkpoints
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# load IdentityNet
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

base_model = 'wangqixun/YamerMIX_v8'  # from https://civitai.com/models/84040?modelVersionId=196039
pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model,
    controlnet=controlnet,
    torch_dtype=torch.float16
)
pipe.cuda()

# load adapter
pipe.load_ip_adapter_instantid(face_adapter)
```

Then, you can customized your own face images

```python
# load an image
face_image = load_image("./examples/yann-lecun_resize.jpg")

# prepare face emb
face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1]  # only use the maximum face
face_emb = face_info['embedding']
face_kps = draw_kps(face_image, face_info['kps'])

# prompt
prompt = "film noir style, ink sketch|vector, male man, highly detailed, sharp focus, ultra sharpness, monochrome, high contrast, dramatic shadows, 1940s style, mysterious, cinematic"
negative_prompt = "ugly, deformed, noisy, blurry, low contrast, realism, photorealistic, vibrant, colorful"

# generate image
image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image_embeds=face_emb,
    image=face_kps,
    controlnet_conditioning_scale=0.8,
    ip_adapter_scale=0.8,
).images[0]
```

To save VRAM, you can enable CPU offloading
```python
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()
```

## Speed Up with LCM-LoRA

Our work is compatible with [LCM-LoRA](https://github.com/luosiallen/latent-consistency-model). First, download the model.

```python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="latent-consistency/lcm-lora-sdxl", filename="pytorch_lora_weights.safetensors", local_dir="./checkpoints")
```

To use it, you just need to load it and infer with a small num_inference_steps. Note that it is recommendated to set guidance_scale between [0, 1].
```python
from diffusers import LCMScheduler

lcm_lora_path = "./checkpoints/pytorch_lora_weights.safetensors"

pipe.load_lora_weights(lcm_lora_path)
pipe.fuse_lora()
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

num_inference_steps = 10
guidance_scale = 0
```

## Start a local gradio demo <a href='https://github.com/gradio-app/gradio'><img src='https://img.shields.io/github/stars/gradio-app/gradio'></a>
Run the following command:

```python
python gradio_demo/app.py
```

or MultiControlNet version:
```python
gradio_demo/app-multicontrolnet.py 
```

## Usage Tips
- For higher similarity, increase the weight of controlnet_conditioning_scale (IdentityNet) and ip_adapter_scale (Adapter).
- For over-saturation, decrease the ip_adapter_scale. If not work, decrease controlnet_conditioning_scale.
- For higher text control ability, decrease ip_adapter_scale.
- For specific styles, choose corresponding base model makes differences.
- We have not supported multi-person yet, only use the largest face as reference facial landmarks.
- We provide a [style template](https://github.com/ahgsql/StyleSelectorXL/blob/main/sdxl_styles.json) for reference.
