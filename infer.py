import cv2
import torch
import numpy as np
from PIL import Image

# 修复 huggingface_hub 导入问题
import huggingface_hub
if not hasattr(huggingface_hub, 'cached_download'):
    from huggingface_hub.file_download import hf_hub_download
    huggingface_hub.cached_download = hf_hub_download

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps

def resize_img(input_image, max_side=1280, min_side=1024, size=None, 
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):

    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio*w), round(ratio*h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio*w), round(ratio*h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y+h_resize_new, offset_x:offset_x+w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


if __name__ == "__main__":
    # ========== 关键修改1：强制设置设备为CPU ==========
    device = torch.device("cpu")
    # 禁用CUDA相关检测，避免代码误判
    torch.cuda.is_available = lambda: False

    # Load face encoder（修改providers，仅用CPU）
    app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])  # 移除CUDAExecutionProvider
    app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 强制CPU推理

    # Path to InstantID models
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Load pipeline（修改torch_dtype为float32，CPU不支持float16）
    controlnet = ControlNetModel.from_pretrained(
        controlnet_path, 
        torch_dtype=torch.float32  # 关键：CPU不兼容float16，改为float32
    )

    # base_model_path = 'stabilityai/stable-diffusion-xl-base-1.0'
    base_model_path = 'wangqixun/YamerMIX_v8'

    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
        base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float32,  # 关键：CPU不兼容float16，改为float32
    )
    # ========== 关键修改2：强制移到CPU ==========
    pipe = pipe.to(device)
    
    # load adapter
    pipe.load_ip_adapter_instantid(face_adapter)
    # ========== 关键修改3：禁用GPU相关优化 ==========
    # 注释掉CPU不支持的offload/tiling（这些是GPU优化）
    # pipe.enable_model_cpu_offload()  # CPU运行时无需offload
    # pipe.enable_vae_tiling()        # CPU下该优化无效，易报错

    # Infer setting
    prompt = "analog film photo of a man. faded film, desaturated, 35mm photo, grainy, vignette, vintage, Kodachrome, Lomography, stained, highly detailed, found footage, masterpiece, best quality"
    n_prompt = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"

    # load an image
    face_image = load_image("./examples/yann-lecun_resize.jpg")
    face_image = resize_img(face_image)

    # prepare face emb
    face_info = app.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
    face_info = sorted(face_info, key=lambda x:(x['bbox'][2]-x['bbox'][0])*(x['bbox'][3]-x['bbox'][1]))[-1] # only use the maximum face
    face_emb = face_info['embedding']
    face_kps = draw_kps(face_image, face_info['kps'])

    # ========== 关键修改4：输入数据移到CPU ==========
    face_emb = torch.from_numpy(face_emb).to(device, dtype=torch.float32)
    # 确保controlnet的输入图像张量也在CPU上
    image = pipe(
        prompt=prompt,
        negative_prompt=n_prompt,
        image_embeds=face_emb,
        image=face_kps,
        controlnet_conditioning_scale=0.8,
        ip_adapter_scale=0.8,
        num_inference_steps=30,
        guidance_scale=5,
        generator=torch.manual_seed(42),  # 可选：固定种子，避免CPU随机数问题
    ).images[0]

    image.save('result.jpg')