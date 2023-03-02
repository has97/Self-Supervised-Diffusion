from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionImageVariationPipeline
from diffusers.utils import logging
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
import torchvision.transforms as transforms
# from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
class Augmentation(nn.Module):
    pipe = None 
    def __init__(self,strength=0.1,guidance=0.75,prompt=["High detail 4K Photo of the image"]*2,diff_steps=50):
        super(Augmentation, self).__init__()
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda:1")
        # self.pipe =  StableDiffusionImageVariationPipeline.from_pretrained("lambdalabs/sd-image-variations-diffusers",revision="v2.0").to("cuda:1")
        # self.pipe.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        # self.pipe.vae.enable_xformers_memory_efficient_attention(attention_op=None)
        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        self.prompt = prompt
        self.strength =strength
        self.guidance_scale =guidance
        self.num_inference_steps= diff_steps
        self.transforms =transform = transforms.Compose([
                                            transforms.Resize(224),
                                            transforms.ToTensor(),
                                            transforms.Lambda(lambda t: (t * 2) - 1)
                                        ])
    def forward(self, image):
        # L=[]
        # for i in range(3)
        # canvas = image.resize((512, 512), Image.BILINEAR)
        canvas = image
        # for i in range(3):
        with autocast(device_type='cuda', dtype=torch.float16):
                # with autocast(device_type='cuda', dtype=torch.float16):
                # print("here")
                out = self.pipe(prompt=self.prompt, image=canvas, strength=self.strength,guidance_scale=self.guidance_scale,num_inference_steps=self.num_inference_steps,num_images_per_prompt=2).images
                # out = self.pipe(image=canvas,guidance_scale=self.guidance_scale,num_inference_steps=self.num_inference_steps).images[0]
        # for i in range(3):
        #     out[i] = out[i].resize(image.size, Image.BILINEAR)
        # L.append(out)
        # print(len(out))
        return out
        
        
        