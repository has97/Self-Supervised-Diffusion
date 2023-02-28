from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import logging
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

class Augmentation(nn.Module):
    pipe = None 
    def __init__(self,strength=0.2,guidance=0.75,prompt=["Painting of the image"],diff_steps=30):
        super(Augmentation, self).__init__()
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to('cuda')
        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        self.prompt = prompt
        self.strength =strength
        self.guidance_scale =guidance
        self.num_inference_steps= diff_steps
    def forward(self, image):
        L=[]
        for i in range(3):
            canvas = image.resize((512, 512), Image.BILINEAR)
            with autocast(device_type='cuda', dtype=torch.float16):
                # with autocast(device_type='cuda', dtype=torch.float16):
                # print("here")
                out = self.pipe(prompt=self.prompt, image=canvas, strength=self.strength,guidance_scale=self.guidance_scale,num_inference_steps=self.num_inference_steps).images[0]
                out = out.resize(image.size, Image.BILINEAR)
            L.append(out)
        return L
        
        
        