from diffusers import StableDiffusionImg2ImgPipeline
from diffusers.utils import logging
from PIL import Image, ImageOps
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Augmentation():
    pipe = None 
    def __init__(self,strength=0.4,guidance=7.5,prompt="sketch of the image",diff_steps=150):
        super(Augmentation, self).__init__()
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", use_auth_token=True,
                revision="fp16", 
                torch_dtype=torch.float16
            ).to('cuda')
        logging.disable_progress_bar()
        self.pipe.set_progress_bar_config(disable=True)
        self.pipe.safety_checker = None
        self.prompt = prompt
        self.strength =strength
        self.guidance_scale =guidance
        self.num_inference_steps= diff_steps
    def forward(self, image):
        canvas = image.resize((512, 512), Image.BILINEAR)
        with autocast("cuda"):
            canvas = self.pipe().images[0]
        canvas = canvas.resize(image.size, Image.BILINEAR)
        return canvas
        
        
        