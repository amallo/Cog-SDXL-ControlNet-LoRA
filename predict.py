import hashlib
import json
import os
import shutil
import subprocess
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import cv2
from cog import BasePredictor, Input, Path
from PIL import Image
from diffusers import (
    DDIMScheduler,
    DiffusionPipeline,
    UniPCMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel
)
from diffusers.models.attention_processor import LoRAAttnProcessor2_0
from diffusers.pipelines.stable_diffusion.safety_checker import (
    StableDiffusionSafetyChecker,
)
from diffusers.utils import load_image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import CLIPImageProcessor

from dataset_and_utils import TokenEmbeddingsHandler
from prompt import PROMPT_GENERAL, NEGATIVE_PROMPT


CONTROL_CACHE = "control-cache"
SDXL_MODEL_CACHE = "./sdxl-cache"
REALVISXL_MODEL_CACHE = "./RealVisXL_V3.0"
REFINER_MODEL_CACHE = "./refiner-cache"
SAFETY_CACHE = "./safety-cache"
FEATURE_EXTRACTOR = "./feature-extractor"
SDXL_URL = "https://weights.replicate.delivery/default/sdxl/sdxl-vae-upcast-fix.tar"
REALVISXL_URL = "https://weights.replicate.delivery/default/RealVisXL/RealVisXL_V3.0.tar"
REFINER_URL = (
    "https://weights.replicate.delivery/default/sdxl/refiner-no-vae-no-encoder-1.0.tar"
)
SAFETY_URL = "https://weights.replicate.delivery/default/sdxl/safety-1.0.tar"
LORA_WAR = "https://replicate.delivery/pbxt/SqRHCOhHkOoGJ5mW93YW0F863hV6IJ1vf96RvI5v65cWofkTA/trained_model.tar"
LORAS = {
    'abandon': "./trained_model/abandon",
    'vegetation': "./trained_model/vegetation",
    'war': "./trained_model/war"
}



class KarrasDPM:
    def from_config(config):
        return DPMSolverMultistepScheduler.from_config(config, use_karras_sigmas=True)


SCHEDULERS = {
    "DDIM": DDIMScheduler,
    "DPMSolverMultistep": DPMSolverMultistepScheduler,
    "HeunDiscrete": HeunDiscreteScheduler,
    "KarrasDPM": KarrasDPM,
    "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler,
    "K_EULER": EulerDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "UNI_PC_MULTISTEP": UniPCMultistepScheduler
}


def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)


class Predictor(BasePredictor):
    
    def setup(self, weights: Optional[Path] = None):
        """Load the model into memory to make running multiple predictions efficient"""
        start = time.time()
        self.tuned_weights = None
        if str(weights) == "weights":
            weights = None

     
        print("Loading safety checker...")
        if not os.path.exists(SAFETY_CACHE):
            download_weights(SAFETY_URL, SAFETY_CACHE)
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            SAFETY_CACHE, torch_dtype=torch.float16
        ).to("cuda")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(FEATURE_EXTRACTOR)

        #if not os.path.exists(SDXL_MODEL_CACHE):
        #    download_weights(SDXL_URL, SDXL_MODEL_CACHE)

        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.float16,
        )

        print("Loading SDXL Controlnet pipeline...")
        self.control_text2img_pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            REALVISXL_MODEL_CACHE,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.control_text2img_pipe.to("cuda")
        
        self.control_text2img_pipe.load_lora_weights(LORAS["abandon"] , weight_name = "lora.safetensors", adapter_name='abandon') 
        self.control_text2img_pipe.load_lora_weights(LORAS["vegetation"] , weight_name = "lora.safetensors", adapter_name='vegetation') 
        #self.control_text2img_pipe.load_lora_weights(LORAS["war"] , weight_name = "lora.safetensors", adapter_name='war')  
        
        if not os.path.exists(REFINER_MODEL_CACHE):
            download_weights(REFINER_URL, REFINER_MODEL_CACHE)

        print("Loading refiner pipeline...")
        self.refiner = DiffusionPipeline.from_pretrained(
            REFINER_MODEL_CACHE,
            text_encoder_2=self.control_text2img_pipe.text_encoder_2,
            vae=self.control_text2img_pipe.vae,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        )
        self.refiner.to("cuda")
        print("setup took: ", time.time() - start)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")
    
    def resize_image(self, image):
        image_width, image_height = image.size
        print("Original width:"+str(image_width)+", height:"+str(image_height))
        new_width, new_height = self.resize_to_allowed_dimensions(image_width, image_height)
        print("new_width:"+str(new_width)+", new_height:"+str(new_height))
        image = image.resize((new_width, new_height))
        return image, new_width, new_height
    
    def resize_to_allowed_dimensions(self, width, height):
        """
        Function re-used from Lucataco's implementation of SDXL-Controlnet for Replicate
        """
        # List of SDXL dimensions
        allowed_dimensions = [
            (512, 2048), (512, 1984), (512, 1920), (512, 1856),
            (576, 1792), (576, 1728), (576, 1664), (640, 1600),
            (640, 1536), (704, 1472), (704, 1408), (704, 1344),
            (768, 1344), (768, 1280), (832, 1216), (832, 1152),
            (896, 1152), (896, 1088), (960, 1088), (960, 1024),
            (1024, 1024), (1024, 960), (1088, 960), (1088, 896),
            (1152, 896), (1152, 832), (1216, 832), (1280, 768),
            (1344, 768), (1408, 704), (1472, 704), (1536, 640),
            (1600, 640), (1664, 576), (1728, 576), (1792, 576),
            (1856, 512), (1920, 512), (1984, 512), (2048, 512)
        ]
        # Calculate the aspect ratio
        aspect_ratio = width / height
        print(f"Aspect Ratio: {aspect_ratio:.2f}")
        # Find the closest allowed dimensions that maintain the aspect ratio
        closest_dimensions = min(
            allowed_dimensions,
            key=lambda dim: abs(dim[0] / dim[1] - aspect_ratio)
        )
        return closest_dimensions

    def image2canny(self, image):
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        return Image.fromarray(image)

    def run_safety_checker(self, image):
        safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(
            "cuda"
        )
        np_image = [np.array(val) for val in image]
        image, has_nsfw_concept = self.safety_checker(
            images=np_image,
            clip_input=safety_checker_input.pixel_values.to(torch.float16),
        )
        return image, has_nsfw_concept

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt",
            default=PROMPT_GENERAL,
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default=NEGATIVE_PROMPT,
        ),
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        prevent_nsfw: bool = Input(
            description="Prevent nsfw content",
            default=True,
        ),
        strength: float = Input(
            description="A higher value enforce image destruction",
            default=0.8,
            ge=0.0,
            le=1.0,
        ),
        condition_scale: float = Input(
            description="The bigger this number is, the more ControlNet interferes",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        num_outputs: int = Input(
            description="Number of images to output",
            ge=1,
            le=4,
            default=1,
        ),
        scheduler: str = Input(
            description="scheduler",
            choices=SCHEDULERS.keys(),
            default="K_EULER",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=50, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        refine: str = Input(
            description="Whether to use refinement steps or not",
            choices=["no_refiner", "base_image_refiner"],
            default="base_image_refiner",
        ),
        refine_steps: int = Input(
            description="For base_image_refiner, the number of steps to refine",
            default=10,
        ),
        apply_watermark: bool = Input(
            description="Applies a watermark to enable determining if an image is generated in downstream applications. If you have other provisions for generating or deploying images safely, you can use this to disable watermarking.",
            default=True,
        ),
        abandon_scale: float = Input(
            description="LoRA additive scale on abandoned buildings.",
            ge=0.0,
            le=1.0,
            default=0.6,
        ),
        vegetation_scale: float = Input(
            description="LoRA additive scale on vegetation.",
            ge=0.0,
            le=1.0,
            default=0.3,
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        # OOMs can leave vae in bad state
        if self.control_text2img_pipe.vae.dtype == torch.float32:
            self.control_text2img_pipe.vae.to(dtype=torch.float16)


        sdxl_kwargs = {}
        print(f"Prompt: {prompt}")
        image = self.load_image(image)
        image, width, height = self.resize_image(image)
        print("txt2img mode")
        sdxl_kwargs["image"] = self.image2canny(image)
        sdxl_kwargs["controlnet_conditioning_scale"] = condition_scale
        sdxl_kwargs["width"] = width
        sdxl_kwargs["height"] = height
        sdxl_kwargs["strength"] = strength
        pipe = self.control_text2img_pipe

        adapters = {}
        if abandon_scale == 0 and vegetation_scale == 0:
            pipe.disable_lora()
        else:    
            pipe.enable_lora()
            if abandon_scale>0:
                adapters['abandon'] = abandon_scale
            if vegetation_scale > 0:
                adapters['vegetation'] = vegetation_scale    
            pipe.set_adapters(list(adapters.keys()), adapter_weights=list(adapters.values()))

        if refine == "base_image_refiner":
            sdxl_kwargs["output_type"] = "latent"
            
        if not apply_watermark:
            # toggles watermark for this prediction
            watermark_cache = pipe.watermark
            pipe.watermark = None
            self.refiner.watermark = None

        pipe.scheduler = SCHEDULERS[scheduler].from_config(pipe.scheduler.config)
        generator = torch.Generator("cuda").manual_seed(seed)

        common_args = {
            "prompt": [prompt] * num_outputs,
            "negative_prompt": [negative_prompt] * num_outputs,
            "guidance_scale": guidance_scale,
            "generator": generator,
            "num_inference_steps": num_inference_steps,
        }

        sdxl_kwargs["cross_attention_kwargs"] = {"scale": 1.0}

        output = pipe(**common_args, **sdxl_kwargs)

        if refine == "base_image_refiner":
            refiner_kwargs = {
                "image": output.images,
            }
            if refine_steps:
                common_args["num_inference_steps"] = refine_steps

            output = self.refiner(**common_args, **refiner_kwargs)

        if not apply_watermark:
            pipe.watermark = watermark_cache
            self.refiner.watermark = watermark_cache

        if prevent_nsfw:
            _, has_nsfw_content = self.run_safety_checker(output.images)

            output_paths = []
            for i, nsfw in enumerate(has_nsfw_content):
                if nsfw:
                    print(f"NSFW content detected in image {i}")
                    continue
                output_path = f"/tmp/out-{i}.png"
                output.images[i].save(output_path)
                output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                f"NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths
