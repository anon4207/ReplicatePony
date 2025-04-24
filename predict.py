import os
import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running predictions efficient"""
        print("Loading pipeline...")
        
        # Load base model
        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",  # Base model - replace if needed
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")
        
        # Download model and LoRAs from your HuggingFace repo
        # You'll need to replace this with your actual HF repo details
        model_path = "path-to-your-model.safetensors"
        
        # Load your custom model
        self.pipe.unet.load_attn_procs(model_path)
        
        print("Model loaded!")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        width: int = Input(description="Width of output image", default=512),
        height: int = Input(description="Height of output image", default=512),
        num_inference_steps: int = Input(description="Number of denoising steps", default=30),
        guidance_scale: float = Input(description="Scale for classifier-free guidance", default=7.5),
        seed: int = Input(description="Random seed", default=None),
        lora_file: str = Input(
            description="LoRA to use (file from your HuggingFace repo)",
            default=None,
        ),
        lora_scale: float = Input(description="LoRA scale", default=0.8),
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        
        # Load LoRA if specified
        if lora_file:
            self.pipe.unet.load_attn_procs(lora_file, weight_name="pytorch_lora_weights.safetensors")
        
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        output_path = "output.png"
        image.save(output_path)
        
        return Path(output_path)
