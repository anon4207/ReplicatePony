import os
import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running predictions efficient"""
        print("Loading pipeline...")
        
        # URL to your CyberRealistic Pony v8 model on HuggingFace
        model_url = "https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/CyberRealistic_Pony_v8.safetensors"
        
        # Load your custom model directly
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_url,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None
        ).to("cuda")
        
        # Enable memory optimization
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # List of available LORAs
        self.available_loras = [
            # Replace with your actual LORA names
            "lora1",
            "lora2", 
            "lora3"
        ]
        
        print("Model loaded successfully!")

    def predict(
        self,
        prompt: str = Input(description="Text prompt for image generation"),
        negative_prompt: str = Input(description="Negative prompt", default=""),
        width: int = Input(description="Width of output image", default=512, ge=256, le=1024),
        height: int = Input(description="Height of output image", default=512, ge=256, le=1024),
        num_inference_steps: int = Input(description="Number of denoising steps", default=30, ge=1, le=100),
        guidance_scale: float = Input(description="Scale for classifier-free guidance", default=7.5, ge=1, le=20),
        seed: int = Input(description="Random seed (0 for random)", default=0),
        lora_name: str = Input(
            description="LORA to use from your HuggingFace repo",
            default=None,
            choices=["None"] + ["lora1", "lora2", "lora3"],  # Replace with your actual LORAs
        ),
        lora_scale: float = Input(description="LORA effect strength", default=0.8, ge=0.1, le=1.0),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Starting generation with prompt: {prompt}")
        
        # Set seed for reproducibility
        if seed == 0:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")
        
        # Apply LORA if specified and it's not "None"
        if lora_name and lora_name != "None":
            print(f"Loading LORA: {lora_name} with scale {lora_scale}")
            lora_url = f"https://huggingface.co/YOUR_USERNAME/YOUR_REPO/resolve/main/{lora_name}.safetensors"
            self.pipe.load_lora_weights(lora_url)
            # Set the LORA scale
            self.pipe.fuse_lora(lora_scale=lora_scale)
        
        # Generate the image
        print(f"Generating image with {num_inference_steps} steps at {width}x{height}...")
        image = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]
        
        # If a LORA was loaded, unfuse it for the next prediction
        if lora_name and lora_name != "None":
            print("Unfusing LORA weights")
            self.pipe.unfuse_lora()
        
        # Save and return the image
        output_path = "output.png"
        image.save(output_path)
        print("Generation complete!")
        
        return Path(output_path)
