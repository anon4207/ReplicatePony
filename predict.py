import os
import torch
from diffusers import StableDiffusionPipeline
from cog import BasePredictor, Input, Path
import requests
from io import BytesIO

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running predictions efficient"""
        print("Loading pipeline...")
        
        # URL to your CyberRealistic Pony v8 model on HuggingFace
        model_url = "https://huggingface.co/tomparisbiz/CyberRachel/resolve/main/cyberrealisticPony_v8.safetensors"
        
        # Load your custom model directly
        self.pipe = StableDiffusionPipeline.from_single_file(
            model_url,
            torch_dtype=torch.float16,
            use_safetensors=True,
            safety_checker=None
        ).to("cuda")
        
        # Enable memory optimization
        self.pipe.enable_xformers_memory_efficient_attention()
        
        # Create a directory for temporary LoRA files
        os.makedirs("temp_loras", exist_ok=True)
        
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
        lora_urls: str = Input(
            description="URLs to LoRA safetensors files, separated by commas (up to 8)",
            default="",
        ),
        lora_scales: str = Input(
            description="LoRA effect strengths, separated by commas (matching the order of lora_urls)",
            default="0.8",
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        print(f"Starting generation with prompt: {prompt}")
        
        # Set seed for reproducibility
        if seed == 0:
            seed = int.from_bytes(os.urandom(4), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")
        
        # Parse LoRA URLs and scales
        lora_url_list = [url.strip() for url in lora_urls.split(',') if url.strip()] if lora_urls else []
        lora_scale_list = [float(scale.strip()) for scale in lora_scales.split(',') if scale.strip()] if lora_scales else []
        
        # Ensure we have scales for all LoRAs, default to 0.8 if missing
        while len(lora_scale_list) < len(lora_url_list):
            lora_scale_list.append(0.8)
        
        # Limit to 8 LoRAs max
        lora_url_list = lora_url_list[:8]
        lora_scale_list = lora_scale_list[:len(lora_url_list)]
        
        temp_lora_paths = []
        
        # Apply LoRAs if URLs are provided
        if lora_url_list:
            try:
                # Download and load each LoRA
                for i, (lora_url, lora_scale) in enumerate(zip(lora_url_list, lora_scale_list)):
                    print(f"Loading LoRA {i+1}/{len(lora_url_list)}: {lora_url} with scale {lora_scale}")
                    
                    # Download the LoRA file to a temporary location
                    lora_path = f"temp_loras/lora_{i}.safetensors"
                    temp_lora_paths.append(lora_path)
                    
                    if lora_url.startswith(("http://", "https://")):
                        response = requests.get(lora_url)
                        response.raise_for_status()
                        with open(lora_path, "wb") as f:
                            f.write(response.content)
                        
                        # Load the LoRA weights (but don't fuse yet)
                        self.pipe.load_lora_weights(
                            lora_path, 
                            adapter_name=f"lora_{i}"
                        )
                
                # Set the weights for all loaded LoRAs
                for i, lora_scale in enumerate(lora_scale_list):
                    self.pipe.set_adapters(
                        [f"lora_{i}"], 
                        adapter_weights=[lora_scale]
                    )
                
                print(f"Successfully loaded {len(lora_url_list)} LoRAs")
            except Exception as e:
                print(f"Error loading LoRAs: {e}")
                # Try to unload any LoRAs that might have been loaded
                try:
                    self.pipe.unload_lora_weights()
                except:
                    pass
        
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
        
        # If LoRAs were loaded, unload them
        if lora_url_list:
            try:
                print("Unloading LoRA weights")
                self.pipe.unload_lora_weights()
                
                # Clean up temporary files
                for path in temp_lora_paths:
                    if os.path.exists(path):
                        os.remove(path)
            except Exception as e:
                print(f"Error unloading LoRAs: {e}")
        
        # Save and return the image
        output_path = "output.png"
        image.save(output_path)
        print("Generation complete!")
        
        return Path(output_path)
