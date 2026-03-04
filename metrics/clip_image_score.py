import os
import uuid
import torch

from metrics.base_metric import BaseMetric
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
)

# Global PyTorch Inductor configurations
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


class ClipImageScore(BaseMetric):
    def __init__(self, device=None):
        self.device = device
        self.negative_prompt = """(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, (NSFW:1.25)"""
        self.width, self.height = 1024, 1024
        self.pipe = None  # Initialize pipeline variable

    def load_model(self, **kwargs):
        """
        Loads the SDXL pipeline and applies VRAM-specific optimizations.
        """
        try:
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                "fluently/Fluently-XL-Final",
                use_safetensors=True,
                low_cpu_memory_usage=False,
            )

            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

            # Detect VRAM and apply Optimization Logic
            if torch.cuda.is_available():
                total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory_bytes / (1024**3)
                print(f"Detected VRAM: {total_memory_gb:.2f} GB")

                if total_memory_gb >= 24:
                    print("High VRAM detected. Using maximum speed mode.")
                    for key in self.pipe.config.keys():
                        if hasattr(getattr(self.pipe, key), "device"):
                            getattr(self.pipe, key).to("cuda")
                            print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB after {key}")
                            torch.cuda.empty_cache()

                elif 8 <= total_memory_gb < 24:
                    print("Medium VRAM detected. Using Model CPU Offload.")
                    self.pipe.enable_model_cpu_offload()
                    self.pipe.vae.enable_tiling()

                else:  # Less than 8 GB
                    print("Low VRAM detected (<8GB). Using Sequential CPU Offload.")
                    self.pipe.enable_sequential_cpu_offload()
                    self.pipe.vae.enable_tiling()

            else:
                print("No NVIDIA GPU detected. Running on CPU (Warning: This will be very slow).")
                self.pipe.to("cpu")

        except Exception as e:
            print(f"Error initializing pipeline: {e}")
            raise e  # Raise the exception instead of just calling exit()

    def compute_score(self, ims_cs, gen_cs, **kwargs):
        """
        :param ims_cs: list of image paths (not currently used in generation loop)
        :param gen_cs: list of candidate captions
        :return: dictionary with score
        """
        # Safety check: ensure model is loaded before computing
        if self.pipe is None:
            self.load_model()

        for img_path, cand_caption in zip(ims_cs, gen_cs):
            torch.cuda.empty_cache()

            images = self.pipe(
                prompt=cand_caption,
                negative_prompt=self.negative_prompt,
                width=self.width,
                height=self.height,
                guidance_scale=3,
                num_inference_steps=20,
                num_images_per_prompt=1,
                cross_attention_kwargs={"scale": 0.65},
                output_type="pil",
            ).images

            self.save_image(images[0], cand_caption)

        return {"clip-image-score": 0}

    def setup(self):
        self.load_model()

    def save_image(self, img, caption):
        namespace = uuid.NAMESPACE_DNS
        unique_name = str(uuid.uuid5(namespace, caption)) + ".png"
        img.save(unique_name)
        return unique_name
