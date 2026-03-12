import os
import uuid
import torch
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from metrics.base_metric import BaseMetric
from diffusers import (
    StableDiffusionXLPipeline,
    EulerAncestralDiscreteScheduler,
)
from models import open_clip
# from models.clip import clip

from PIL import Image

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
            )

            self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)

            # Detect VRAM and apply Optimization Logic
            if torch.cuda.is_available():
                total_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                total_memory_gb = total_memory_bytes / (1024**3)
                print(f"Detected VRAM: {total_memory_gb:.2f} GB")

                if total_memory_gb >= 15:
                    print("High VRAM detected. Using maximum speed mode.")
                    for key in self.pipe.config.keys():
                        if hasattr(getattr(self.pipe, key), "device"):
                            getattr(self.pipe, key).to("cuda")
                            print(f"Max memory reserved: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB after {key}")
                            torch.cuda.empty_cache()

                elif 8 <= total_memory_gb < 15:
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
        self.model, _, self.processor = open_clip.create_model_and_transforms(
            'ViT-L-14', pretrained='laion2b_s32b_b82k', cache_dir='./checkpoints/'
        )
        self.model.to(self.device)


    def compute_score(self, ims_cs, gen_cs, **kwargs):
        """
        :param ims_cs: list of image paths (not currently used in generation loop)
        :param gen_cs: list of candidate captions
        :return: dictionary with score
        """
        # Safety check: ensure model is loaded before computing
        if self.pipe is None:
            self.load_model()
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        clip_scores = list()

        for img_path, cand_caption in zip(ims_cs, gen_cs):
            torch.cuda.empty_cache()

            original_image = Image.open(img_path)
            original_batch_tensor = self.processor(original_image).unsqueeze(0).to(self.device)
            original_image_embbed_vec = self.model.encode_image(original_batch_tensor)

            regenerated_image = self.generate_image(cand_caption)
            regenerated_batch_tensor = self.processor(regenerated_image).unsqueeze(0).to(self.device)
            generated_image_embbed_vec = self.model.encode_image(regenerated_batch_tensor)

            score = cos(generated_image_embbed_vec, original_image_embbed_vec).item()
            clip_scores.append(float(score))

        return {"clip-image-score": {
            "overall": sum(clip_scores) / len(clip_scores),
            "score_per_cap": clip_scores
        }}

    def setup(self, regenerated_image_dir="."):
        self.load_model()
        self.cache_regenerated_dir = f"{regenerated_image_dir}/data/clip-image-regenrated"

    def generate_image(self, cand_caption: str) -> Image.Image:
        namespace = uuid.NAMESPACE_DNS
        unique_name = str(uuid.uuid5(namespace, cand_caption)) + ".jpg"
        if not os.path.isdir(self.cache_regenerated_dir):
            os.makedirs(self.cache_regenerated_dir)

        file_name = f"{self.cache_regenerated_dir}/{unique_name}"
        if os.path.isfile(file_name):
            return Image.open(file_name)
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

        self.save_image(images[0], file_name)

        return images[0]

    def save_image(self, img, file_name):
        img.save(file_name)
        return file_name
