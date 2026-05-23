import os
import time
import uuid

import torch
from diffusers import (
    EulerAncestralDiscreteScheduler,
    StableDiffusionXLPipeline,
)
from PIL import Image

from metrics.base_metric import BaseMetric
from models import clip


# Global PyTorch Inductor configurations
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


class ClipImageScore(BaseMetric):
    METRIC_NAME = "CLIP-Image-Score"

    def __init__(self, device=None):
        self.device = device
        self.pipe = None
        self.model = None
        self.processor = None

        self.width = 1024
        self.height = 1024

        self.cache_regenerated_dir = None

        self.negative_prompt = (
            "(deformed, distorted, disfigured:1.3), poorly drawn, "
            "bad anatomy, wrong anatomy, extra limb, missing limb, "
            "floating limbs, (mutated hands and fingers:1.4), "
            "disconnected limbs, mutation, mutated, ugly, disgusting, "
            "blurry, amputation, (NSFW:1.25)"
        )

    @property
    def requires_references(self) -> bool:
        return False

    def setup(self, regenerated_image_dir="."):
        self.cache_regenerated_dir = (
            f"{regenerated_image_dir}/data/clip-image-regenerated"
        )
        self.load_model()

    def load_model(self, **kwargs):
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "fluently/Fluently-XL-Final",
            use_safetensors=True,
        )

        self.pipe.scheduler = (
            EulerAncestralDiscreteScheduler.from_config(
                self.pipe.scheduler.config
            )
        )

        self._configure_pipeline_memory()

        self.model, self.processor = clip.load(
            "ViT-B/32",
            device=self.device,
            download_root="./checkpoints/",
        )

    def compute_score(
            self,
            ims_cs,
            gen_cs,
            **kwargs,
    ):
        if self.pipe is None or self.model is None:
            raise RuntimeError(
                "ClipImageScore model not initialized. Call setup() first."
            )

        cosine_similarity = torch.nn.CosineSimilarity(
            dim=1,
            eps=1e-6,
        )

        clip_scores = []

        start_time = time.perf_counter()

        for img_path, cand_caption in zip(ims_cs, gen_cs):
            torch.cuda.empty_cache()

            original_embedding = self._encode_image(
                Image.open(img_path)
            )

            regenerated_image = self.generate_image(cand_caption)

            regenerated_embedding = self._encode_image(
                regenerated_image
            )

            score = cosine_similarity(
                regenerated_embedding,
                original_embedding,
            ).item()

            clip_scores.append(float(score))

        elapsed_seconds = time.perf_counter() - start_time

        return {
            self.METRIC_NAME: {
                "overall": sum(clip_scores) / len(clip_scores),
                "score_per_cap": clip_scores,
                "time": elapsed_seconds,
            }
        }

    def _encode_image(self, image: Image.Image):
        batch_tensor = (
            self.processor(image)
            .unsqueeze(0)
            .to(self.device)
        )

        return self.model.encode_image(batch_tensor)

    def _configure_pipeline_memory(self):
        if not torch.cuda.is_available():
            print(
                "No NVIDIA GPU detected. Running on CPU "
                "(Warning: this will be very slow)."
            )
            self.pipe.to("cpu")
            return

        total_memory_bytes = torch.cuda.get_device_properties(
            0
        ).total_memory
        total_memory_gb = total_memory_bytes / (1024 ** 3)

        print(f"Detected VRAM: {total_memory_gb:.2f} GB")

        if total_memory_gb >= 15:
            print("High VRAM detected. Using maximum speed mode.")

            for key in self.pipe.config.keys():
                module = getattr(self.pipe, key, None)

                if hasattr(module, "device"):
                    module.to("cuda")
                    torch.cuda.empty_cache()

        elif total_memory_gb >= 8:
            print("Medium VRAM detected. Using model CPU offload.")
            self.pipe.enable_model_cpu_offload()
            self.pipe.vae.enable_tiling()

        else:
            print("Low VRAM detected. Using sequential CPU offload.")
            self.pipe.enable_sequential_cpu_offload()
            self.pipe.vae.enable_tiling()

    def generate_image(self, cand_caption: str) -> Image.Image:
        namespace = uuid.NAMESPACE_DNS
        unique_name = str(
            uuid.uuid5(namespace, cand_caption)
        ) + ".jpg"

        os.makedirs(
            self.cache_regenerated_dir,
            exist_ok=True,
        )

        file_name = (
            f"{self.cache_regenerated_dir}/{unique_name}"
        )

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

        images[0].save(file_name)

        return images[0]