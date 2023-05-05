import os
import gc
import json
import subprocess
import mimetypes
import shutil
from zipfile import ZipFile
from typing import List
from argparse import Namespace
import time
import torch

from cog import BasePredictor, Input, Path
from functools import lru_cache

from dreambooth import main


class Predictor(BasePredictor):
    def setup(self):
        assert torch.cuda.is_available()

    @lru_cache(maxsize=10)
    def get_weights(self, weights: str):
        destination_path = self.weights_path(weights)
        if not os.path.exists(destination_path):
            self.download_weights(weights)

        return destination_path

    def weights_path(self, weights: str):
        if not os.path.exists("/src/weights"):
            os.makedirs("/src/weights")
        return os.path.join("/src/weights", weights)

    def download_weights(self, weights: str):
        print(f"Downloading weights for {weights}...")

        url = f"https://storage.googleapis.com/<bucket>/{weights}.tar"

        dest = self.weights_path(weights)
        output = subprocess.check_output(
            ['/src/pgettar', url, dest, str(16)])

    def predict(
        self,
        instance_prompt: str = Input(
            description="The prompt you use to describe your training images, in the format: `a [identifier] [class noun]`, where the `[identifier]` should be a rare token. Relatively short sequences with 1-3 letters work the best (e.g. `sks`, `xjy`). `[class noun]` is a coarse class descriptor of the subject (e.g. cat, dog, watch, etc.). For example, your `instance_prompt` can be: `a sks dog`, or with some extra description `a photo of a sks dog`. The trained model will learn to bind a unique identifier with your specific subject in the `instance_data`.",
        ),
        class_prompt: str = Input(
            description="The prompt or description of the coarse class of your training images, in the format of `a [class noun]`, optionally with some extra description. `class_prompt` is used to alleviate overfitting to your customised images (the trained model should still keep the learnt prior so that it can still generate different dogs when the `[identifier]` is not in the prompt). Corresponding to the examples of the `instant_prompt` above, the `class_prompt` can be `a dog` or `a photo of a dog`.",
        ),
        with_prior_preservation: bool = Input(
            description="Flag to add prior preservation loss.",
            default=True,
        ),
        prior_loss_weight: float = Input(
            description="The weight of prior preservation loss.",
            default=1.0,
        ),
        instance_data: Path = Input(
            description="A ZIP file containing your training images (JPG, PNG, etc. size not restricted). These images contain your 'subject' that you want the trained model to embed in the output domain for later generating customized scenes beyond the training images. For best results, use images without noise or unrelated objects in the background.",
        ),
        class_data: Path = Input(
            description="An optional ZIP file containing the training data of class images. This corresponds to `class_prompt` above, also with the purpose of keeping the model generalizable. By default, the pretrained stable-diffusion model will generate N images (determined by the `num_class_images` you set) based on the `class_prompt` provided. But to save time or to have your preferred specific set of `class_data`, you can also provide them in a ZIP file.",
            default=None,
        ),
        num_class_images: int = Input(
            description="Minimal class images for prior preservation loss. If not enough images are provided in class_data, additional images will be"
            " sampled with class_prompt.",
            default=50,
        ),
        seed: int = Input(
            description="A seed for reproducible training.", default=1337
        ),
        resolution: int = Input(
            description="The resolution for input images. All the images in the train/validation dataset will be resized to this"
            " resolution.",
            default=768,
            choices=[512, 768],
        ),
        center_crop: bool = Input(
            description="Whether to center crop the input images to the resolution. If not set, the images will be randomly"
            " cropped. The images will be resized to the resolution first before cropping.",
            default=False,
        ),
        train_text_encoder: bool = Input(
            description="Whether to train the text encoder. If set, the text encoder should be float32 precision.",
            default=True,
        ),
        train_batch_size: int = Input(
            description="Batch size (per device) for the training dataloader.",
            default=1,
        ),
        sample_batch_size: int = Input(
            description="Batch size (per device) for sampling images.",
            default=4,
        ),
        num_train_epochs: int = Input(default=1),
        max_train_steps: int = Input(
            description="Total number of training steps to perform. If provided, overrides num_train_epochs.",
            default=2000,
        ),
        gradient_accumulation_steps: int = Input(
            description="Number of updates steps to accumulate before performing a backward/update pass.",
            default=1,
        ),
        gradient_checkpointing: bool = Input(
            description="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
            default=False,
        ),
        learning_rate: float = Input(
            description="Initial learning rate (after the potential warmup period) to use.",
            default=5e-6,
        ),
        scale_lr: bool = Input(
            description="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
            default=False,
        ),
        lr_scheduler: str = Input(
            description="The scheduler type to use",
            choices=[
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ],
            default="constant",
        ),
        lr_warmup_steps: int = Input(
            description="Number of steps for the warmup in the lr scheduler.",
            default=0,
        ),
        lr_num_cycles: int = Input(
            description="Number of hard resets of the lr in cosine_with_restarts scheduler.",
            default=1,
        ),
        lr_power: float = Input(
            description="Power factor of the polynomial scheduler.",
            default=1.0,
        ),
        use_8bit_adam: bool = Input(
            description="Whether or not to use 8-bit Adam from bitsandbytes.",
            default=False,
        ),
        adam_beta1: float = Input(
            default=0.9,
            description="The beta1 parameter for the Adam optimizer.",
        ),
        adam_beta2: float = Input(
            default=0.999,
            description="The beta2 parameter for the Adam optimizer.",
        ),
        adam_weight_decay: float = Input(
            default=1e-2,
            description="Weight decay to use",
        ),
        adam_epsilon: float = Input(
            default=1e-08,
            description="Epsilon value for the Adam optimizer",
        ),
        max_grad_norm: float = Input(
            default=1.0,
            description="Max gradient norm.",
        ),
        # Replicate add-ons
        weights: str = Input(
            description="Which weights to use",
        ),
        hflip: bool = Input(
            description="Whether or not to horizontally flip training images 50 percent of the time.",
            default=False,
        ),
        generate_images: str = Input(
            description='json of samples to generate: [{"name": "sample_name", "input": {"prompt": "a sks dog", "num_samples": 4, "save_guidance_scale": 7.5, "save_infer_steps": 50}}]',
            default=None
        ),
        info: bool = Input(
            description="log extra information about the run", default=False
        ),
    ) -> List[Path]:

        if info:
            os.system("nvidia-smi")
            os.system("df -h")
            os.system("free -h")
            print(self.get_weights.cache_info())

        start = time.time()
        pretrained_model = self.get_weights(weights)
        print("loading weights took: %0.2f" % (time.time() - start))

        cog_instance_data = "cog_instance_data"
        cog_class_data = "cog_class_data"
        cog_output_dir = "checkpoints"
        cog_generated_images = "cog_generated_images"

        for path in [
            cog_instance_data,
            cog_class_data,
            cog_output_dir,
            cog_generated_images,
        ]:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)

        # extract zip contents, flattening any paths present within it
        with ZipFile(str(instance_data), "r") as zip_ref:
            for zip_info in zip_ref.infolist():
                if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                    "__MACOSX"
                ):
                    continue
                mt = mimetypes.guess_type(zip_info.filename)
                if mt and mt[0] and mt[0].startswith("image/"):
                    zip_info.filename = os.path.basename(zip_info.filename)
                    zip_ref.extract(zip_info, cog_instance_data)

        if class_data is not None:
            with ZipFile(str(class_data), "r") as zip_ref:
                for zip_info in zip_ref.infolist():
                    if zip_info.filename[-1] == "/" or zip_info.filename.startswith(
                        "__MACOSX"
                    ):
                        continue
                    mt = mimetypes.guess_type(zip_info.filename)
                    if mt and mt[0] and mt[0].startswith("image/"):
                        zip_info.filename = os.path.basename(zip_info.filename)
                        zip_ref.extract(zip_info, cog_class_data)

        if generate_images is not None:
            generate_images = json.loads(generate_images)

        # some settings are fixed for the replicate model
        args = {
            "pretrained_model_name_or_path": pretrained_model,
            "revision": "fp16",
            "tokenizer_name": None,
            "instance_data_dir": cog_instance_data,
            "class_data_dir": cog_class_data,
            "instance_prompt": instance_prompt,
            "class_prompt": class_prompt,
            "with_prior_preservation": with_prior_preservation,
            "prior_loss_weight": prior_loss_weight,
            "num_class_images": num_class_images,
            "output_dir": cog_output_dir,
            "seed": seed,
            "resolution": resolution,
            "center_crop": center_crop,
            "train_text_encoder": train_text_encoder,
            "train_batch_size": train_batch_size,
            "sample_batch_size": sample_batch_size,
            "num_train_epochs": num_train_epochs,
            "max_train_steps": max_train_steps,
            "checkpointing_steps": 500,  # not used
            "checkpoints_total_limit": None,
            "resume_from_checkpoint": None,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "gradient_checkpointing": gradient_checkpointing,
            "learning_rate": learning_rate,
            "scale_lr": scale_lr,
            "lr_scheduler": lr_scheduler,
            "lr_warmup_steps": lr_warmup_steps,
            "lr_num_cycles": lr_num_cycles,
            "lr_power": lr_power,
            "use_8bit_adam": use_8bit_adam,
            "dataloader_num_workers": 0,  # not used
            "adam_beta1": adam_beta1,
            "adam_beta2": adam_beta2,
            "adam_weight_decay": adam_weight_decay,
            "adam_epsilon": adam_epsilon,
            "max_grad_norm": max_grad_norm,
            "push_to_hub": False,
            "hub_token": None,
            "hub_model_id": None,
            "logging_dir": "logs",
            "allow_tf32": True,
            "report_to": "all",
            "validation_prompt": "",  # not used
            "num_validation_images": 4,  # not used
            "validation_steps": 100,  # not used
            "mixed_precision": "fp16",
            "prior_generation_precision": "fp16",
            "local_rank": -1,
            "enable_xformers_memory_efficient_attention": True,
            "set_grads_to_none": True,
            "offset_noise": True,
            "hflip": hflip,
            "generate_images": generate_images,
            "generate_images_dir": cog_generated_images,
        }

        args = Namespace(**args)

        main(args)

        gc.collect()
        torch.cuda.empty_cache()
        os.system("nvidia-smi")

        results = []

        weights_path = "output.zip"

        directory = Path(cog_output_dir)
        with ZipFile(weights_path, "w") as zip:
            for file_path in directory.rglob("*"):
                print(file_path)
                zip.write(file_path, arcname=file_path.relative_to(directory))

        results.append(Path(weights_path))

        directory = Path(cog_generated_images)
        for file_path in directory.rglob("*"):
            print(file_path)
            results.append(file_path)

        return results
