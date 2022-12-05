# Based on https://github.com/huggingface/diffusers/blob/8b84f8519264942fa0e52444881390767cb766c5/examples/dreambooth/train_dreambooth.py

# Reasons for not using that file directly:
#
#   1) Use our already loded model from `init()`
#   2) Callback to run after every iteration

# Deps

import argparse
import hashlib
import itertools
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# DDA
from precision import revision, torch_dtype
from send import send, get_now
from utils import Storage
import subprocess
import re
import shutil

# Our original code in docker-diffusers-api:

HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")


params = {
    # Defaults
    "pretrained_model_name_or_path": "dreambooth",  # DDA, TODO
    "revision": revision,  # DDA, was: None
    "tokenizer_name": None,
    "instance_data_dir": "instance_data_dir",  # DDA TODO
    "class_data_dir": "class_data_dir",  # DDA, was: None,
    # instance_prompt
    "class_prompt": None,
    "with_prior_preservation": False,
    "prior_loss_weight": 1.0,
    "num_class_images": 100,
    "output_dir": "text-inversion-model",
    "seed": None,
    "resolution": 512,
    "center_crop": None,
    "train_text_encoder": None,
    "train_batch_size": 1,  # DDA, was: 4
    "sample_batch_size": 1,  # DDA, was: 4,
    "num_train_epochs": 1,
    "max_train_steps": 800,  # DDA, was: None,
    "gradient_accumulation_steps": 1,
    "gradient_checkpointing": True,  # DDA was: None (needed for 16GB)
    "learning_rate": 5e-6,
    "scale_lr": False,
    "lr_scheduler": "constant",
    "lr_warmup_steps": 0,  # DDA, was: 500,
    "use_8bit_adam": True,  # DDA, was: None (needed for 16GB)
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_weight_decay": 1e-6,
    "adam_epsilon": 1e-08,
    "max_grad_norm": 1.0,
    "push_to_hub": None,
    "hub_token": HF_AUTH_TOKEN,
    "hub_model_id": None,
    "logging_dir": "logs",
    "mixed_precision": None if revision == "" else revision,  # DDA, was: None
    "local_rank": -1,
}

dest_url = "s3://banana-docker-diffusers.s3.us-west-1.amazonaws.com/test/"
if dest_url:
    storage = Storage(dest_url)
    # fp16 model timings: zip 1m20s, tar+zstd 4s and a tiny bit smaller!
    compress_start = get_now()

    print("UPLOAD START...")
    upload_result = storage.upload_file("test.py", "test.py")
    print(upload_result)

    result.get("$timings").update({"upload": upload_result["$time"]})

