import os
import json

import warnings

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from conversation import conv_templates, SeparatorStyle

from transformers import AutoProcessor, PaliGemmaForConditionalGeneration

from llama_cpp import Llama

from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, BitsAndBytesConfig
import bitsandbytes, flash_attn

from PIL import Image
import requests
import torch
import copy
import time
import math
import gc

import torch
from torch.autograd import profiler

from flask import Flask, render_template, request, jsonify