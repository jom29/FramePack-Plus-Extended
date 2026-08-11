import gc
from diffusers_helper.hf_login import login

import os
import json


# ==========================================================
# LOCAL HUGGING FACE CACHE
# ==========================================================

HF_CACHE = os.path.abspath(
    os.path.realpath(
        os.path.join(
            os.path.dirname(__file__),
            './hf_download'
        )
    )
)

HF_HUB_CACHE = os.path.join(
    HF_CACHE,
    'hub'
)

os.environ['HF_HOME'] = HF_CACHE
os.environ['HF_HUB_CACHE'] = HF_HUB_CACHE
os.environ['HUGGINGFACE_HUB_CACHE'] = HF_HUB_CACHE

# IMPORTANT:
# Never attempt an internet download during runtime.
os.environ['HF_HUB_OFFLINE'] = '1'

print()
print("========================================")
print("LOCAL HUGGING FACE CACHE")
print("========================================")
print("HF_HOME      :", os.environ['HF_HOME'])
print("HF_HUB_CACHE :", os.environ['HF_HUB_CACHE'])
print("HF_HUB_OFFLINE : 1")
print("========================================")




import gradio as gr
import torch
import traceback
import einops
import safetensors.torch as sf
import numpy as np
import argparse
import math
import time

from PIL import Image
from PIL.PngImagePlugin import PngInfo
from diffusers import AutoencoderKLHunyuanVideo
from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, state_dict_weighted_merge, state_dict_offset_merge, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.thread_utils import AsyncStream, async_run
from diffusers_helper.gradio.progress_bar import make_progress_bar_css, make_progress_bar_html
from transformers import SiglipImageProcessor, SiglipVisionModel
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket
from typing import Tuple, Any
from utils.lora_utils import merge_lora_to_state_dict
from utils.fp8_optimization_utils import optimize_state_dict_with_fp8, apply_fp8_monkey_patch

parser = argparse.ArgumentParser()
parser.add_argument('--share', action='store_true')
parser.add_argument("--server", type=str, default='0.0.0.0')
parser.add_argument("--port", type=int, required=False)
parser.add_argument("--inbrowser", action='store_true')
args = parser.parse_args()

# for win desktop probably use --server 127.0.0.1 --inbrowser
# For linux server probably use --server 127.0.0.1 or do not use any cmd flags

print(args)

free_mem_gb = get_cuda_free_memory_gb(gpu)
high_vram = free_mem_gb > 60

print(f'Free VRAM {free_mem_gb} GB')
print(f'High-VRAM Mode: {high_vram}')

text_encoder = LlamaModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder', torch_dtype=torch.float16).cpu()
text_encoder_2 = CLIPTextModel.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='text_encoder_2', torch_dtype=torch.float16).cpu()
tokenizer = LlamaTokenizerFast.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer')
tokenizer_2 = CLIPTokenizer.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='tokenizer_2')
vae = AutoencoderKLHunyuanVideo.from_pretrained("hunyuanvideo-community/HunyuanVideo", subfolder='vae', torch_dtype=torch.float16).cpu()

feature_extractor = SiglipImageProcessor.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='feature_extractor')
image_encoder = SiglipVisionModel.from_pretrained("lllyasviel/flux_redux_bfl", subfolder='image_encoder', torch_dtype=torch.float16).cpu()

def load_transfomer():
    print("Loading transformer ...")
    #transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(f"{os.environ['HF_HOME']}/hub/models--lllyasviel--FramePackI2V_HY/snapshots/86cef4396041b6002c957852daac4c91aaa47c79", torch_dtype=torch.bfloat16).cpu()
    transformer.eval()
    transformer.high_quality_fp32_output_for_inference = True
    print("transformer.high_quality_fp32_output_for_inference = True")

    transformer.to(dtype=torch.bfloat16)
    transformer.requires_grad_(False)
    return transformer


transformer = None  # load later
transformer_dtype = torch.bfloat16
previous_lora_file = None
previous_lora_multiplier = None
previous_fp8_optimization = None

vae.eval()
text_encoder.eval()
text_encoder_2.eval()
image_encoder.eval()

if not high_vram:
    vae.enable_slicing()
    vae.enable_tiling()

vae.to(dtype=torch.float16)
image_encoder.to(dtype=torch.float16)
text_encoder.to(dtype=torch.float16)
text_encoder_2.to(dtype=torch.float16)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
text_encoder_2.requires_grad_(False)
image_encoder.requires_grad_(False)

if not high_vram:
    # DynamicSwapInstaller is same as huggingface's enable_sequential_offload but 3x faster
    #DynamicSwapInstaller.install_model(transformer, device=gpu)
    DynamicSwapInstaller.install_model(text_encoder, device=gpu)
else:
    text_encoder.to(gpu)
    text_encoder_2.to(gpu)
    image_encoder.to(gpu)
    vae.to(gpu)
    # transformer.to(gpu)

stream = AsyncStream()

outputs_folder = './outputs/'
os.makedirs(outputs_folder, exist_ok=True)


def render_pipeline():

    print()

    print("========================================")
    print("Render Pipeline")
    print("========================================")

    print()

    print("Pipeline Entry")


# ==========================================================
# M6 : Runtime State
# ==========================================================

class RuntimeState:

    def __init__(self):

        # Pipeline Collections
        self.frames = []
        self.processed_frames = []
        self.tensors = []
        self.latents = []
        self.segment_collection = []




        # Runtime Components
        self.conditioning = {}
        self.resources = {}
        self.timeline = {}
        self.compatibility = {}



      # ==========================================================
      # M7 Experimental Runtime
      # ==========================================================

        self.experimental = {

          "latent_collection": None,
 
          "start_index": 0,

         "end_index": 1

          }


# ==========================================================
# Segment Runtime
# One FramePack generation session
# ==========================================================

class SegmentRuntime:

    def __init__(self):

        self.segment_index = 0

        self.start_index = 0
        self.end_index = 1

        self.start_frame = None
        self.end_frame = None

        self.start_tensor = None
        self.end_tensor = None

        self.start_latent = None
        self.end_latent = None

        # ============================
        # Runtime State
        # ============================

        self.generated_latents = None

        self.history_latents = None
        self.history_pixels = None

        self.rolling_anchor_latent = None

        self.total_generated_latent_frames = 0

        self.clean_latent_indices = None

        self.history_before = None
        self.history_after = None






# ==========================================================
# M8.1 : Timeline Segment
# ==========================================================

class TimelineSegment:

    def __init__(
        self,
        segment_index,
        start_keyframe,
        end_keyframe,
        first_section=None,
        last_section=None
    ):

        self.segment_index = segment_index

        self.start_keyframe = start_keyframe
        self.end_keyframe = end_keyframe

        self.first_section = first_section
        self.last_section = last_section

        # ==========================================================
        # M12 : Planner Duration
        # ==========================================================

        self.duration_seconds = 0.0
        self.scheduler_sections = 0



# ==========================================================
# M8.1 : Timeline Planner
# ==========================================================

class TimelinePlanner:

    def __init__(
        self,
        total_keyframes
    ):

        self.total_keyframes = total_keyframes

        # Timeline Graph
        self.segments = []

        # Section Ownership Table
        self.section_ownership = []

        # Build timeline graph only
        self.build_segments()

    # ==========================================================
    # M8.1
    # Timeline Graph
    # ==========================================================

    def build_segments(self):

        if self.total_keyframes < 2:
            return

        total_segments = self.total_keyframes - 1

        for segment_index in range(total_segments):

            segment = TimelineSegment(

                segment_index=segment_index,

                start_keyframe=segment_index,

                end_keyframe=segment_index + 1,

                first_section=None,

                last_section=None

            )

            self.segments.append(segment)

    # ==========================================================
    # M8.2
    # Section Ownership
    # ==========================================================

    def build_section_ownership(
        self,
        total_sections
    ):

        self.section_ownership.clear()

        if len(self.segments) == 0:
            return

        total_segments = len(self.segments)

        sections_per_segment = max(
            1,
            total_sections // total_segments
        )

        current_segment = 0

        for section in range(total_sections):

            self.section_ownership.append(current_segment)

            if (
                (section + 1) % sections_per_segment == 0
                and current_segment < total_segments - 1
            ):
                current_segment += 1


        # ==========================================================
        # M12 : Count Scheduler Sections
        # ==========================================================

        for segment in self.segments:
         segment.scheduler_sections = (self.section_ownership.count(segment.segment_index))





    def get_active_segment(
        self,
        section_index
    ):

        if len(self.section_ownership) == 0:
            return None

        if section_index >= len(self.section_ownership):
            section_index = len(self.section_ownership) - 1

        segment_index = self.section_ownership[section_index]

        return self.segments[segment_index]

    def print_section_ownership(self):

      print()

      print("========================================")
      print("M8.2 : Section Ownership")
      print("========================================")
      print()

      print("Total Keyframes :", self.total_keyframes)
      print("Total Segments  :", len(self.segments))
      print("Total Sections  :", len(self.section_ownership))
      print()

      print("----------------------------------------")
      print("Timeline Graph")
      print("----------------------------------------")

      for segment in self.segments:

        print(
            f"Segment {segment.segment_index}"
        )

        print(
            f"Anchor Pair : "
            f"{segment.start_keyframe} -> {segment.end_keyframe}"
        )

        print()

      print("----------------------------------------")
      print("Ownership Table")
      print("----------------------------------------")

      for section_index, segment_index in enumerate(self.section_ownership):

        segment = self.segments[segment_index]

        print(

            f"Section {section_index:02d}"

            f"  -->  "

            f"Segment {segment.segment_index}"

            f"  ({segment.start_keyframe}"

            f" -> "

            f"{segment.end_keyframe})"

        )

      print()

      print("========================================")
      print("M8.2 Completed")
      print("========================================")
      print()










@torch.no_grad()
def worker(
    frame_collection,
    prompt,
    n_prompt,
    prompt_queue,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    resolution,
    teacache_threshold,
    lora_file,
    lora_multiplier,
    fp8_optimization
   ):

    # ==========================================================
    # M6 : Runtime State
    # ==========================================================

    runtime = RuntimeState()
    prompt_queue = json.loads(prompt_queue)
  

    if not isinstance(prompt_queue, list):
      raise ValueError("Prompt queue must be a list.")

    if len(prompt_queue) == 0:
       raise ValueError("Prompt queue is empty.")

    print()
    print("========================================")
    print("PROMPT QUEUE RECEIVED")
    print("========================================")
    print("Prompt Count :", len(prompt_queue))
    print("========================================")

    
    
    print()
    print("RUNNING FILE:")
    print(__file__)



    print()

    print("========================================")
    print("Incoming API Payload")
    print("========================================")

    print()

    print("Input Image :", input_image is not None)

    print("End Image   :", end_image is not None)







   # --------------------------------------------------
   # M5.2 : Runtime Frame Collection
   # --------------------------------------------------

    from PIL import Image
    import numpy as np

    runtime.frames.clear()

    print()
    print("========================================")
    print("W6.4.6 : RUNTIME RECEIVED FRAME ORDER")
    print("========================================")

    for index, frame in enumerate(frame_collection):

        print("Frame Type :", type(frame))

        image_path = frame["path"]

        print(f"Runtime Frame {index} : {os.path.basename(image_path)}")
        print(f"Runtime Path {index}  : {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = np.array(image)

        runtime.frames.append(image)

    print("----------------------------------------")

    for index, frame in enumerate(runtime.frames):
      print(f"runtime.frames[{index}] : shape={frame.shape}")

    print("========================================")



    global transformer, previous_lora_file, previous_lora_multiplier, previous_fp8_optimization
    start_time = time.time()
    
    render_pipeline()



    # ==========================================================
    # M5.3 : Processed Frame Collection
    # ==========================================================
    # Goal:
    #   Convert every frame inside frame_collection into a
    #   normalized image using the same preprocessing pipeline.
    #
    # This does NOT replace FramePack's original variables yet.
    # It is only building a new collection for verification.
    # ==========================================================
    
    processed_frame_collection = []
    
    for frame in runtime.frames:
    
        # Original frame size
          H, W, C = frame.shape
    
          # Find the nearest bucket resolution
          height, width = find_nearest_bucket(
          H,
          W,
          resolution=resolution
          )
    
          # Resize and crop the frame
          processed = resize_and_center_crop(
            frame,
            target_width=width,
            target_height=height
            )
    
          # Store the processed frame
          processed_frame_collection.append(processed)
    



     # ==========================================================
     # M5.4 : Tensor Collection
     # ==========================================================
     #
     # Goal:
     # Convert every processed frame into the tensor format
     # expected by the FramePack runtime.
     #
     # This does NOT replace FramePack's original tensor
     # variables yet. It only builds a new tensor collection
     # for verification.
     #
     # ==========================================================
      
    tensor_collection = []
      
    for processed in processed_frame_collection:
      
      # ------------------------------------------------------
      # Convert NumPy image -> Float Tensor
      # ------------------------------------------------------
        tensor = torch.from_numpy(processed).float()
      
             # ------------------------------------------------------
             # Normalize pixel values
             # Range:
             #     0   -> -1
             #     255 ->  1
             # ------------------------------------------------------
        tensor = tensor / 127.5 - 1
      
               # ------------------------------------------------------
               # Rearrange dimensions
               #
               # From:
               #     (H, W, C)
               #
               # To:
               #     (1, C, H, W, 1)
               # ------------------------------------------------------
        tensor = tensor.permute(2, 0, 1)[None, :, None]
      
               # ------------------------------------------------------
               # Store tensor
               # ------------------------------------------------------
        tensor_collection.append(tensor)
      
      
  
   


    model_changed = transformer is None or (
        lora_file != previous_lora_file
        or lora_multiplier != previous_lora_multiplier
        or fp8_optimization != previous_fp8_optimization
    )




    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    






    # ==========================================================
    # M8.1 : Timeline Planner
    # ==========================================================

    planner = TimelinePlanner(
      total_keyframes=len(runtime.frames)
    )



    planner.build_section_ownership(total_latent_sections)

    # ==========================================================
    # M12 : Distribute User Duration
    # ==========================================================

    total_segments = max(len(planner.segments), 1)

    if len(prompt_queue) < total_segments:
       raise ValueError(
        f"Prompt queue contains {len(prompt_queue)} prompts, "
        f"but {total_segments} segments are required."
      )

    segment_duration = (total_second_length / total_segments)

    

    for segment in planner.segments:
      segment.duration_seconds = segment_duration

    print()
    print("========================================")
    print("M12 : Segment Duration")
    print("========================================")

    for segment in planner.segments:
      print(f"Segment {segment.segment_index}"f" : {segment.duration_seconds:.2f}s")

      print("========================================")

    planner.print_section_ownership()




    for segment in planner.segments:

      print(f"Segment {segment.segment_index}")

      print(
        f"Anchor Pair : {segment.start_keyframe} -> {segment.end_keyframe}"
      )

      print()



    # ==========================================================
    # M8.2
    # Ownership Lookup
    # ==========================================================

    def get_active_segment(
      self,
      scheduler_window
     ):

       if len(self.section_ownership) == 0:
         return None

       if scheduler_window >= len(self.section_ownership):
        scheduler_window = len(self.section_ownership) - 1

        segment_index = self.section_ownership[scheduler_window]

       return self.segments[segment_index]





    job_id = generate_timestamp()
    job_id = f'{job_id}_{resolution}_{seed}{"_teacache" if use_teacache else ""}'

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )


   
        # ==========================================================
        # Text Encoder Preparation
        # Prompt encoding is now performed per segment.
        # ==========================================================

        if not high_vram:
           fake_diffusers_current_device(text_encoder, gpu)
           load_model_as_complete(text_encoder_2, target_device=gpu)


          # Processing input image (start frame)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))



        start_frame = runtime.frames[0]

        H, W, C = start_frame.shape

        height, width = find_nearest_bucket(
        H,
        W,
        resolution=resolution
       )

        input_image_np = resize_and_center_crop(
        start_frame,
        target_width=width,
        target_height=height
        )





        lora_text = ""
        if lora_file is not None:
            lora_text = f"\nlora_file={os.path.basename(lora_file)}\nlora_multiplier={lora_multiplier}"
        
        with open(os.path.join(outputs_folder, f'{job_id}.txt'), "w") as file:
            file.write(f"seed={seed}\nresolution={resolution}\nprompt={prompt}\ntotal_second_length={total_second_length}\nuse_teacache={use_teacache}\nn_prompt={n_prompt}\nlatent_window_size={latent_window_size}\nsteps={steps}\ncfg={cfg}\ngs={gs}\nrs={rs}\nteacache_threshold={teacache_threshold}{lora_text}")

        metadata = PngInfo()
        metadata.add_text("prompt", prompt)
        metadata.add_text("n_prompt", n_prompt)
        metadata.add_text("seed", str(seed))
        metadata.add_text("resolution", str(resolution))
        metadata.add_text("use_teacache", str(use_teacache))
        metadata.add_text("total_second_length", str(total_second_length))
        metadata.add_text("latent_window_size", str(latent_window_size))
        metadata.add_text("steps", str(steps))
        metadata.add_text("cfg", str(cfg))
        metadata.add_text("gs", str(gs))
        metadata.add_text("rs", str(rs))
        metadata.add_text("teacache_threshold", str(teacache_threshold))
        if lora_file is not None:
            metadata.add_text("lora_file", str(os.path.basename(lora_file)))
            metadata.add_text("lora_multiplier", str(lora_multiplier))
            #metadata.add_text("gpu_memory_preservation", str(gpu_memory_preservation))
            Image.fromarray(input_image_np).save(os.path.join(outputs_folder, f'{job_id}.png'), pnginfo=metadata)



        
        input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
        input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None]
        


       




        # Processing end image (if provided)

        has_end_image = len(runtime.frames) > 1

        if has_end_image:

         stream.output_queue.push(
         ('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...')))
        )

        end_frame = runtime.frames[-1]

        H_end, W_end, C_end = end_frame.shape

        end_image_np = resize_and_center_crop(
         end_frame,
         target_width=width,
         target_height=height
        )

        Image.fromarray(end_image_np).save(
         os.path.join(outputs_folder, f'{job_id}_end.png')
        )

        end_image_pt = torch.from_numpy(end_image_np).float() / 127.5 - 1

        end_image_pt = end_image_pt.permute(2, 0, 1)[None, :, None]








        # VAE encoding
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'VAE encoding ...'))))

        if not high_vram:
            load_model_as_complete(vae, target_device=gpu)



       
        

      # ==========================================================
      # M5.5 : Latent Collection
      # ==========================================================
      #
      # Goal:
      # Convert every tensor inside tensor_collection into
      # FramePack latent representations using the official
      # vae_encode() helper.
      #
      # This does NOT replace FramePack's original latent
      # variables yet.
      #
      # ==========================================================

        latent_collection = []

        for tensor in tensor_collection:

         latent = vae_encode(
         tensor,
         vae
         )

         latent_collection.append(latent)

        # ==========================================================
        # M7.1
        # Experimental Runtime State
        # ==========================================================

        runtime.experimental["latent_collection"] = latent_collection

        # ==========================================================
        # Build Segment Runtime Collection
        # ==========================================================

        runtime.segment_collection.clear()

        for i in range(len(latent_collection) - 1):

           segment = SegmentRuntime()

           segment.segment_index = i
           segment.duration_seconds = planner.segments[i].duration_seconds

           segment.start_index = i
           segment.end_index = i + 1

           segment.start_frame = runtime.frames[i]
           segment.end_frame = runtime.frames[i + 1]

           segment.start_tensor = tensor_collection[i]
           segment.end_tensor = tensor_collection[i + 1]

           segment.start_latent = latent_collection[i]
           segment.end_latent = latent_collection[i + 1]
           runtime.segment_collection.append(segment)

     


        # CLIP Vision
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'CLIP Vision encoding ...'))))

        if not high_vram:
            load_model_as_complete(image_encoder, target_device=gpu)

        image_encoder_output = hf_clip_vision_encode(input_image_np, feature_extractor, image_encoder)
        image_encoder_last_hidden_state = image_encoder_output.last_hidden_state
        
        if has_end_image:
            end_image_encoder_output = hf_clip_vision_encode(end_image_np, feature_extractor, image_encoder)
            end_image_encoder_last_hidden_state = end_image_encoder_output.last_hidden_state
            # Combine both image embeddings or use a weighted approach
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state + end_image_encoder_last_hidden_state) / 2

        # Dtype
        image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer_dtype)

        # Load transformer model
        if model_changed:
            stream.output_queue.push(("progress", (None, "", make_progress_bar_html(0, "Loading transformer ..."))))

            transformer = None
            time.sleep(1.0)  # wait for the previous model to be unloaded
            torch.cuda.empty_cache()
            gc.collect()

            previous_lora_file = lora_file
            previous_lora_multiplier = lora_multiplier
            previous_fp8_optimization = fp8_optimization

            transformer = load_transfomer()  # bfloat16, on cpu

            if lora_file is not None or fp8_optimization:
                state_dict = transformer.state_dict()

                # LoRA should be merged before fp8 optimization
                if lora_file is not None:
                    # TODO It would be better to merge the LoRA into the state dict before creating the transformer instance.
                    # Use from_config() instead of from_pretrained to make the instance without loading.

                    print(f"Merging LoRA file {os.path.basename(lora_file)} ...")
                    state_dict = merge_lora_to_state_dict(state_dict, lora_file, lora_multiplier, device=gpu)
                    gc.collect()

                if fp8_optimization:
                    TARGET_KEYS = ["transformer_blocks", "single_transformer_blocks"]
                    EXCLUDE_KEYS = ["norm"]  # Exclude norm layers (e.g., LayerNorm, RMSNorm) from FP8

                    # inplace optimization
                    print("Optimizing for fp8")
                    state_dict = optimize_state_dict_with_fp8(state_dict, gpu, TARGET_KEYS, EXCLUDE_KEYS, move_to_device=False)

                    # apply monkey patching
                    apply_fp8_monkey_patch(transformer, state_dict, use_scaled_mm=False)
                    gc.collect()

                info = transformer.load_state_dict(state_dict, strict=True, assign=True)
                print(f"LoRA and/or fp8 optimization applied: {info}")

            if not high_vram:
                DynamicSwapInstaller.install_model(transformer, device=gpu)
            else:
                transformer.to(gpu)
        # Sampling
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Start sampling ...'))))
        rnd = None

        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(
        size=(1, 16, 1 + 2 + 16, height // 8, width // 8),
        dtype=torch.float32
        ).cpu()

        history_pixels = None

        total_generated_latent_frames = 0

        # ===========================================
        # POC : Rolling Anchor Latent
        # ===========================================

        current_anchor_latent = latent_collection[0]



        
      
        # ==========================================================
        # POC-10 : Segment Change Detection
        # ==========================================================
        # ==========================================================
        # M1 : Segment Execution Loop
        # ==========================================================

        for runtime_segment in runtime.segment_collection:
            print()
            print("========================================")
            print(f"RUN SEGMENT {runtime_segment.segment_index}")
            print("========================================")

            # ==========================================================
            # POC : Segment Prompt Selection
            # ==========================================================

            segment_index = runtime_segment.segment_index

            segment_prompt_data = prompt_queue[segment_index]

            segment_positive_prompt = segment_prompt_data["positive"]
            segment_negative_prompt = segment_prompt_data["negative"]

            print()
            print("========================================")
            print("POC : SEGMENT PROMPT")
            print("========================================")
            print("Segment :", segment_index)
            print("Positive:", segment_positive_prompt)
            print("Negative:", segment_negative_prompt)
            print("========================================")


           # ==========================================================
           # M7 : Reset diffusion generator per planner segment
           # ==========================================================

            rnd = torch.Generator("cpu").manual_seed(seed)

            print()
            print("========================================")
            print("M7 : RANDOM GENERATOR")
            print("========================================")
            print("Segment :", runtime_segment.segment_index)
            print("Seed    :", seed)
            print("========================================")

            previous_segment_index = -1

            history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8),dtype=torch.float32).cpu()

            history_pixels = None

            total_generated_latent_frames = 0

            current_anchor_latent = runtime_segment.start_latent

            # ----------------------------------------------------------
            # POC #2 : READ PREVIOUS SEGMENT ROLLING ANCHOR
            # ----------------------------------------------------------

            previous_rolling_anchor = None

            if runtime_segment.segment_index > 0:

                previous_segment = runtime.segment_collection[
                    runtime_segment.segment_index - 1
                ]

                previous_rolling_anchor = (
                    previous_segment.rolling_anchor_latent
                )

                if previous_rolling_anchor is not None:

                    print()
                    print("========================================")
                    print("POC #2 : PREVIOUS ROLLING ANCHOR FOUND")
                    print("========================================")
                    print(
                        "Current Segment :",
                        runtime_segment.segment_index
                    )
                    print(
                        "Previous Segment :",
                        previous_segment.segment_index
                    )
                    print(
                        "Anchor Shape :",
                        tuple(previous_rolling_anchor.shape)
                    )
                    print("========================================")

            else:

                print()
                print("========================================")
                print("POC #2 : NO PREVIOUS ROLLING ANCHOR")
                print("========================================")
                print("Segment :", runtime_segment.segment_index)
                print("========================================")



            



            timeline_iteration = 0

            # ==========================================================
            # M13 : Segment-local Scheduler
            # ==========================================================

            segment_latent_sections = (runtime_segment.duration_seconds * 30) / (latent_window_size * 4)

            segment_latent_sections = int(max(round(segment_latent_sections), 1))

            current_section = segment_latent_sections

            print()
            print("========================================")
            print("M13 : SEGMENT SCHEDULER")
            print("========================================")
            print("Segment :", runtime_segment.segment_index)
            print("Duration :", runtime_segment.duration_seconds)
            print("Sections :", segment_latent_sections)
            print("========================================")

            latent_paddings = list(reversed(range(segment_latent_sections)))

            if segment_latent_sections > 4:
              latent_paddings = [3] + [2] * (segment_latent_sections - 3) + [1, 0]





            # ==========================================================
            # M8 : Segment-local CLIP Vision
            # ==========================================================

            segment_start_np = resize_and_center_crop(runtime_segment.start_frame,target_width=width,target_height=height)

            segment_end_np = resize_and_center_crop(runtime_segment.end_frame,target_width=width,target_height=height)

            # ----------------------------------------------------------
            # PREVIOUS KEYFRAME
            # ----------------------------------------------------------

            previous_frame_np = None
            previous_embed = None

            if runtime_segment.segment_index > 0:

               previous_segment = runtime.segment_collection[runtime_segment.segment_index - 1]

               previous_frame_np = resize_and_center_crop(previous_segment.start_frame,target_width=width,target_height=height)

            # ----------------------------------------------------------
            # INCOMING MOTION
            # ----------------------------------------------------------

            incoming_motion = None

            if previous_embed is not None:

               incoming_motion = (segment_start_embed - previous_embed)


            # ----------------------------------------------------------
            # PREVIOUS KEYFRAME EMBEDDING
            # ----------------------------------------------------------

            if previous_frame_np is not None:

               previous_embed = hf_clip_vision_encode(previous_frame_np,feature_extractor,image_encoder).last_hidden_state




            if not high_vram:
               load_model_as_complete(image_encoder, target_device=gpu)

            segment_start_embed = hf_clip_vision_encode(segment_start_np,feature_extractor,image_encoder).last_hidden_state

            segment_end_embed = hf_clip_vision_encode(segment_end_np,feature_extractor,image_encoder).last_hidden_state

            # ----------------------------------------------------------
            # LOOK-AHEAD IMAGE
            # ----------------------------------------------------------

            lookahead_embed = None

            if runtime_segment.segment_index + 1 < len(runtime.segment_collection):

               next_segment = runtime.segment_collection[runtime_segment.segment_index + 1]

               lookahead_np = resize_and_center_crop(next_segment.end_frame,target_width=width,target_height=height)

               lookahead_embed = hf_clip_vision_encode(lookahead_np,feature_extractor,image_encoder).last_hidden_state


            

        

            # ----------------------------------------------------------
            # TEMPORAL START CONDITIONING
            # ----------------------------------------------------------

            if incoming_motion is not None:

               temporal_start_embed = (segment_start_embed + incoming_motion * 0.15)

            else:

               temporal_start_embed = segment_start_embed


            image_encoder_last_hidden_state = (temporal_start_embed * 0.5 + segment_end_embed * 0.5)


            # IMPORTANT:
            # Convert BOTH branches to the transformer's dtype.
            image_encoder_last_hidden_state = (image_encoder_last_hidden_state.to(transformer_dtype))




            print()
            print("========================================")
            print("M8 : SEGMENT CLIP VISION")
            print("========================================")
            print("Segment :", runtime_segment.segment_index)
            print("Embedding :", tuple(image_encoder_last_hidden_state.shape))
            print("========================================")


            # ==========================================================
            # POC : Segment-local Text Prompt Encoding
            # ==========================================================

            print() 
            print("========================================")
            print("POC : SEGMENT TEXT ENCODING")
            print("========================================")
            print("Segment :", segment_index)
            print("Positive:", segment_positive_prompt)
            print("Negative:", segment_negative_prompt)
            print("========================================")


            # ==========================================================
            # Original FramePack text encoder device preparation
            # ==========================================================

            if not high_vram:
                fake_diffusers_current_device(text_encoder, gpu)
                load_model_as_complete(text_encoder_2, target_device=gpu)


            # ==========================================================
            # Encode segment positive prompt
            # ==========================================================

            llama_vec, clip_l_pooler = encode_prompt_conds(
                segment_positive_prompt,
                text_encoder,
                text_encoder_2,
                tokenizer,
                tokenizer_2
            )


            # ==========================================================
            # Encode segment negative prompt
            # ==========================================================

            if cfg == 1:
                llama_vec_n = torch.zeros_like(llama_vec)
                clip_l_pooler_n = torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                    segment_negative_prompt,
                    text_encoder,
                    text_encoder_2,
                    tokenizer,
                    tokenizer_2
                )


            # ==========================================================
            # Original FramePack prompt padding
            # ==========================================================

            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(
                llama_vec,
                length=512
            )

            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(
                llama_vec_n,
                length=512
            )


            # ==========================================================
            # Original FramePack dtype conversion
            # ==========================================================

            llama_vec = llama_vec.to(transformer_dtype)
            llama_vec_n = llama_vec_n.to(transformer_dtype)
            clip_l_pooler = clip_l_pooler.to(transformer_dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer_dtype)


            # ==========================================================
            # POC : Debug device
            # ==========================================================

            print("DEBUG DEVICE")
            print("llama_vec:", llama_vec.device, llama_vec.dtype)
            print("llama_vec_n:", llama_vec_n.device, llama_vec_n.dtype)
            print("clip_l_pooler:", clip_l_pooler.device, clip_l_pooler.dtype)
            print("clip_l_pooler_n:", clip_l_pooler_n.device, clip_l_pooler_n.dtype)
            print("llama_attention_mask:", llama_attention_mask.device)
            print("llama_attention_mask_n:", llama_attention_mask_n.device)


            for latent_padding in latent_paddings:
                is_last_section = latent_padding == 0
                is_first_section = latent_padding == latent_paddings[0]
                latent_padding_size = latent_padding * latent_window_size


                print(f"=== ENTER LOOP : latent_padding={latent_padding} ===")


                print()
                print("========================================")
                print("BOUNDARY TEST")
                print("Segment :", runtime_segment.segment_index)
                print("First Section :", is_first_section)
                print("Last Section :", is_last_section)
                print("Latent Padding :", latent_padding)
               
                print("========================================")



                print("Before planner lookup")

                active_segment = planner.segments[runtime_segment.segment_index]


                print()
                print("========================================")
                print("W6.4.7 : SEGMENT CONDITIONING ORDER")
                print("========================================")

                print("Segment :", runtime_segment.segment_index)
                print("Start Index :", runtime_segment.start_index)
                print("End Index   :", runtime_segment.end_index)

                print("Start Frame Shape :", runtime_segment.start_frame.shape)
                print("End Frame Shape   :", runtime_segment.end_frame.shape)

                print("Start Latent Shape :", runtime_segment.start_latent.shape)
                print("End Latent Shape   :", runtime_segment.end_latent.shape)

                print("========================================")


                planner_start_latent = runtime_segment.start_latent
                planner_end_latent = runtime_segment.end_latent


                print("Start Latent Mean :", float(planner_start_latent.mean()))
                print("End Latent Mean   :", float(planner_end_latent.mean()))


                # ==========================================================
                # POC : Match Start Latent Statistics to End Latent
                # ==========================================================

                start_mean = planner_start_latent.mean()
                start_std  = planner_start_latent.std()

                end_mean = planner_end_latent.mean()
                end_std  = planner_end_latent.std()

                planner_start_latent = ((planner_start_latent - start_mean) / (start_std + 1e-6)) * end_std + end_mean


                if stream.input_queue.top() == 'end':
                   stream.output_queue.push(('end', None))
                   return

                print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

                indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size,             latent_window_size, 1, 2, 16], dim=1)
                clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)


                print("========================================")
                print("POC-1 : Latent Index")
                print("========================================")

                print(clean_latent_indices)
                print(clean_latent_indices.shape)

                clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)

                # ==========================================================
                # SEGMENT START : PURE KEYFRAME CONDITIONING
                # ==========================================================
                # Do NOT blend the previous segment's generated endpoint
                # into the current segment's start keyframe.
                #
                # Each segment starts strictly from its own keyframe.
            
                clean_latents_pre = planner_start_latent.to(history_latents)


                


                clean_latents_2x = torch.zeros_like(clean_latents_2x)

                clean_latents_4x = torch.zeros_like(clean_latents_4x)


                # ==========================================================
                # POC-1 : Three Keyframe Conditioning
                # ==========================================================

                planner_end_latent = planner_end_latent.to(history_latents)



                # ==========================================================
                # POC-SEGMENT-01
                # Inject planner end latent only on the first scheduler
                # section, matching the original FramePack Plus behavior.
                # ==========================================================
          

                if is_first_section:
                   clean_latents_post = planner_end_latent

                # ==========================================================
                # Planner Conditioning
                # ==========================================================
                # The conditioning packet now consists only of the current
                # segment's start and end keyframes.
                # History remains available through clean_latents_2x and
                # clean_latents_4x.
                # ==========================================================

                clean_latents = torch.cat([clean_latents_pre,clean_latents_post],dim=2)



                print()
                print("========================================")
                print("STEP 4 : CONDITIONING PACKET")
                print("========================================")

                print("Timeline :", timeline_iteration)
                print()

                print("Start Latent :", id(clean_latents_pre))
                print("End Latent   :", id(clean_latents_post))

                print()

                print("Packet Shape :", clean_latents.shape)

                print()

                print("Slice 0 Mean :", float(clean_latents[:,:,0].mean()))
                print("Slice 1 Mean :", float(clean_latents[:,:,1].mean()))

                print("========================================")


                if not high_vram:
                   unload_complete_models()
                   move_model_to_device_with_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation)

                if use_teacache:
                   transformer.initialize_teacache(enable_teacache=True, num_steps=steps, rel_l1_thresh=teacache_threshold)
                else:
                    transformer.initialize_teacache(enable_teacache=False)

                def callback(d):
                    preview = d['denoised']
                    preview = vae_decode_fake(preview)

                    preview = (preview * 255.0).detach().cpu().numpy().clip(0, 255).astype(np.uint8)
                    preview = einops.rearrange(preview, 'b c t h w -> (b h) (t w) c')

                    if stream.input_queue.top() == 'end':
                       stream.output_queue.push(('end', None))
                       raise KeyboardInterrupt('User ends the task.')

                    current_step = d['i'] + 1
                    percentage = max(1, int(100.0 * ((current_step + ((segment_latent_sections - current_section) * steps)) / (segment_latent_sections * steps)))) # max() to avoid "division by zero" errors
                    elapsed_time = int(time.time() - start_time)
                    time_left = int((100 * elapsed_time / percentage) - elapsed_time)
                    hint = f'Sampling {current_step}/{steps}'
                    desc = f'Total progress {percentage}%, elapsed {elapsed_time // 60}:{elapsed_time % 60:02}, time_left {time_left // 60}:{time_left % 60:02}, Section {segment_latent_sections - current_section + 1}/{segment_latent_sections}<br/>Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds.'
                    stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                    return


                # ==========================================================
                # POC : Verify Segment Prompt Embeddings
                # ==========================================================

                print()
                print("========================================")
                print("POC : SAMPLER PROMPT CONDITIONING")
                print("========================================")
                print("Segment :", segment_index)
                print("Positive prompt :", segment_positive_prompt)
                print("Negative prompt :", segment_negative_prompt)
                print("Positive embedding :", tuple(llama_vec.shape))
                print("Negative embedding :", tuple(llama_vec_n.shape))
                print("========================================")



                generated_latents = sample_hunyuan(
                    transformer=transformer,
                    sampler='unipc',
                    width=width,
                    height=height,
                    frames=num_frames,
                    real_guidance_scale=cfg,
                    distilled_guidance_scale=gs,
                    guidance_rescale=rs,
                    # shift=3.0,
                    num_inference_steps=steps,
                    generator=rnd,
                    prompt_embeds=llama_vec,
                    prompt_embeds_mask=llama_attention_mask,
                    prompt_poolers=clip_l_pooler,
                    negative_prompt_embeds=llama_vec_n,
                    negative_prompt_embeds_mask=llama_attention_mask_n,
                    negative_prompt_poolers=clip_l_pooler_n,
                    device=gpu,
                    dtype=torch.bfloat16,
                    image_embeddings=image_encoder_last_hidden_state,
                    latent_indices=latent_indices,
                    clean_latents=clean_latents,
                    clean_latent_indices=clean_latent_indices,
                    clean_latents_2x=clean_latents_2x,
                    clean_latent_2x_indices=clean_latent_2x_indices,
                    clean_latents_4x=clean_latents_4x,
                    clean_latent_4x_indices=clean_latent_4x_indices,
                    callback=callback,
                    )


                

                # ===========================================
                # POC : Update Rolling Anchor
                # ===========================================

                current_anchor_latent = generated_latents[:, :, -1:].detach().clone()

                # ----------------------------------------------------------
                # POC #2 : SAVE ROLLING ANCHOR TO CURRENT SEGMENT
                # ----------------------------------------------------------

                runtime_segment.rolling_anchor_latent = (
                    current_anchor_latent.detach().clone()
                )

                print()
                print("========================================")
                print("POC #2 : SAVED ROLLING ANCHOR")
                print("========================================")
                print("Segment :", runtime_segment.segment_index)
                print("Shape   :", tuple(runtime_segment.rolling_anchor_latent.shape))
                print("Mean    :", float(runtime_segment.rolling_anchor_latent.mean()))
                print("Std     :", float(runtime_segment.rolling_anchor_latent.std()))
                print("========================================")


                print()
                print("========================================")
                print("STEP 6 : ROLLING ANCHOR")
                print("========================================")
                print("Timeline :", timeline_iteration)
                print()
                print("Anchor ID :", id(current_anchor_latent))
                print("Shape :", current_anchor_latent.shape)

                print()

                print("Mean :", float(current_anchor_latent.mean()))

                print("Std  :", float(current_anchor_latent.std()))

                print("========================================")


                print(f"Encoding {'final' if is_last_section else 'intermediate'} output video {job_id}.mp4 ...")


                print()
                print("========================================")
                print("POC-12 : HISTORY BEFORE UPDATE")
                print("========================================")
                print("Timeline :", timeline_iteration)
                print("History Shape :", tuple(history_latents.shape))
                print("Generated Shape :", tuple(generated_latents.shape))
                print("========================================")
           

                total_generated_latent_frames += int(generated_latents.shape[2])
                history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)


                print()
                print("========================================")
                print("STEP 7 : HISTORY EVOLUTION")
                print("========================================")

                print("Timeline :", timeline_iteration)

                print()

                print("Generated Shape :", tuple(generated_latents.shape))
                print("History Shape   :", tuple(history_latents.shape))

                print()

                print("Generated Mean :", float(generated_latents.mean()))
                print("Generated Std  :", float(generated_latents.std()))

                print()

                print("History Mean :", float(history_latents.mean()))
                print("History Std  :", float(history_latents.std()))

                print()

                print("History Frames :", history_latents.shape[2])

                print("========================================")

            

                if not high_vram:
                   offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                   load_model_as_complete(vae, target_device=gpu)

                real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]


                #LATENT STORED
                runtime_segment.generated_latents = real_history_latents.detach().clone()

                print()
                print("========================================")
                print("M2 : LATENT STORED")
                print("========================================")
                print("Segment :", runtime_segment.segment_index)
                print("Shape   :", runtime_segment.generated_latents.shape)
                print("========================================")



                # ==========================================================
                # M9 : Preserve Complete History
                # ==========================================================

                runtime_segment.full_history_latents = (history_latents.detach().clone())

                print()
                print("========================================")
                print("M9 : FULL HISTORY STORED")
                print("========================================")
                print("Segment :", runtime_segment.segment_index)
                print("History Shape :", tuple(runtime_segment.full_history_latents.shape))
                print("========================================")


                # ==========================================================
                # M2
                # Only decode after ALL segments finish.
                # ==========================================================

                if (runtime_segment.segment_index == len(runtime.segment_collection) - 1 and is_last_section):


                  # ==========================================================
                  # M4 : Latent Timeline Manager
                  # ==========================================================

                  latent_timeline = []

                  print()
                  print("========================================")
                  print("M4 : LATENT TIMELINE")
                  print("========================================")

                  total_frames = 0

                  for seg in runtime.segment_collection:

                      if seg.generated_latents is None:
                         continue

                      frame_count = seg.generated_latents.shape[2]

                      latent_timeline.append(
                       {
                        "segment": seg.segment_index,

                        "latent": seg.generated_latents,

                        "full_history": seg.full_history_latents,

                        "frames": frame_count,

                        "start_frame": total_frames,

                        "end_frame": total_frames + frame_count - 1,

                        "shape": tuple(seg.generated_latents.shape),

                        "dtype": str(seg.generated_latents.dtype),
                        }
                       )

                      print(f"Segment {seg.segment_index}"f" | Frames {frame_count}"f" | Shape {tuple(seg.generated_latents.shape)}"f" | Timeline {total_frames} -> {total_frames + frame_count - 1}")

                      total_frames += frame_count

                  print("----------------------------------------")
                  print("Timeline Frames :", total_frames)
                  print("Segments :", len(latent_timeline))
                  print("========================================")


                  if len(latent_timeline) == 0:
                      raise RuntimeError("No stored latent segments found.")

                  # ==========================================================
                  # M4 POC
                  # Use the first timeline entry for decoding.
                  # (This will be replaced by the stitch manager later.)
                  # ==========================================================

                  # ==========================================================
                  # M5 : Timeline Builder
                  # ==========================================================

                  timeline_builder = []

                  print()
                  print("========================================")
                  print("M5 : BUILD TIMELINE")
                  print("========================================")

                  for item in latent_timeline:

                     timeline_builder.append(item["latent"])
                     print(f"M11 : Timeline uses GENERATED latent"f" | Segment {item['segment']}"f" | Frames {item['latent'].shape[2]}")

                     print(f"Use Segment {item['segment']}"f" | Frames {item['frames']}")

                  print("----------------------------------------")
                  print("Timeline Entries :", len(timeline_builder))
                  print("========================================")

                  # ----------------------------------------------------------
                  # POC
                  # Still output the first entry.
                  # Stitching comes next milestone.
                  # ----------------------------------------------------------

                  # ==========================================================
                  # M6 : Timeline Assembly (POC)
                  # ==========================================================

                  timeline_output = timeline_builder[0]

                  print()
                  print("========================================")
                  print("M6 : TIMELINE ASSEMBLY")
                  print("========================================")

                  for i in range(1, len(timeline_builder)):

                    current = timeline_builder[i]

                    # ==========================================================
                    # M10 : Configurable Timeline Overlap
                    # ==========================================================

                    timeline_overlap = 2

                    overlap = min(timeline_overlap,current.shape[2] - 1)

                    print(f"Append Segment {i}"f" | Total {current.shape[2]}"f" | Overlap {overlap}"f" | Append {current.shape[2] - overlap}")

                    timeline_output = torch.cat([timeline_output,current[:, :, overlap:]],dim=2)

                    print("----------------------------------------")
                    print("Final Timeline :", tuple(timeline_output.shape))
                    print("========================================")

                  if history_pixels is None:
                      history_pixels = vae_decode(timeline_output, vae).cpu()
                  else:
                      section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                      overlapped_frames = latent_window_size * 4 - 3

                      current_pixels = vae_decode(timeline_output[:, :, :section_latent_frames], vae).cpu()

                      print()
                      print("===== VAE Decode Info =====")
                      print("Shape :", tuple(current_pixels.shape))
                      print("Min   :", float(current_pixels.min()))
                      print("Max   :", float(current_pixels.max()))
                      print("Dtype :", current_pixels.dtype)
                      print("===========================")
                      print()
                
                      history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)



                  if not high_vram:
                     unload_complete_models()



                  #output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
                  output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')

                  save_bcthw_as_mp4(history_pixels, output_filename, fps=16, crf=mp4_crf)

                  #print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

                  stream.output_queue.push(('file', output_filename))
                  print(f"=== EXIT LOOP : latent_padding={latent_padding} ===")


            
                current_section -= 1
                timeline_iteration += 1


                print("Finished scheduler iteration", timeline_iteration - 1)
                print("--------------------------------")



                if is_last_section:
                   elapsed_time = int(time.time() - start_time)
                   print(f"Final stats : {elapsed_time} ({elapsed_time // 60}:{elapsed_time % 60:02}) ({elapsed_time/total_second_length:.2f}/s) for {total_second_length}s at {resolution} - {width} x {height}")
                   with open(os.path.join(outputs_folder, '_stats.txt'), "a") as file:
                    file.write(f"{resolution},{total_second_length},{elapsed_time},{elapsed_time/total_second_length:.2f}\n")
                   break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(text_encoder, text_encoder_2, image_encoder, vae, transformer)


     


    stream.output_queue.push(('end', None))
    return


def process(
    frame_collection,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    resolution,
    teacache_threshold,
    lora_file,
    lora_multiplier,
    fp8_optimization
):
    global stream
    assert frame_collection is not None
    assert len(frame_collection) > 0, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(
    worker,
    frame_collection,
    prompt,
    n_prompt,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    resolution,
    teacache_threshold,
    lora_file,
    lora_multiplier,
    fp8_optimization
    )



    output_filename = None

    while True:
        flag, data = stream.output_queue.next()
        #l yielding : [result_video, preview_image, progress_desc, progress_bar, start_button, end_button]
        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break


def process_multikey(
    frame_collection,
    prompt,
    n_prompt,
    prompt_queue,
    seed,
    total_second_length,
    latent_window_size,
    steps,
    cfg,
    gs,
    rs,
    gpu_memory_preservation,
    use_teacache,
    mp4_crf,
    resolution,
    teacache_threshold,
    lora_file,
    lora_multiplier,
    fp8_optimization
):

    global stream

    assert frame_collection is not None
    assert len(frame_collection) > 0

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(

        worker,

        frame_collection,

        prompt,

        n_prompt,

        prompt_queue,

        seed,

        total_second_length,

        latent_window_size,

        steps,

        cfg,

        gs,

        rs,

        gpu_memory_preservation,

        use_teacache,

        mp4_crf,

        resolution,

        teacache_threshold,

        lora_file,

        lora_multiplier,

        fp8_optimization

    )

    output_filename = None

    while True:

        flag, data = stream.output_queue.next()

        if flag == 'file':
            output_filename = data
            yield output_filename, gr.update(), gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'progress':
            preview, desc, html = data
            yield gr.update(), gr.update(visible=True, value=preview), desc, html, gr.update(interactive=False), gr.update(interactive=True)

        if flag == 'end':
            yield output_filename, gr.update(visible=False), gr.update(), '', gr.update(interactive=True), gr.update(interactive=False)
            break



def end_process():
    stream.input_queue.push('end')

def res_change(pil_image, resolution, gpu_memory_preservation):
    width, height = 0, 0
    if pil_image:
        width, height = pil_image.size
        height, width = find_nearest_bucket(height, width, resolution=resolution)
        #print(f"{width}x{height}")
    if resolution >= 800:
        gpu_memory_preservation = 6 + (0.1 * round((resolution - 640) / 10))
        print(f"Resolution changed -> setting gpu_mem to {gpu_memory_preservation}")
    return gr.update(label=f"Resolution : {width}x{height}"), gpu_memory_preservation

def str_to_bool(value: str) -> bool:
    """
    Convert a string value to boolean.
    
    Whitespace is stripped and comparison is done in lowercase.
    """
    return value.strip().lower() == "true"


def safe_convert(value: Any, conv_func, default: Any) -> Any:
    """
    Safely convert a value using conv_func. If conversion fails,
    return the default.
    """
    try:
        return conv_func(value)
    except (ValueError, TypeError):
        return default

def extract_metadata(
    pil_image: Image.Image,
    prompt: str,
    seed: int,
    resolution: int,
    use_teacache: bool,
    total_second_length: float,
    latent_window_size: int,
    teacache_threshold: float,
    lora_file:str,
    lora_multiplier: float
) -> Tuple[str, int, int, bool, float, int, float, str, float]:
    """
    Extracts metadata from a PIL image's info dictionary using provided defaults.
    
    If the image contains metadata (via its 'info' attribute), each parameter is
    updated by converting the metadata value. If conversion fails, the original
    default is maintained.
    """
    if pil_image and hasattr(pil_image, "info"):
        print("Extracting metadata from dropped image...")
        metadata = pil_image.info

        # Display all metadata
        for key, value in metadata.items():
            print(f"{key}: {value}")

        prompt = metadata.get("prompt", prompt)
        seed = safe_convert(metadata.get("seed", seed), int, seed)
        resolution = safe_convert(metadata.get("resolution", resolution), int, resolution)
        
        # The metadata value for use_teacache might not be a string; ensure conversion.
        use_teacache_val = str(metadata.get("use_teacache", str(use_teacache)))
        use_teacache = str_to_bool(use_teacache_val)
        
        total_second_length = safe_convert(metadata.get("total_second_length", total_second_length), float, total_second_length)
        latent_window_size = safe_convert(metadata.get("latent_window_size", latent_window_size), int, latent_window_size)
        teacache_threshold = safe_convert(metadata.get("teacache_threshold", 0.15), float, teacache_threshold)
        lora_file_name = metadata.get("lora_file")
        if(lora_file_name is not None):
            lora_file = os.path.realpath(os.path.join(os.path.dirname(__file__), 'loras', lora_file_name))
            print(f"found lora metadata -> {lora_file}")
            lora_multiplier = safe_convert(metadata.get("lora_multiplier", lora_multiplier), float, lora_multiplier)
        if(metadata.get("f1") == "True"):
            print("Warning! The image you loaded looks like it was generated with F1, your result will be different.")
    else:
        print("Image deleted or missing metadata.")

    return prompt, seed, resolution, use_teacache, total_second_length, latent_window_size, teacache_threshold, lora_file, lora_multiplier

#------------- open output folder -------------
import subprocess
import platform

def open_output_folder():
    path = os.path.normpath(outputs_folder)
    if platform.system() == "Windows":
        os.startfile(path)
    elif platform.system() == "Darwin":
        subprocess.Popen(["open", path])
    elif "microsoft-standard-WSL2" in platform.uname().release:
        subprocess.Popen(["explorer.exe", subprocess.check_output(["wslpath", "-w", path])])
    else:
        subprocess.Popen(["xdg-open", path])
#------------- open output folder -------------

quick_prompts = [
    'The girl dances gracefully, with clear movements, full of charm.',
    'A character doing some simple body movements.',
]
quick_prompts = [[x] for x in quick_prompts]


css = make_progress_bar_css()
block = gr.Blocks(css=css,analytics_enabled=False).queue()
with block:
    gr.Markdown('# FramePackPlus')
    with gr.Row():
        with gr.Column():
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(sources='upload', type="pil", label="Start Frame", height=320, show_fullscreen_button=False, interactive=True)
                with gr.Column():
                    end_image = gr.Image(sources='upload', type="numpy", label="End Frame (Optional)", height=320)
                multikey_frames = gr.JSON(
                 label="MultiKey Frames",
                 visible=False
                )
            
            prompt = gr.Textbox(label="Prompt", value='')
            example_quick_prompts = gr.Dataset(samples=quick_prompts, label='Quick List', samples_per_page=1000, components=[prompt])
            example_quick_prompts.click(lambda x: x[0], inputs=[example_quick_prompts], outputs=prompt, show_progress=False, queue=False)

            with gr.Row():
                start_button = gr.Button(value="Start Generation")
                end_button = gr.Button(value="End Generation", interactive=False)

            with gr.Group():
                with gr.Row():
                    use_teacache = gr.Checkbox(label='Use TeaCache', value=True, info='Faster speed, but often makes hands and fingers slightly worse.')
                    seed = gr.Number(label="Seed", value=31337, precision=0)
                total_second_length = gr.Slider(label="Total Video Length (Seconds)", minimum=1, maximum=120, value=5, step=0.1)
                resolution = gr.Slider(label="Resolution", minimum=240, maximum=1024, value=256, step=16)
                gpu_memory_preservation = gr.Slider(label="GPU Inference Preserved Memory (GB) (larger means slower)", minimum=6, maximum=32, value=6, step=0.1, info="Set this number to a larger value if you encounter OOM. Larger value causes slower speed.")
                latent_window_size = gr.Slider(label="Latent Window Size", minimum=1, maximum=33, value=9, step=1, visible=True)  # Should not change
                teacache_threshold = gr.Slider(label="Teacache Threshold", minimum=0.05, maximum=0.5, value=0.15, step=0.05, info="0.1 for 1.6x speedup, default 0.15 for 2.1x speedup", visible=True)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=25, step=1, info='Changing this value is not recommended.')
                n_prompt = gr.Textbox(label="Negative Prompt", value="", visible=True)  # Not used
                cfg = gr.Slider(label="CFG Scale", minimum=1.0, maximum=32.0, value=1.0, step=0.01, visible=True)  # Should not change
                gs = gr.Slider(label="Distilled CFG Scale", minimum=1.0, maximum=32.0, value=10.0, step=0.01, info='Changing this value is not recommended.')
                rs = gr.Slider(label="CFG Re-Scale", minimum=0.0, maximum=1.0, value=0.0, step=0.01, visible=False)  # Should not change
                mp4_crf = gr.Slider(label="x264 Compression", minimum=0, maximum=30, value=16, step=1, info="Lower means better quality. 0 is uncompressed. Default is 16")
            with gr.Group():
                lora_file = gr.File(label="LoRA File", file_count="single", type="filepath")
                lora_multiplier = gr.Slider(label="LoRA Multiplier", minimum=0.0, maximum=1.0, value=0.8, step=0.1)
                fp8_optimization = gr.Checkbox(label="FP8 Optimization", value=False)
            
            input_image.change(fn=extract_metadata, inputs=[input_image,prompt, seed, resolution, use_teacache, total_second_length, latent_window_size, teacache_threshold, lora_file, lora_multiplier], outputs=[prompt, seed, resolution, use_teacache, total_second_length, latent_window_size, teacache_threshold, lora_file, lora_multiplier]).then(fn=res_change, inputs=[input_image,resolution,gpu_memory_preservation], outputs=[resolution, gpu_memory_preservation])
            resolution.change(fn=res_change, inputs=[input_image,resolution,gpu_memory_preservation], outputs=[resolution, gpu_memory_preservation])

        with gr.Column():
            preview_image = gr.Image(label="Next Latents", height=200, visible=False)
            result_video = gr.Video(label="Finished Frames", autoplay=False, show_share_button=False, height=512, loop=True)
            #gr.Markdown('Note that the ending actions will be generated before the starting actions due to the inverted sampling. If the starting action is not in the video, you just need to wait, and it will be generated later.')
            progress_desc = gr.Markdown('', elem_classes='no-generating-animation')
            progress_bar = gr.HTML('', elem_classes='no-generating-animation')
            open_output_folder_button = gr.Button("📂")
            open_output_folder_button.click(fn=open_output_folder, inputs=[], outputs=[])

    ips = [input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, teacache_threshold, lora_file, lora_multiplier, fp8_optimization]

   
    prompt_queue = gr.Textbox(label="Prompt Queue",visible=False)

    
    multikey_ips = [
      multikey_frames,
      prompt,
      n_prompt,
      prompt_queue,
      seed,
      total_second_length,
      latent_window_size,
      steps,
      cfg,
      gs,
      rs,
      gpu_memory_preservation,
      use_teacache,
      mp4_crf,
      resolution,
      teacache_threshold,
      lora_file,
      lora_multiplier,
      fp8_optimization
    ]



    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])

    multikey_button = gr.Button(visible=False)

    multikey_button.click(
     fn=process_multikey,
     inputs=multikey_ips,
     outputs=[
        result_video,
        preview_image,
        progress_desc,
        progress_bar,
        start_button,
        end_button
    ],
    api_name="process_multikey"
    )

    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
