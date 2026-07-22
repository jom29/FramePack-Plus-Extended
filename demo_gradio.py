import gc
from diffusers_helper.hf_login import login

import os

os.environ['HF_HOME'] = os.path.abspath(os.path.realpath(os.path.join(os.path.dirname(__file__), './hf_download')))

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
    transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained('lllyasviel/FramePackI2V_HY', torch_dtype=torch.bfloat16).cpu()
    #transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(f"{os.environ['HF_HOME']}/hub/models--lllyasviel--FramePackI2V_HY/snapshots/86cef4396041b6002c957852daac4c91aaa47c79", torch_dtype=torch.bfloat16).cpu()
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





@torch.no_grad()
def worker(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, teacache_threshold, lora_file, lora_multiplier, fp8_optimization):

    # ==========================================================
    # M6 : Runtime State
    # ==========================================================

    runtime = {}

    runtime["frames"] = []
    runtime["processed_frames"] = []
    runtime["tensors"] = []
    runtime["latents"] = []

    runtime["conditioning"] = {}
    runtime["resources"] = {}
    runtime["timeline"] = {}
    runtime["compatibility"] = {}




    
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

    frame_collection = []

    if input_image is not None:
      frame_collection.append(input_image)

    if end_image is not None:
      frame_collection.append(end_image)

    runtime["frames"] = frame_collection





    print()
    print("========================================")
    print("M5.2 : Runtime Frame Collection")
    print("========================================")
    print()

    print(f"Total Frames : {len(frame_collection)}")
    print()

    for index, frame in enumerate(frame_collection):

     print(f"Frame {index}")

     if frame is None:
        print("Status : None")
     else:
        print("Shape  :", frame.shape)
        print("DType  :", frame.dtype)

     print()







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
    
    for frame in frame_collection:
    
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
     # Verification Logs
     # ==========================================================
    
    print()
    print("========================================")
    print("M5.3 : Processed Frame Collection")
    print("========================================")
    print()
    
    print(f"Total Processed Frames : {len(processed_frame_collection)}")
    print()
    
    for index, processed in enumerate(processed_frame_collection):
    
        print(f"Processed Frame {index}")
    
        print("Shape :", processed.shape)
        print("DType :", processed.dtype)
    
        print()
    








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
      
      
    # ==========================================================
    # Verification Logs
    # ==========================================================
      
    print()
    print("========================================")
    print("M5.4 : Tensor Collection")
    print("========================================")
    print()
      
    print(f"Total Tensors : {len(tensor_collection)}")
    print()
      
    for index, tensor in enumerate(tensor_collection):
      
        print(f"Tensor {index}")
      
        print("Shape :", tuple(tensor.shape))
        print("DType :", tensor.dtype)
        print("Device:", tensor.device)
      
        print("Min   :", float(tensor.min()))
        print("Max   :", float(tensor.max()))
      
        print()
      







   


    model_changed = transformer is None or (
        lora_file != previous_lora_file
        or lora_multiplier != previous_lora_multiplier
        or fp8_optimization != previous_fp8_optimization
    )

    total_latent_sections = (total_second_length * 30) / (latent_window_size * 4)
    total_latent_sections = int(max(round(total_latent_sections), 1))

    job_id = generate_timestamp()
    job_id = f'{job_id}_{resolution}_{seed}{"_teacache" if use_teacache else ""}'

    stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Starting ...'))))

    try:
        # Clean GPU
        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )


   
        # Text encoding

        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Text encoding ...'))))

        if not high_vram:
            fake_diffusers_current_device(text_encoder, gpu)  # since we only encode one text - that is one model move and one encode, offload is same time consumption since it is also one load and one encode.
            load_model_as_complete(text_encoder_2, target_device=gpu)

        llama_vec, clip_l_pooler = encode_prompt_conds(prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        if cfg == 1:
            llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
        else:
            llama_vec_n, clip_l_pooler_n = encode_prompt_conds(n_prompt, text_encoder, text_encoder_2, tokenizer, tokenizer_2)

        llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
        llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)


          # Processing input image (start frame)
        stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing start frame ...'))))

        H, W, C = input_image.shape
        height, width = find_nearest_bucket(H, W, resolution=resolution)
        input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)



      




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
        has_end_image = end_image is not None
        if has_end_image:
            stream.output_queue.push(('progress', (None, '', make_progress_bar_html(0, 'Processing end frame ...'))))
            
            H_end, W_end, C_end = end_image.shape
            end_image_np = resize_and_center_crop(end_image, target_width=width, target_height=height)
            
            Image.fromarray(end_image_np).save(os.path.join(outputs_folder, f'{job_id}_end.png'))
            
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
       # Verification Logs
       # ==========================================================
        print()
        print("========================================")
        print("M5.5 : Latent Collection")
        print("========================================")
        print()

        print(f"Total Latents : {len(latent_collection)}")
        print()

        for index, latent in enumerate(latent_collection):

          print(f"Latent {index}")

          print("Shape :", tuple(latent.shape))
          print("DType :", latent.dtype)
          print("Device:", latent.device)

          print("Min   :", float(latent.min()))
          print("Max   :", float(latent.max()))

          print()






        # ==========================================================
        # M5.5A : Compatibility Layer
        # ==========================================================
        #
        # Goal:
        # Bridge the new latent_collection with the original
        # FramePack runtime.
        #
        # The original sampler still expects:
        #
        #     start_latent
        #     end_latent
        #
        # These variables are recreated from the collection so
        # the remaining FramePack pipeline continues to work.
        # 
        # NOTE:
        # This is TEMPORARY.
        #
        # Future milestones will remove this bridge and allow
        # the sampler to work directly with latent_collection.
        #
        # ==========================================================

        print()
        print("========================================")
        print("M5.5A : Compatibility Layer")
        print("========================================")
        print()

        # ----------------------------------------------------------
        # Assign Start Latent
        # ----------------------------------------------------------

        start_latent = latent_collection[0]

        print("Start Latent Assigned")
        print("Shape :", tuple(start_latent.shape))
        print()

        # ----------------------------------------------------------
        # Assign End Latent
        # ----------------------------------------------------------

        if len(latent_collection) > 1:

          end_latent = latent_collection[1]

          print("End Latent Assigned")
          print("Shape :", tuple(end_latent.shape))
          print()

        else:

           end_latent = None

           print("No End Latent")
           print()
       



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
        llama_vec = llama_vec.to(transformer_dtype)
        llama_vec_n = llama_vec_n.to(transformer_dtype)
        clip_l_pooler = clip_l_pooler.to(transformer_dtype)
        clip_l_pooler_n = clip_l_pooler_n.to(transformer_dtype)
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

        rnd = torch.Generator("cpu").manual_seed(seed)
        num_frames = latent_window_size * 4 - 3

        history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
        history_pixels = None
        total_generated_latent_frames = 0

        latent_paddings = list(reversed(range(total_latent_sections)))

        if total_latent_sections > 4:
            # In theory the latent_paddings should follow the above sequence, but it seems that duplicating some
            # items looks better than expanding it when total_latent_sections > 4
            # One can try to remove below trick and just
            # use `latent_paddings = list(reversed(range(total_latent_sections)))` to compare
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
            #latent_paddings = list(reversed(range(total_latent_sections)))
        
        current_section = total_latent_sections
        
        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            is_first_section = latent_padding == latent_paddings[0]
            latent_padding_size = latent_padding * latent_window_size

            if stream.input_queue.top() == 'end':
                stream.output_queue.push(('end', None))
                return

            print(f'latent_padding_size = {latent_padding_size}, is_last_section = {is_last_section}, is_first_section = {is_first_section}')

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)

            clean_latents_pre = start_latent.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
            
            # Use end image latent for the first section if provided
            if has_end_image and is_first_section:
                clean_latents_post = end_latent.to(history_latents)
                clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

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
                percentage = max(1, int(100.0 * ((current_step + ((total_latent_sections - current_section) * steps)) / (total_latent_sections * steps)))) # max() to avoid "division by zero" errors
                elapsed_time = int(time.time() - start_time)
                time_left = int((100 * elapsed_time / percentage) - elapsed_time)
                hint = f'Sampling {current_step}/{steps}'
                desc = f'Total progress {percentage}%, elapsed {elapsed_time // 60}:{elapsed_time % 60:02}, time_left {time_left // 60}:{time_left % 60:02}, Section {total_latent_sections - current_section + 1}/{total_latent_sections}<br/>Total generated frames: {int(max(0, total_generated_latent_frames * 4 - 3))}, Video length: {max(0, (total_generated_latent_frames * 4 - 3) / 30) :.2f} seconds.'
                stream.output_queue.push(('progress', (preview, desc, make_progress_bar_html(percentage, hint))))
                return

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

            print(f"Encoding {'final' if is_last_section else 'intermediate'} output video {job_id}.mp4 ...")

            if is_last_section:
                generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)

            total_generated_latent_frames += int(generated_latents.shape[2])
            history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)

            if not high_vram:
                offload_model_from_device_for_memory_preservation(transformer, target_device=gpu, preserved_memory_gb=8)
                load_model_as_complete(vae, target_device=gpu)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]

            if history_pixels is None:
                history_pixels = vae_decode(real_history_latents, vae).cpu()
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3

                current_pixels = vae_decode(real_history_latents[:, :, :section_latent_frames], vae).cpu()
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            if not high_vram:
                unload_complete_models()

            #output_filename = os.path.join(outputs_folder, f'{job_id}_{total_generated_latent_frames}.mp4')
            output_filename = os.path.join(outputs_folder, f'{job_id}.mp4')

            save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)

            #print(f'Decoded. Current latent shape {real_history_latents.shape}; pixel shape {history_pixels.shape}')

            stream.output_queue.push(('file', output_filename))

            current_section -= 1

            if is_last_section:
                elapsed_time = int(time.time() - start_time)
                print(f"Final stats : {elapsed_time} ({elapsed_time // 60}:{elapsed_time % 60:02}) ({elapsed_time/total_second_length:.2f}/s) for {total_second_length}s at {resolution} - {width} x {height}")
                with open(os.path.join(outputs_folder, '_stats.txt'), "a") as file:
                    file.write(f"{resolution},{total_second_length},{elapsed_time},{elapsed_time/total_second_length:.2f}\n")
                break
    except:
        traceback.print_exc()

        if not high_vram:
            unload_complete_models(
                text_encoder, text_encoder_2, image_encoder, vae, transformer
            )

    stream.output_queue.push(('end', None))
    return


def process(input_image, end_image, prompt, n_prompt, seed, total_second_length, latent_window_size, steps, cfg, gs, rs, gpu_memory_preservation, use_teacache, mp4_crf, resolution, teacache_threshold, lora_file, lora_multiplier, fp8_optimization):
    global stream
    assert input_image is not None, 'No input image!'

    yield None, None, '', '', gr.update(interactive=False), gr.update(interactive=True)

    stream = AsyncStream()

    async_run(
    worker,
    np.array(input_image),
    end_image,
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
                resolution = gr.Slider(label="Resolution", minimum=240, maximum=1024, value=640, step=16)
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
    start_button.click(fn=process, inputs=ips, outputs=[result_video, preview_image, progress_desc, progress_bar, start_button, end_button])
    end_button.click(fn=end_process)


block.launch(
    server_name=args.server,
    server_port=args.port,
    share=args.share,
    inbrowser=args.inbrowser,
)
