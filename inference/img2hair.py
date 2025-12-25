import os
import torch
import numpy as np
import k_diffusion as K
import torchvision.transforms as T
import torchvision
import traceback
import gc
import cv2
import mediapipe as mp

from models.strand_codec import StrandCodec
from models.rgb_to_material import RGB2MaterialModel
from utils.diffusion_utils import sample_images_cfg_yield
from utils.strand_util import sample_strands_from_scalp_with_density
from data_loader.dataloader import DiffLocksDataset
from platform_config import cfg
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DEFAULT_BODY_DATA_DIR = "data_loader/difflocks_bodydata"
# Pre-calculate in CPU to avoid errors
tbn_space_to_world_cpu = torch.tensor([[1.,0.,0.],[0.,0.,1.],[0.,-1.,0.]]).float()
torch.set_num_threads(4)

# --- UTILS ---
def interpolate_tbn(barys, vertex_idxs, v_tangents, v_bitangents, v_normals):
    nr_positions = barys.shape[0]
    sampled_tangents = v_tangents[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3)
    weighted_tangents = sampled_tangents * barys.reshape(nr_positions,3,1)
    point_tangents = weighted_tangents.sum(axis=1)
    norm = np.linalg.norm(point_tangents, axis=-1, keepdims=True)
    point_tangents = point_tangents / (norm + 1e-8)

    sampled_normals = v_normals[vertex_idxs.reshape(-1),:].reshape(nr_positions,3,3)
    weighted_normals = sampled_normals * barys.reshape(nr_positions,3,1)
    point_normals = weighted_normals.sum(axis=1)
    norm = np.linalg.norm(point_normals, axis=-1, keepdims=True)
    point_normals = point_normals / (norm + 1e-8)

    point_bitangents = np.cross(point_normals, point_tangents)
    norm = np.linalg.norm(point_bitangents, axis=-1, keepdims=True)
    point_bitangents = point_bitangents / (norm + 1e-8)

    point_tangents = np.cross(point_bitangents, point_normals)
    norm = np.linalg.norm(point_tangents, axis=-1, keepdims=True)
    point_tangents = point_tangents / (norm + 1e-8)
    return point_tangents, point_bitangents, point_normals

def tbn_space_to_world_gpu_native(root_uv, strands_positions, scalp_mesh_data):
    """Truly GPU-native TBN to World transformation to avoid CPU-GPU sync overhead."""
    device = strands_positions.device
    dtype = strands_positions.dtype
    
    # Ensure all required mesh data is on the correct device/dtype
    def to_torch(x, is_index=False):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=torch.long if is_index else dtype)
        return torch.from_numpy(x).to(device=device, dtype=torch.long if is_index else dtype)

    # Note: These are large maps, but they are only moved once if the dictionary is reused
    scalp_vertex_idxs_map = to_torch(scalp_mesh_data["vertex_idxs_map"], is_index=True)
    scalp_bary_map = to_torch(scalp_mesh_data["bary_map"])
    mesh_v_tangents = to_torch(scalp_mesh_data["v_tangents"])
    mesh_v_bitangents = to_torch(scalp_mesh_data["v_bitangents"])
    mesh_v_normals = to_torch(scalp_mesh_data["v_normals"])
    scalp_v = to_torch(scalp_mesh_data["verts"])
    
    tex_size = scalp_vertex_idxs_map.shape[0]
    
    # 1. Get pixel indices
    # root_uv should be on device already
    pixel_indices = torch.floor(root_uv * tex_size).long()
    pixel_indices = torch.clamp(pixel_indices, 0, tex_size - 1)
    
    # 2. Indexing on GPU
    # vertex_idxs must be long for further indexing
    vertex_idxs = scalp_vertex_idxs_map[pixel_indices[:, 0], pixel_indices[:, 1], :].long() # [N, 3]
    barys = scalp_bary_map[pixel_indices[:, 0], pixel_indices[:, 1], :] # [N, 3]
    
    # 3. Interpolate TBN (Ported interpolate_tbn to torch)
    nr_positions = barys.shape[0]
    
    # Sample and weight tangents
    # Ensure indices are long
    v_idxs_flat = vertex_idxs.reshape(-1).long()
    sampled_tangents = mesh_v_tangents[v_idxs_flat].reshape(nr_positions, 3, 3)
    weighted_tangents = sampled_tangents * barys.reshape(nr_positions, 3, 1)
    point_tangents = weighted_tangents.sum(dim=1)
    point_tangents = point_tangents / (torch.norm(point_tangents, dim=-1, keepdim=True) + 1e-8)

    # Sample and weight normals
    sampled_normals = mesh_v_normals[v_idxs_flat].reshape(nr_positions, 3, 3)
    weighted_normals = sampled_normals * barys.reshape(nr_positions, 3, 1)
    point_normals = weighted_normals.sum(dim=1)
    point_normals = point_normals / (torch.norm(point_normals, dim=-1, keepdim=True) + 1e-8)

    # Compute bitangents
    point_bitangents = torch.cross(point_normals, point_tangents, dim=-1)
    point_bitangents = point_bitangents / (torch.norm(point_bitangents, dim=-1, keepdim=True) + 1e-8)

    # Recompute orthogonal tangents
    point_tangents = torch.cross(point_bitangents, point_normals, dim=-1)
    point_tangents = point_tangents / (torch.norm(point_tangents, dim=-1, keepdim=True) + 1e-8)

    # 4. Basis Change
    strands_tbn = torch.stack((point_tangents, point_bitangents, point_normals), dim=2) # [N, 3, 3]
    indices_tbn = torch.tensor([0, 2, 1], device=device, dtype=torch.long)
    strands_tbn = torch.index_select(strands_tbn, 2, indices_tbn)
    strands_tbn[..., 0] = -strands_tbn[..., 0]
    
    # Apply rotation to positions
    orig_points = torch.matmul(strands_tbn, strands_positions.transpose(1, 2)).transpose(1, 2)
    
    # 5. Get root positions in world space
    sampled_v = scalp_v[v_idxs_flat].reshape(nr_positions, 3, 3)
    weighted_v = sampled_v * barys.reshape(nr_positions, 3, 1)
    roots_positions = weighted_v.sum(dim=1)
    
    # Final world points
    return orig_points + roots_positions[:, None, :]

def tbn_space_to_world_cpu_safe(root_uv, strands_positions, scalp_mesh_data):
    target_device = strands_positions.device
    target_dtype = torch.float32 
    scalp_index_map = scalp_mesh_data["index_map"]
    scalp_vertex_idxs_map = scalp_mesh_data["vertex_idxs_map"]
    scalp_bary_map = scalp_mesh_data["bary_map"]
    mesh_v_tangents = scalp_mesh_data["v_tangents"]
    mesh_v_bitangents = scalp_mesh_data["v_bitangents"]
    mesh_v_normals = scalp_mesh_data["v_normals"]
    scalp_v = scalp_mesh_data["verts"]
    tex_size = scalp_vertex_idxs_map.shape[0]
    root_uv_np = root_uv.cpu().numpy() if torch.is_tensor(root_uv) else root_uv
    pixel_indices = np.floor(root_uv_np * tex_size).astype(int)
    pixel_indices = np.clip(pixel_indices, 0, tex_size - 1)
    vertex_idxs = scalp_vertex_idxs_map[pixel_indices[:, 0], pixel_indices[:, 1], :]
    barys = scalp_bary_map[pixel_indices[:, 0], pixel_indices[:, 1], :]
    root_tangent, root_bitangent, root_normal = interpolate_tbn(barys, vertex_idxs, mesh_v_tangents, mesh_v_bitangents, mesh_v_normals)
    strands_tbn_np = np.stack((root_tangent, root_bitangent, root_normal), axis=2)
    strands_tbn = torch.as_tensor(strands_tbn_np, device=target_device, dtype=target_dtype)
    indices_tbn = torch.tensor([0, 2, 1], device=target_device, dtype=torch.long)
    strands_tbn = torch.index_select(strands_tbn, 2, indices_tbn)
    strands_tbn[..., 0] = -strands_tbn[..., 0]
    orig_points = torch.matmul(strands_tbn, strands_positions.permute(0, 2, 1)).permute(0, 2, 1)
    nr_positions = vertex_idxs.shape[0]
    sampled_v_np = scalp_v[vertex_idxs.reshape(-1), :].reshape(nr_positions, 3, 3)
    sampled_v = torch.as_tensor(sampled_v_np, device=target_device, dtype=target_dtype)
    barys_tensor = torch.as_tensor(barys, device=target_device, dtype=target_dtype)
    weighted_v = sampled_v * barys_tensor.reshape(nr_positions, 3, 1)
    roots_positions = weighted_v.sum(dim=1)
    strds_points = orig_points + roots_positions[:, None, :]
    return strds_points

def force_cleanup():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

# CROP FACE 2.8x (Correct Reference)
def crop_face(image, face_landmarks, output_size, crop_size_multiplier=2.8):
    h, w, _ = image.shape
    xs = [l.x for l in face_landmarks]; ys = [l.y for l in face_landmarks]
    min_x, max_x = min(xs) * w, max(xs) * w
    min_y, max_y = min(ys) * h, max(ys) * h
    cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
    face_w, face_h = max_x - min_x, max_y - min_y
    size = max(face_w, face_h) * crop_size_multiplier # 2.8x
    x1 = int(cx - size / 2); y1 = int(cy - size / 2)
    x2 = int(cx + size / 2); y2 = int(cy + size / 2)
    pad_l = max(0, -x1); pad_t = max(0, -y1)
    pad_r = max(0, x2 - w); pad_b = max(0, y2 - h)
    if any([pad_l, pad_t, pad_r, pad_b]):
        image = np.pad(image, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode='constant')
        x1 += pad_l; y1 += pad_t; x2 += pad_l; y2 += pad_t
    crop = image[y1:y2, x1:x2]
    try: return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
    except: return cv2.resize(image, (output_size, output_size))

class Mediapipe:
    def __init__(self):
        asset_path = 'inference/assets/face_landmarker_v2_with_blendshapes.task'
        if not os.path.exists(asset_path): asset_path = os.path.join(os.getcwd(), asset_path)
        base_options = python.BaseOptions(model_asset_path=asset_path)
        options = vision.FaceLandmarkerOptions(base_options=base_options, output_face_blendshapes=True,
                                             output_facial_transformation_matrixes=True, num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
    def run(self, image):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        res = self.detector.detect(mp_image)
        return (res.face_blendshapes[0], res.face_landmarks[0]) if res.face_landmarks else (None, None)

class DiffLocksInference():
    def __init__(self, path_ckpt_strandcodec, path_config_difflocks, path_ckpt_difflocks,
                 path_ckpt_rgb2material=None, cfg_val=1.0, nr_iters_denoise=100, nr_chunks_decode=150):
        self.nr_chunks_decode_strands = nr_chunks_decode
        self.nr_iters_denoise = nr_iters_denoise
        self.cfg_val = cfg_val
        self.paths = { 'codec': path_ckpt_strandcodec, 'config': path_config_difflocks, 'diff': path_ckpt_difflocks, 'mat': path_ckpt_rgb2material }
        self.mediapipe_img = Mediapipe()
        self.normalization_dict = DiffLocksDataset.get_normalization_data()
        scalp_path = os.path.join(DEFAULT_BODY_DATA_DIR, "scalp.ply")
        if not os.path.exists(scalp_path): scalp_path = "data_loader/difflocks_bodydata/scalp.ply"
        self.scalp_trimesh, self.scalp_mesh_data = DiffLocksDataset.compute_scalp_data(scalp_path)
        self.norm_dict_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.normalization_dict.items()}
        self.mesh_data_cpu = {k: v.cpu() if torch.is_tensor(v) else v for k, v in self.scalp_mesh_data.items()}
        self.tbn_space_to_world = tbn_space_to_world_cpu_safe

    # NOW A GENERATOR (YIELD)
    @torch.inference_mode()
    def rgb2hair(self, rgb_img, out_path=None, cfg_val=None, progress=None):
        if out_path: os.makedirs(out_path, exist_ok=True)
        actual_cfg = cfg_val if cfg_val is not None else self.cfg_val
        
        if progress is not None: progress(0, desc="Initializing...")
        # INITIAL LOG
        yield "log", f"⚙️ Configuration: CFG={actual_cfg} | Steps={self.nr_iters_denoise} | Precision=float32"

        try:
            # 1. GEOMETRY
            yield "status", "👤 1/5: Detecting Face and Geometry..."
            if progress is not None: progress(0.05, desc="Detecting Face...")
            frame = (rgb_img.permute(0,2,3,1).squeeze(0)*255).byte().cpu().numpy()
            _, lms = self.mediapipe_img.run(frame)
            if not lms: 
                yield "error", "No face detected in the image."
                return
            
            cropped_face = crop_face(frame, lms, 770)
            del frame
            rgb_img_gpu = torch.tensor(cropped_face).to("cuda" if torch.cuda.is_available() else "cpu").permute(2,0,1).unsqueeze(0).float()/255.0
            rgb_img_cpu = rgb_img_gpu.cpu().clone() # Backup for final save
            yield "log", "✅ Face detected and cropped (Zoom 2.8x)"
            
            # 2. DINO
            yield "status", "🦖 2/5: Extracting Features (DINOv2)..."
            if progress is not None: progress(0.1, desc="Extracting DINO features...")
            
            # Load DINOv2 model with caching
            from inference.load_dinov2 import load_dinov2
            dinov2, tf = load_dinov2(device=rgb_img_gpu.device)
            
            # Extract features
            with torch.no_grad():
                out = dinov2.forward_features(tf(rgb_img_gpu))
            patch = out["x_norm_patchtokens"]
            cls_tok = out["x_norm_clstoken"]
            h = w = int(patch.shape[1]**0.5)
            patch_emb = patch.reshape(patch.shape[0], h, w, -1).permute(0, 3, 1, 2).contiguous()
            
            patch_emb_cpu = patch_emb.cpu().clone()
            cls_tok_cpu = cls_tok.cpu().clone()
            del dinov2, out, patch, cls_tok, patch_emb, rgb_img_gpu
            force_cleanup()
            yield "log", "✅ Embeddings successfully generated"
            
            # 3. DIFFUSION
            yield "status", "🌫️ 3/5: Diffusion (Generating Hair)..."
            yield "log", "⏳ Loading diffusion model..."
            conf = K.config.load_config(self.paths['config'])
            model = K.config.make_denoiser_wrapper(conf)(K.config.make_model(conf).to("cuda" if torch.cuda.is_available() else "cpu"))
            model.float() # Force float32
            # DEBUG PATCH

            if not os.path.exists(self.paths['diff']):
                print(f"ERROR: File not found: {self.paths['diff']}")
            
            try:
                ckpt = torch.load(self.paths['diff'], map_location='cpu', weights_only=False)
            except AttributeError as e:
                if "'NoneType' object has no attribute 'seek'" in str(e):
                    print("CRITICAL ERROR: torch.load failed with NoneType seek error.")
                    print("This usually means the file is corrupt or empty.")
                    # Try to delete it so it redownloads?
                    # os.remove(self.paths['diff'])
                    raise RuntimeError(f"Corrupt model file: {self.paths['diff']}. Please delete it and restart to redownload.") from e
                raise e
            # END DEBUG PATCH
            model.inner_model.load_state_dict(ckpt['model_ema'])
            model.float() # Force float32 again after loading weights (in case weights were half)
            print(f"💎 Using Full Precision (float32) - VRAM: {cfg.vram_gb:.1f}GB")
            del ckpt; force_cleanup()
            model.eval(); model.inner_model.condition_dropout_rate = 0.0
            
            # Embeddings precision
            cls_tok_gpu = cls_tok_cpu.to("cuda" if torch.cuda.is_available() else "cpu").float()
            patch_emb_gpu = patch_emb_cpu.to("cuda" if torch.cuda.is_available() else "cpu").float()

            extra = {'latents_dict': {"dinov2": {"cls_token": cls_tok_gpu, "final_latent": patch_emb_gpu}}}
            
            # DEBUG DTYPES
            print(f"DEBUG: Model dtype: {next(model.parameters()).dtype}")
            print(f"DEBUG: cls_tok_gpu dtype: {cls_tok_gpu.dtype}")
            print(f"DEBUG: patch_emb_gpu dtype: {patch_emb_gpu.dtype}")
            
            yield "log", f"🎨 Starting sampling ({self.nr_iters_denoise} steps)... This will take a few minutes."
            
            def p_callback(info):
                i = info['i']
                if progress is not None:
                    progress(0.2 + 0.6 * (i / self.nr_iters_denoise), desc=f"Diffusion {i}/{self.nr_iters_denoise}")
                
                # Yield logs to app.py every 10 steps
                if i % 10 == 0:
                    # We can't 'yield' from inside a callback function directly in Python, 
                    # but we can print for the terminal and let the outer loop handle the yielding.
                    print(f"🔄 Diffusion: Step {i}/{self.nr_iters_denoise} (sigma={info['sigma']:.4f})")
            
            # Sampling (No autocast to match reference)
            # Use yielding version for progress updates
            scalp = None
            last_log_step = -1
            for x_step, i, sigma in sample_images_cfg_yield(1, actual_cfg, [-1., 10000.], model, conf['model'], self.nr_iters_denoise, extra, callback=p_callback):
                scalp = x_step
                
                # Manually yield to the generator every 10 steps to update UI
                if i % 10 == 0 and i != last_log_step:
                    yield "log", f"🔄 Diffusion Step {i}/{self.nr_iters_denoise}..."
                    last_log_step = i
            
            # NaN Check
            if torch.isnan(scalp).any():
                yield "error", "Model generated NaN values. This can occur due to precision issues (float16) or high CFG. Try lowering CFG or restarting."
                return

            # CRITICAL: Convert to float32 for decoder compatibility
            scalp_cpu = scalp.cpu().float().clone()
            sigma_data = conf['model']["sigma_data"]
            del model, scalp, extra, conf; force_cleanup()
            
            density = (scalp_cpu[:,-1:]*(0.5/sigma_data)+0.5).clamp(0,1)
            density[density<0.02] = 0.0
            
            # Debug logging for density map
            d_sum = density.sum().item()
            d_max = density.max().item()
            print(f"DEBUG img2hair: density_map sum = {d_sum:.2f}, max = {d_max:.4f}")


            if density.sum() == 0: 
                yield "error", "Model generated an empty density map. Try a different image or adjust CFG."
                return
            yield "log", f"✅ Neural texture generated (density sum: {density.sum():.1f})"
            
            # 4. DECODING
            yield "status", "🧬 4/5: Decoding in 3D (GPU)..."
            yield "log", "⏳ Loading Strand VAE/Codec..."
            if progress is not None: progress(0.85, desc="Decoding strands...")
            
            # Move codec to GPU for speed
            codec_device = "cuda" if torch.cuda.is_available() else "cpu"
            codec = StrandCodec(do_vae=False, decode_type="dir", nr_verts_per_strand=256).to(codec_device)
            codec.load_state_dict(torch.load(self.paths['codec'], map_location=codec_device, weights_only=False))
            codec.eval()
            
            # Move normalization dict to GPU
            norm_dict_gpu = {k: v.to(codec_device) if torch.is_tensor(v) else v for k, v in self.normalization_dict.items()}
            mesh_data_gpu = {k: v.to(codec_device) if torch.is_tensor(v) else v for k, v in self.scalp_mesh_data.items()}
            
            yield "log", f"⏳ Processing {self.nr_chunks_decode_strands} geometry chunks (GPU Accelerated)..."
            
            # Ensure GPU for decoder
            scalp_texture = scalp_cpu[:,0:-1].to(codec_device).float()
            density_f32 = density.to(codec_device).float()

            def decoding_callback(i, total):
                if i % 10 == 0:
                    print(f"🧬 Decoding: Chunk {i}/{total}")
            
            # Use the native GPU function for speed and stability
            tbn_func = tbn_space_to_world_gpu_native if torch.cuda.is_available() else tbn_space_to_world_cpu_safe
            
            # Call the function with the appropriate device-aware function
            strands, _ = sample_strands_from_scalp_with_density(
                scalp_texture, density_f32, codec, norm_dict_gpu, 
                mesh_data_gpu, tbn_func, self.nr_chunks_decode_strands,
                callback=decoding_callback)
            
            if strands is None or strands.shape[0] == 0:
                yield "error", "Decoding failed: no strands generated. Check the density map sum."
                return

            yield "log", f"✅ 3D Geometry built: {strands.shape[0]} strands generated"
            
            # 5. SAVE
            yield "status", "💾 5/5: Saving Files..."
            if progress is not None: progress(0.95, desc="Saving results...")
            if out_path and strands is not None:
                positions = strands.cpu().numpy()
                npz_full_path = os.path.join(out_path, "difflocks_output_strands.npz")
                npz_preview_path = os.path.join(out_path, "difflocks_output_strands_preview.npz")
                
                # Save full version
                np.savez_compressed(npz_full_path, positions=positions)
                
                # Save preview version (optimized for 3D plot)
                try:
                    # Optimized for 3D preview: 1000 strands and 24 points per strand
                    num_strands = positions.shape[0]
                    points_per_strand = positions.shape[1]
                    
                    # 1. Target exactly ~1000 strands
                    strand_step = max(1, num_strands // 1000)
                    
                    # 2. Reduce points to 24 per strand
                    target_points = 24
                    point_step = max(1, points_per_strand // target_points)
                    
                    preview_positions = positions[::strand_step, ::point_step, :]
                    
                    np.savez_compressed(npz_preview_path, positions=preview_positions)
                    yield "log", f"✅ Optimized preview: {preview_positions.shape[0]} strands, {preview_positions.shape[1]} points"
                except Exception as e:
                    yield "log", f"⚠️ Error creating optimized preview: {e}"
                
                # Copy input image
                cv2.imwrite(os.path.join(out_path, "input_cropped.png"), cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                
            yield "status", "✅ Process completed!"
            if progress is not None: progress(1.0, desc="Completed")
            yield "result", strands, None

        except Exception as e:
            traceback.print_exc()
            yield "error", f"Inference error: {str(e)}"
        finally:
            force_cleanup()

    def file2hair(self, fpath, out, cfg_val=None, progress=None):
        img = cv2.imread(fpath)
        if img is None: raise FileNotFoundError(f"{fpath}")
        rgb = torch.tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).to("cuda" if torch.cuda.is_available() else "cpu").permute(2,0,1).unsqueeze(0).float()/255.
        # Propagate the generator
        yield from self.rgb2hair(rgb, out, cfg_val=cfg_val, progress=progress)
