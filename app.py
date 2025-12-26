# app.py - DiffLocks Studio (Universal V2 - Kaggle Style)
# ============================================================================
# Improvements:
# - Premium Kaggle-style UI with Dual Progress Bars
# - Real-time Log Redirection (Stdout/Stderr to Debug Console)
# - Phase-based Progress Tracking with Time Estimation
# - Robust NaN Protection and float32 fallback for mid-range GPUs
# ============================================================================

import os
import sys
import time
import shutil
import subprocess
import gc
import zipfile
import traceback
import warnings
import logging
import threading
import base64
import builtins
from html import escape

import gradio as gr
import numpy as np
import torch
from pathlib import Path
import plotly.graph_objects as go

# --- 0. SILENCING ---
warnings.filterwarnings("ignore")
for logger_name in ["natten", "matplotlib", "PIL", "mediapipe", "absl", "tensorflow"]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# --- 1. PLATFORM SETUP ---
try:
    from platform_config import cfg
except ImportError:
    sys.path.append(".")
    from platform_config import cfg

if str(cfg.repo_dir) not in sys.path:
    sys.path.append(str(cfg.repo_dir))

# Detect ZeroGPU
try:
    import spaces
    HAS_ZEROGPU = True
except ImportError:
    HAS_ZEROGPU = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CPU = (DEVICE == "cpu")

# --- 2. KAGGLE UI CONSTANTS ---
PHASES = [
    ("init", "🚀 Initializing", 5, 0.00, 0.01),
    ("1/5", "📸 Preprocessing image", 10, 0.01, 0.03),
    ("2/5", "🔍 Extracting features", 20, 0.03, 0.08),
    ("3/5", "✨ Running diffusion", 180, 0.08, 0.55), # GPU: 3 min (optimized from 6)
    ("4/5", "🧶 Decoding strands", 120, 0.55, 0.75), # GPU: 2 min (optimized from 3)
    ("5/5", "🏁 Finalizing inference", 5, 0.75, 0.78),
    ("preview_2d", "🎨 Creating 2D preview", 10, 0.78, 0.82),
    ("preview_3d", "🎨 Creating Interactive 3D", 15, 0.82, 0.88),
    ("obj_export", "📦 Exporting OBJ", 30, 0.88, 0.94),
    ("blender", "🟧 Blender export", 60, 0.94, 1.00),
]

# --- 3. LOG CAPTURE & PROGRESS TRACKER ---

class VerboseLogCapture:
    def __init__(self):
        self.logs = []
        self.lock = threading.Lock()
        self.original_stdout = None
        self.original_stderr = None
        self.capturing = False
        
    def write(self, msg):
        if self.original_stdout:
            try: self.original_stdout.write(msg)
            except: pass
        if self.capturing:
            # Filter out redundant model internal prints
            skip_terms = ["cfg_val_cur", "for sigma", "tensor(", "width for this lvl", "initializing LVL", "making up cross layer"]
            with self.lock:
                # Handle multi-line messages
                lines = msg.splitlines()
                ts = time.strftime('%H:%M:%S')
                for line in lines:
                    line_clean = line.strip()
                    if line_clean and not any(term in line_clean for term in skip_terms):
                        # Clean up tqdm lines if they leak
                        if "|" in line_clean and "%" in line_clean and "[" in line_clean:
                            continue
                        self.logs.append(f"[{ts}] {line_clean}")
                
                if len(self.logs) > 500:
                    self.logs = self.logs[-400:]
        
    def flush(self):
        if self.original_stdout:
            try: self.original_stdout.flush()
            except: pass
    
    def get_logs(self):
        with self.lock: return list(self.logs)
    
    def add_log(self, msg):
        with self.lock:
            ts = time.strftime('%H:%M:%S')
            self.logs.append(f"[{ts}] {msg}")
    
    def start(self):
        self.capturing = True
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
        return self
        
    def stop(self):
        self.capturing = False
        if self.original_stdout: sys.stdout = self.original_stdout
        if self.original_stderr: sys.stderr = self.original_stderr

class ProgressTracker:
    def __init__(self, is_cpu=False):
        self.start_time = time.time()
        self.current_phase_index = 0
        self.current_phase_start_time = time.time()
        self.phases = []
        for p in PHASES:
            p_list = list(p)
            if is_cpu and "3/5" in str(p_list[0]):
                p_list[2] = 57600 # 16 hours
            self.phases.append(p_list)
        self.total_estimated = sum(p[2] for p in self.phases)
        
    def set_phase(self, phase_id):
        for i, phase in enumerate(self.phases):
            if phase[0] in str(phase_id).lower():
                self.current_phase_index = i
                self.current_phase_start_time = time.time()
                self.current_phase_id = phase[0] # Keep track of ID
                return
    
    def get_phase_name(self):
        # Use the stored ID or search
        phase_id = getattr(self, 'current_phase_id', self.phases[self.current_phase_index][0])
        for pid, name, _, _, _ in self.phases:
            if pid == phase_id:
                return name
        return "Processing"

    def get_progress(self, is_complete=False):
        if is_complete or self.current_phase_index >= len(self.phases):
            return 100, 0, "Complete", 100, 0
            
        current_phase = self.phases[self.current_phase_index]
        phase_id, phase_name, phase_duration, phase_start_pct, phase_end_pct = current_phase
        
        phase_elapsed = time.time() - self.current_phase_start_time
        
        # Calculate progress within phase (0-100%)
        phase_internal_progress = min((phase_elapsed / phase_duration) * 100, 99) if phase_duration > 0 else 0
        phase_remaining = max(phase_duration - phase_elapsed, 0)
        
        # Calculate total progress using the percentage ranges defined in PHASES
        # Total progress = start_pct + (end_pct - start_pct) * (internal_progress / 100)
        total_progress = (phase_start_pct + (phase_end_pct - phase_start_pct) * (phase_internal_progress / 100)) * 100
        total_progress = min(max(total_progress, 0), 99) # Cap at 99% until fully complete
        
        # Estimate total remaining time
        remaining_phases_time = sum(self.phases[i][2] for i in range(self.current_phase_index + 1, len(self.phases)))
        total_remaining = phase_remaining + remaining_phases_time
        
        return total_progress, total_remaining, phase_name, phase_internal_progress, phase_remaining

# --- 4. HTML RENDERING FUNCTIONS ---

def format_time(seconds):
    if seconds < 0: seconds = 0
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"

def render_debug_console(logs_list):
    if not logs_list: logs_list = ["[Waiting for process to start...]"]
    lines_html = []
    # Take the last 200 lines to keep it performant
    for line in logs_list[-200:]: 
        escaped = escape(line)
        color = "#d4d4d8"
        if any(x in line for x in ["❌", "ERROR", "ERR]"]): color = "#f87171"
        elif any(x in line for x in ["✅", "SUCCESS", "COMPLETED"]): color = "#34d399"
        elif any(x in line for x in ["⚠️", "WARNING"]): color = "#fbbf24"
        elif any(x in line for x in ["🔄", "Diffusion:"]): color = "#818cf8"
        elif any(x in line for x in ["🟧", "[Blender]"]): color = "#fb923c"
        lines_html.append(f'<div style="color: {color}; margin: 2px 0; overflow-anchor: none;">{escaped}</div>')
    
    content = "".join(lines_html)
    
    # Stable container with auto-scroll logic
    return f'''
    <div id="debug-console-container" style="
        background-color: #09090b; 
        border: 1px solid #27272a; 
        border-radius: 8px; 
        padding: 12px; 
        font-family: 'JetBrains Mono', 'Fira Code', monospace; 
        font-size: 12px; 
        line-height: 1.5; 
        height: 450px; 
        overflow-y: scroll; 
        position: relative;
        display: flex;
        flex-direction: column;
    ">
        <div style="flex: 1 1 auto; min-height: 0;"></div>
        <div id="debug-console-content" style="flex: 0 0 auto;">{content}</div>
        <div id="debug-console-anchor" style="height: 1px; min-height: 1px; flex: 0 0 auto;"></div>
        
        <img src="data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" 
             style="display:none;" 
             onload="
                (function() {{
                    var el = document.getElementById('debug-console-container');
                    if (el) {{
                        var threshold = 150;
                        var isAtBottom = (el.scrollHeight - el.scrollTop - el.clientHeight) < threshold;
                        // Si estamos cerca del final o es la primera vez (scrollTop 0), forzamos scroll
                        if (isAtBottom || el.scrollTop === 0) {{
                            var scrollDown = function() {{ el.scrollTop = el.scrollHeight; }};
                            scrollDown();
                            setTimeout(scrollDown, 20);
                            setTimeout(scrollDown, 100);
                        }}
                    }}
                }})();
             ">
    </div>
    '''

def render_image_html(image_path, title="2D Preview"):
    if not image_path or not Path(image_path).exists():
        return f'''
        <div style="background-color: #18181b; border: 1px solid #3f3f46; border-radius: 8px; padding: 40px; text-align: center; color: #a1a1aa; font-size: 14px;">
            <div style="font-size: 48px; margin-bottom: 10px;">🖼️</div>
            <div>No preview available</div>
        </div>
        '''
    with open(image_path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    return f'''
    <div style="background-color: #18181b; border: 1px solid #3f3f46; border-radius: 8px; padding: 12px; text-align: center; display: flex; justify-content: center; align-items: center;">
        <img src="data:image/png;base64,{img_data}" style="width: 66%; height: auto; border-radius: 6px;" alt="{title}"/>
    </div>
    '''

def create_dual_progress_html(total_pct, total_rem, step_name, step_pct, step_rem, status_type="info"):
    colors = {"info": "#818cf8", "success": "#34d399", "error": "#f87171", "warning": "#fbbf24"}
    main_color = colors.get(status_type, "#818cf8")
    return f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #18181b 0%, #27272a 100%); border-radius: 12px; border: 1px solid {main_color}44; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
        <div style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <span style="font-size: 20px; font-weight: 700; color: #fafafa;">💇‍♀️ Total Progress</span>
                    <span style="color: #a1a1aa; font-size: 14px;">~{format_time(total_rem)} remaining</span>
                </div>
                <span style="background: {main_color}33; color: {main_color}; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 700;">{int(total_pct + 0.5)}%</span>
            </div>
            <div style="background: #3f3f46; height: 12px; border-radius: 6px; overflow: hidden;">
                <div style="width: {total_pct}%; height: 100%; background: linear-gradient(90deg, {main_color}, {main_color}cc); transition: width 0.5s ease-in-out;"></div>
            </div>
        </div>
        <div style="background: #27272a; border: 1px solid #3f3f46; border-radius: 10px; padding: 15px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 15px; font-weight: 600; color: #e4e4e7;">{step_name}</span>
                    <span style="color: #71717a; font-size: 12px;">~{format_time(step_rem)}</span>
                </div>
                <span style="color: #a5b4fc; font-size: 13px; font-weight: 600;">{int(step_pct + 0.5)}%</span>
            </div>
            <div style="background: #3f3f46; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="width: {step_pct}%; height: 100%; background: linear-gradient(90deg, #a5b4fc, #818cf8); transition: width 0.3s ease-in-out;"></div>
            </div>
        </div>
    </div>
    '''

def create_complete_html():
    return f'''
    <div style="background: rgba(16, 185, 129, 0.1); border: 1px solid #10b981; border-radius: 8px; padding: 12px;">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 20px;">✅</span>
                <div>
                    <div style="color: #10b981; font-weight: 700; font-size: 14px;">100% COMPLETE</div>
                    <div style="color: #6ee7b7; font-size: 12px;">All assets generated and ready for download.</div>
                </div>
            </div>
        </div>
    </div>
    '''

def create_error_html(error_msg):
    return f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #18181b 0%, #27272a 100%); border-radius: 12px; border: 1px solid #f8717166; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 28px;">❌</span>
            <span style="font-size: 16px; font-weight: 600; color: #f87171;">{escape(str(error_msg)[:200])}</span>
        </div>
    </div>
    '''

# --- 5. MODEL LOADER ---
from inference.img2hair import DiffLocksInference

def download_checkpoints_hf():
    """Auto-download checkpoints if on HF Spaces and missing."""
    if cfg.platform != 'huggingface': return
    
    ckpt_dir = cfg.checkpoints_dir
    diffusion_dir = ckpt_dir / "difflocks_diffusion"
    vae_dir = ckpt_dir / "strand_vae"
    
    # Check if we already have the main files
    if list(diffusion_dir.glob("scalp_*.pth")) and list(vae_dir.glob("strand_codec.pt")):
        return

    print("🚀 [HF SPACES] Downloading missing checkpoints...")
    
    # Try official Meshcapade if credentials available
    mesh_user = os.environ.get("MESH_USER")
    mesh_pass = os.environ.get("MESH_PASS")
    
    if mesh_user and mesh_pass:
        print("🔐 [HF SPACES] Using Meshcapade credentials from Environment Variables...")
        # Here we would call the official download logic
        # For now, it will use the download_checkpoints.py script if it was more complete
        pass

    # Fallback to current HF Repo (will require token if private)
    try:
        from huggingface_hub import snapshot_download
        token = os.environ.get("HF_TOKEN")
        snapshot_download(
            repo_id="arqdariogomez/difflocks-assets-hybrid",
            repo_type="dataset",
            allow_patterns=["checkpoints/*", "assets/*"],
            local_dir=str(cfg.repo_dir),
            local_dir_use_symlinks=False,
            token=token if token else None
        )
        print("✅ [HF SPACES] Checkpoints downloaded successfully.")
    except Exception as e:
        print(f"❌ [HF SPACES] Error downloading checkpoints: {e}")
        print("💡 Ensure MESH_USER/MESH_PASS or HF_TOKEN are set in Space Secrets.")

# Run download check before initializing file lists
download_checkpoints_hf()

# Search for checkpoints in unified paths
checkpoints_dir = cfg.checkpoints_dir
ckpt_files = list((checkpoints_dir / "difflocks_diffusion").glob("scalp_*.pth"))
vae_files = list((checkpoints_dir / "strand_vae").glob("strand_codec.pt"))
conf_path = cfg.configs_dir / "config_scalp_texture_conditional.json"

print(f"[{cfg.platform.upper()}] Searching for checkpoints in: {checkpoints_dir.absolute()}")
print(f"Checkpoint files found: {[str(f) for f in ckpt_files]}")
print(f"VAE files found: {[str(f) for f in vae_files]}")

# Check if files exist
if not ckpt_files:
    print("ERROR: No checkpoint files found (scalp_*.pth)")
    print(f"Searching in: {checkpoints_dir / 'difflocks_diffusion'}")
    print(f"Directory contents: {list((checkpoints_dir / 'difflocks_diffusion').glob('*'))}")

if not vae_files:
    print("ERROR: No VAE files found (strand_codec.pt)")
    print(f"Searching in: {checkpoints_dir / 'strand_vae'}")
    print(f"Directory contents: {list((checkpoints_dir / 'strand_vae').glob('*'))}")

model = None

def load_model():
    global model, ckpt_files, vae_files
    if model is not None: return

    if not ckpt_files or not vae_files:
        # Re-check files after download attempt just in case
        ckpt_files = list((checkpoints_dir / "difflocks_diffusion").glob("scalp_*.pth"))
        vae_files = list((checkpoints_dir / "strand_vae").glob("strand_codec.pt"))
        
        if not ckpt_files or not vae_files:
            raise FileNotFoundError(f"Missing checkpoints! Search path: {checkpoints_dir.absolute()}")
    
    print(f"Loading Model on {DEVICE} (Precision=float32): {ckpt_files[0].name}")
    model = DiffLocksInference(str(vae_files[0]), str(conf_path), str(ckpt_files[0]), DEVICE)
    print("✅ Model Loaded!")

# --- 6. UTILITY FUNCTIONS ---

def generate_preview_2d(npz_path, output_dir, log_capture=None):
    try:
        import matplotlib.pyplot as plt
        data = np.load(npz_path)
        positions = data['positions']
        n_strands = positions.shape[0]
        step = max(1, n_strands // 20000)
        pts = positions.reshape(-1, 3)
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        
        # Rotate to match Kaggle view
        rx, ry, rz = x, z, y
        theta = np.radians(180)
        c, s = np.cos(theta), np.sin(theta)
        final_x = rx * c + ry * s
        final_y = -rx * s + ry * c
        final_z = rz
        
        rotated = np.stack([final_x, final_y, final_z], axis=1).reshape(n_strands, -1, 3)
        subset = rotated[::step, ::3, :]
        sx = subset[:, :, 0].flatten()
        sy = subset[:, :, 1].flatten()
        sz = subset[:, :, 2].flatten()
        
        mask = np.abs(sz) > 0.001
        sx, sy, sz = sx[mask], sy[mask], sz[mask]
        
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(10, 10), facecolor='#18181b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#18181b')
        
        idx = np.argsort(sy)
        d_min, d_max = sy.min(), sy.max()
        depth_norm = (sy[idx] - d_min) / (d_max - d_min + 1e-8)
        
        ax.scatter(sx[idx], sz[idx], c=depth_norm, cmap='copper', s=0.5, alpha=0.7, linewidths=0)
        ax.axis('off')
        ax.set_aspect('equal')
        
        plt.tight_layout(pad=0)
        preview_path = Path(output_dir) / "preview_2d.png"
        fig.savefig(preview_path, facecolor='#18181b', bbox_inches='tight', dpi=120, pad_inches=0.1)
        plt.close(fig)
        
        del positions, pts, rotated, subset
        gc.collect()
        return str(preview_path)
    except Exception as e:
        if log_capture: log_capture.add_log(f"❌ 2D Preview Error: {e}")
        return None

def generate_preview_3d(npz_path, log_capture=None):
    try:
        if not Path(npz_path).exists():
            if log_capture: log_capture.add_log(f"⚠️ 3D Preview: File not found at {npz_path}")
            return None
            
        if log_capture: log_capture.add_log(f"🎨 Interactive 3D: Loading {Path(npz_path).name}...")
        data = np.load(npz_path)
        if 'positions' not in data:
            if log_capture: log_capture.add_log(f"⚠️ 3D Preview: 'positions' not in NPZ")
            return None
            
        positions = data['positions']
        if len(positions.shape) != 3:
            if log_capture: log_capture.add_log(f"⚠️ 3D Preview: Invalid shape {positions.shape}")
            return None
            
        num_strands, points_per_strand, _ = positions.shape
        if num_strands == 0:
            if log_capture: log_capture.add_log("⚠️ 3D Preview: No strands found in data")
            return None

        if log_capture: log_capture.add_log(f"🎨 Interactive 3D: Loaded {num_strands} strands, {points_per_strand} points")
        
        # Balanced strand count for speed and quality
        target_strands = 1000
        strand_step = max(1, num_strands // target_strands)
        
        # Downsampling points per strand to 24 for better performance/look
        # Original is usually 32 points
        target_points = 24
        point_step = max(1, points_per_strand // target_points)
        
        subset = positions[::strand_step, ::point_step, :]
        n_s, n_p, _ = subset.shape
        
        if log_capture:
            log_capture.add_log(f"🎨 Interactive 3D: Rendering {n_s} strands with {n_p} points each")
        
        # Rotation and axis adjustment (matching Kaggle workflow)
        x_base = subset[:, :, 0]
        y_base = -subset[:, :, 2]
        z_base = subset[:, :, 1]
        
        theta = np.radians(180)
        c, s = np.cos(theta), np.sin(theta)
        fx = x_base * c + y_base * s
        fy = -x_base * s + y_base * c
        fz = z_base
        
        # Color calculation (Gradient from root to tip)
        t = np.linspace(0, 1, n_p)
        
        # ULTRA group_size for maximum speed
        # With 1000 strands, grouping by 250 means only 4 objects total
        traces = []
        group_size = 250
        for i in range(0, n_s, group_size):
            end_idx = min(i + group_size, n_s)
            
            g_fx = fx[i:end_idx]
            g_fy = fy[i:end_idx]
            g_fz = fz[i:end_idx]
            
            gn_s = g_fx.shape[0]
            nan_col = np.full((gn_s, 1), np.nan)
            
            gx_plot = np.hstack([g_fx, nan_col]).flatten().tolist()
            gy_plot = np.hstack([g_fy, nan_col]).flatten().tolist()
            gz_plot = np.hstack([g_fz, nan_col]).flatten().tolist()
            
            # Use 't' as color index for each point + 0 for the NaN
            g_color_idx = np.tile(np.append(t, 0), gn_s)
            
            traces.append(go.Scatter3d(
                x=gx_plot, y=gy_plot, z=gz_plot,
                mode='lines',
                line=dict(
                    width=2.2, # Slightly thicker as requested
                    color=g_color_idx.tolist(),
                    colorscale=[
                        [0.0, 'rgb(51, 51, 51)'],       # Raíz: Gris oscuro (Sólido)
                        [0.5, 'rgb(255, 255, 255)'],    # Centro: Blanco brillante (Sólido)
                        [1.0, 'rgb(30, 30, 35)']        # Puntas: Gris muy oscuro (Para fundir con el fondo #18181b)
                    ],
                ),
                hoverinfo='none',
                showlegend=False
            ))
        
        # 2. NO REFERENCE MARKERS (Removed red dot as requested)
        
        fig = go.Figure(data=traces)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#18181b',
            plot_bgcolor='#18181b',
            scene=dict(
                xaxis=dict(visible=False), # Hide grids as requested
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='#18181b',
                dragmode='turntable', 
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.8),
                    center=dict(x=0, y=0, z=0),
                    up=dict(x=0, y=0, z=1)
                ),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
                height=600,
                showlegend=False,
                modebar=dict(
                    bgcolor='rgba(0,0,0,0)',
                    color='#a1a1aa',
                    activecolor='#ffffff'
                ),
                uirevision=True,
                scene_camera_projection_type='perspective'
            )
        
        if log_capture:
            log_capture.add_log(f"✅ Hybrid 3D: Plot created with {len(traces)} groups")
        return fig
    except Exception as e:
        if log_capture: 
            log_capture.add_log(f"❌ Error in Interactive 3D: {str(e)}")
            import traceback
            log_capture.add_log(traceback.format_exc())
        return None

def create_empty_3d_plot(message="🎨 Interactive 3D preview will appear here after generation"):
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='#18181b',
        plot_bgcolor='#18181b',
        scene=dict(
            bgcolor='#18181b',
            xaxis=dict(visible=False, showgrid=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550,
        autosize=True,
        annotations=[dict(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="#a1a1aa")
        )]
    )
    return fig

def export_obj(npz_path, obj_path, log_capture=None):
    try:
        data = np.load(npz_path)
        positions = data['positions']
        num_strands, points_per_strand, _ = positions.shape
        with open(obj_path, 'w', buffering=4*1024*1024) as f:
            f.write(f"# DiffLocks Hair Export\n# Strands: {num_strands}, Points: {points_per_strand}\n\n")
            chunk_size = 5000
            for i in range(0, num_strands, chunk_size):
                end_idx = min(i + chunk_size, num_strands)
                chunk = positions[i:end_idx]
                flat = chunk.reshape(-1, 3)
                lines = "\n".join(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}" for p in flat)
                f.write(lines + "\n")
            f.write("\n# Polylines\n")
            for i in range(0, num_strands, chunk_size):
                end_idx = min(i + chunk_size, num_strands)
                lines = []
                for s in range(i, end_idx):
                    start = s * points_per_strand + 1
                    indices = " ".join(map(str, range(start, start + points_per_strand)))
                    lines.append(f"l {indices}")
                f.write("\n".join(lines) + "\n")
        del positions
        gc.collect()
        return True
    except Exception as e:
        if log_capture: log_capture.add_log(f"❌ OBJ Error: {e}")
        return False

def export_blender(npz_path, job_dir, formats, log_capture):
    format_map = {'.blend': 'blend', '.abc': 'abc', '.usd': 'usd'}
    keys = [v for k, v in format_map.items() if any(v.lower() in f.lower() for f in formats)]
    if not keys:
        log_capture.add_log("⚠️ No Blender formats selected")
        return []
    if not cfg.blender_exe.exists():
        log_capture.add_log(f"❌ Blender not found at: {cfg.blender_exe}")
        return []
    
    # Ensure blender is executable (Linux)
    if os.name != 'nt':
        try:
            os.chmod(cfg.blender_exe, 0o755)
        except: pass
        
    script = cfg.repo_dir / "inference/converter_blender.py"
    output_base = job_dir / "hair"
    cmd = [str(cfg.blender_exe), "-b", "-P", str(script), "--", str(npz_path), str(output_base)] + keys
    log_capture.add_log(f"🟧 Starting Blender export: {keys}")
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=300)
        if stdout:
            for line in stdout.strip().split('\n')[-10:]:
                if line.strip(): log_capture.add_log(f"[Blender] {line.strip()}")
        if stderr:
            for line in stderr.strip().split('\n')[-5:]:
                if line.strip() and "warning" not in line.lower(): log_capture.add_log(f"[Blender ERR] {line.strip()}")
        outputs = []
        for ext in ['.blend', '.abc', '.usd']:
            path = Path(f"{output_base}{ext}")
            if path.exists(): outputs.append(str(path))
        if not outputs: log_capture.add_log("⚠️ No Blender files were created")
        return outputs
    except Exception as e:
        log_capture.add_log(f"❌ Blender exception: {e}")
        return []

# --- 7. MAIN INFERENCE FUNCTION ---

def run_inference_logic(image, cfg_scale, export_formats, progress=gr.Progress()):
    log_capture = VerboseLogCapture()
    log_capture.start()
    tracker = ProgressTracker(is_cpu=IS_CPU)
    
    job_id = f"job_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    log_capture.add_log(f"🚀 JOB STARTED: {job_id}")
    log_capture.add_log(f"Device: {DEVICE} | Precision: float32")
    
    try:
        if image is None: raise ValueError("Please upload an image first!")
        
        # Phase: Init
        tracker.set_phase("init")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()), 
            generate_btn: gr.update(interactive=False),
            plot_3d: None,
            preview_2d: render_image_html(None),
            download_file: [],
            result_group: gr.update(visible=False),
            download_group: gr.update(visible=False)
        }
        
        # Load model
        load_model()
        
        img_path = job_dir / "input.png"
        if isinstance(image, str): shutil.copy(image, img_path)
        else: image.save(img_path)
        
        # Run model
        model.cfg_val = float(cfg_scale)
        
        # We need to make sure plot_3d_fig is initialized for the finally block
        plot_3d_fig = None
        
        # Generator approach for real-time updates
        for update in model.file2hair(str(img_path), str(job_dir), cfg_val=float(cfg_scale), progress=progress):
            if isinstance(update, tuple):
                dtype, val = update[0], update[1]
                if dtype == "status":
                    tracker.set_phase(val)
                elif dtype == "log":
                    log_capture.add_log(val)
                elif dtype == "error":
                    raise Exception(val)
            
            # Yield every single update to ensure real-time log/progress
            yield {
                status_html: create_dual_progress_html(*tracker.get_progress()),
                debug_console: render_debug_console(log_capture.get_logs()),
                result_group: gr.update(visible=False),
                download_group: gr.update(visible=False) # Keep downloads hidden
            }

        npz_path = job_dir / "difflocks_output_strands.npz"
        if not npz_path.exists(): raise Exception("No result was generated")
        
        # Previews & Exports
        tracker.set_phase("preview_2d")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        preview_img_path = generate_preview_2d(npz_path, job_dir, log_capture)
        preview_2d_html = render_image_html(preview_img_path)
        
        tracker.set_phase("preview_3d")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False) 
        }
        
        # Use optimized preview if available
        preview_npz = job_dir / "difflocks_output_strands_preview.npz"
        # Wait a tiny bit for filesystem sync if needed
        for _ in range(5):
            if preview_npz.exists(): break
            time.sleep(1)
            
        plot_3d_fig = generate_preview_3d(preview_npz if preview_npz.exists() else npz_path, log_capture)
        
        # SHOW BOTH PREVIEWS TOGETHER (2D and 3D) in Tabs
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()),
            preview_2d: preview_2d_html,
            plot_3d: plot_3d_fig if plot_3d_fig else gr.update(value=create_empty_3d_plot("⚠️ Could not render 3D")),
            result_group: gr.update(visible=True),
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        
        tracker.set_phase("obj_export")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        obj_path = job_dir / "hair.obj"
        export_obj(npz_path, obj_path, log_capture)
        yield { 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        
        tracker.set_phase("blender")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        blender_outputs = export_blender(npz_path, job_dir, export_formats, log_capture)
        yield { 
            debug_console: render_debug_console(log_capture.get_logs()),
            download_group: gr.update(visible=False)
        }
        
        # ZIP creation removed per user request
        # 12. Final completion
        log_capture.add_log("✨ Process finished successfully!")
        
        # Track completion
        tracker.get_progress(is_complete=True)
        
        # Ensure we return 100% and show all results
        final_status = create_complete_html()
        final_downloads = [str(f) for f in [obj_path] + [Path(p) for p in blender_outputs if p] if f and Path(f).exists()]
        
        # Final result yield
        yield {
            plot_3d: gr.update(value=plot_3d_fig if plot_3d_fig else create_empty_3d_plot("⚠️ Preview not available")),
            preview_2d: preview_2d_html if 'preview_2d_html' in locals() else gr.update(),
            status_html: final_status,
            result_group: gr.update(visible=True),
            download_file: final_downloads,
            download_group: gr.update(visible=True), 
            debug_console: render_debug_console(log_capture.get_logs()),
            generate_btn: gr.update(interactive=True)
        }
        
    except Exception as e:
        log_capture.add_log(f"❌ Error: {str(e)}")
        yield { status_html: create_error_html(str(e)), debug_console: render_debug_console(log_capture.get_logs()), generate_btn: gr.update(interactive=True) }
    finally:
        log_capture.stop()

# Wrapper for HuggingFace ZeroGPU if available
if HAS_ZEROGPU:
    run_inference = spaces.GPU(run_inference_logic)
else:
    run_inference = run_inference_logic

# --- 8. GRADIO UI ---

CSS = """
/* === FORCE DARK ON ALL === */
* {
    --block-label-background-fill: #3f3f46 !important;
    --block-label-text-color: #fafafa !important;
    --block-title-background-fill: #3f3f46 !important;
    --block-title-text-color: #fafafa !important;
}

/* === IMAGE LABEL FIX === */
.gr-image .label-wrap,
.gr-image > div:first-child > span,
[data-testid="image"] .label-wrap {
    background: #3f3f46 !important;
    color: #fafafa !important;
    max-height: 32px !important;
}

/* === CHECKBOX TEXT FIX === */
.gr-checkbox-group label,
.gr-checkbox-group label span,
.gr-checkbox-group input + span,
.checkbox-group label,
.checkbox-group label span,
[data-testid="checkbox-group"] label,
[data-testid="checkbox-group"] label span,
.gr-checkbox-group .gr-checkbox label,
.gr-checkbox-group .gr-checkbox span {
    color: #e4e4e7 !important;
    background: transparent !important;
    background-color: transparent !important;
}

/* === ACCORDION TEXT FIX === */
.gr-accordion,
.gr-accordion summary,
.gr-accordion summary span,
.gr-accordion > div,
details,
details summary,
details summary span,
details > div {
    color: #e4e4e7 !important;
    background-color: #27272a !important;
}

/* === HIDE FOOTER === */
footer { display: none !important; }

/* === DISABLE LOADING ANIMATIONS === */
.generating, .loading, .pending {
    animation: none !important;
    opacity: 1 !important;
}

/* === PLOTLY === */
.gr-plot, [data-testid="plot"] {
    background: #18181b !important;
    border: 1px solid #3f3f46 !important;
    border-radius: 8px !important;
    min-height: 550px !important;
    overflow: visible !important;
}

.js-plotly-plot .modebar {
    background: rgba(39, 39, 42, 0.9) !important;
}

.js-plotly-plot .modebar-btn {
    color: #a1a1aa !important;
}

.js-plotly-plot .modebar-btn:hover {
    color: #fafafa !important;
}

/* === SCROLLBAR === */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #18181b; }
::-webkit-scrollbar-thumb { background: #52525b; border-radius: 4px; }

/* === PLOTLY OVERRIDE === */
.js-plotly-plot, .plotly, .user-select-none {
    background: #18181b !important;
}
"""

js_func = """
function() {
    document.body.classList.add('dark');
    setInterval(() => {
        document.querySelectorAll('.generating, .loading, .pending').forEach(el => {
            el.style.animation = 'none';
            el.style.opacity = '1';
        });
        document.querySelectorAll('.gr-checkbox-group label, .gr-checkbox-group span').forEach(el => {
            el.style.color = '#e4e4e7';
            el.style.backgroundColor = 'transparent';
        });
        document.querySelectorAll('.gr-accordion, .gr-accordion summary, details, details summary').forEach(el => {
            el.style.color = '#e4e4e7';
        });
    }, 300);
}
"""

dark_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.indigo,
    secondary_hue=gr.themes.colors.zinc,
    neutral_hue=gr.themes.colors.zinc,
).set(
    body_background_fill="#1f1f23",
    body_background_fill_dark="#1f1f23",
    background_fill_primary="#27272a",
    background_fill_primary_dark="#27272a",
    background_fill_secondary="#18181b",
    background_fill_secondary_dark="#18181b",
    block_background_fill="#27272a",
    block_background_fill_dark="#27272a",
    block_border_color="#3f3f46",
    block_border_color_dark="#3f3f46",
    block_label_background_fill="#3f3f46",
    block_label_background_fill_dark="#3f3f46",
    block_label_text_color="#fafafa",
    block_label_text_color_dark="#fafafa",
    block_title_background_fill="#3f3f46",
    block_title_background_fill_dark="#3f3f46",
    block_title_text_color="#fafafa",
    block_title_text_color_dark="#fafafa",
    input_background_fill="#18181b",
    input_background_fill_dark="#18181b",
    input_border_color="#3f3f46",
    input_border_color_dark="#3f3f46",
    body_text_color="#e4e4e7",
    body_text_color_dark="#e4e4e7",
    body_text_color_subdued="#a1a1aa",
    body_text_color_subdued_dark="#a1a1aa",
    button_primary_background_fill="#6366f1",
    button_primary_background_fill_dark="#6366f1",
    button_primary_background_fill_hover="#818cf8",
    button_primary_background_fill_hover_dark="#818cf8",
    button_primary_text_color="#ffffff",
    button_primary_text_color_dark="#ffffff",
    border_color_primary="#3f3f46",
    border_color_primary_dark="#3f3f46",
)

with gr.Blocks(theme=dark_theme, css=CSS, title="DiffLocks Studio", js=js_func) as demo:
    # --- 8.1. CPU WARNING BANNER ---
    if IS_CPU:
        with gr.Row(elem_classes="cpu-warning"):
            gr.HTML(f'''
                <div style="background: rgba(239, 68, 68, 0.1); border: 1px solid #ef4444; border-radius: 8px; padding: 16px; margin-bottom: 20px;">
                    <div style="display: flex; align-items: center; gap: 12px;">
                        <span style="font-size: 24px;">⚠️</span>
                        <div style="flex-grow: 1;">
                            <h3 style="color: #ef4444; margin: 0; font-size: 16px; font-weight: 700;">No GPU detected! (Running on {cfg.platform.upper()})</h3>
                            <p style="color: #fca5a5; margin: 4px 0 0 0; font-size: 14px;">
                                Inference on CPU will take approximately <b>16 HOURS</b>. 
                                We strongly recommend using a GPU environment (Kaggle/Colab) for a 6-minute inference.
                            </p>
                        </div>
                    </div>
                </div>
            ''')

    # --- 8.2. HEADER ---
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown(f"""
                # 💇‍♀️ DiffLocks Studio
                ### High-fidelity 3D hair generation from a single image.
                *Platform: **{cfg.platform.upper()}** | Device: **{DEVICE}** | Precision: **float32***
            """)
        with gr.Column(scale=1):
            gr.Markdown(f"<div style='text-align: right; color: #71717a; font-size: 12px;'>v1.0.0-optimized</div>")

    # --- 8.3. MAIN INTERFACE ---
    with gr.Row(equal_height=False):
        # LEFT COLUMN: INPUTS
        with gr.Column(scale=4):
            with gr.Group():
                gr.Markdown("### 📥 Step 1: Input Image")
                image_input = gr.Image(type="filepath", label="Single Image (RGB)", height=400)
                
                with gr.Accordion("⚙️ Advanced Settings", open=False):
                    cfg_slider = gr.Slider(1.0, 7.0, 2.5, step=0.1, label="CFG Scale")
                    
                generate_btn = gr.Button("🚀 Generate 3D Hair", variant="primary", size="lg")

            with gr.Group():
                gr.Markdown("### 📦 Step 2: Export Formats")
                format_checkboxes = gr.CheckboxGroup(
                    choices=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"],
                    value=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"],
                    label="Select formats to generate"
                )
                gr.Markdown("<small>Blender export includes procedural hair curves and materials.</small>")

        # RIGHT COLUMN: OUTPUTS & PREVIEW
        with gr.Column(scale=6):
            # PROGRESS SECTION
            status_html = gr.HTML(value=create_dual_progress_html(0, 0, "Ready to start", 0, 0))
            
            with gr.Group(visible=False) as result_group:
                # Optimized layout: Tabs for 3D and 2D previews (3D first by default)
                with gr.Tabs() as preview_tabs:
                    with gr.Tab("🎨 Interactive 3D (Optimized Preview)", id="tab_3d"):
                        gr.Markdown("<small>✨ <b>Optimized Preview:</b> This is a simplified version for real-time interaction. For full quality and complete hair density, please download the export formats below.</small>")
                        plot_3d = gr.Plot(label="Interactive 3D")
                    
                    with gr.Tab("🖼️ 2D Preview", id="tab_2d"):
                        preview_2d = gr.HTML(render_image_html(None))
                
                with gr.Group(visible=False) as download_group:
                    gr.Markdown("### 📥 Download Results")
                    download_file = gr.File(label="Generated Assets", file_count="multiple")

            with gr.Accordion("📜 Debug Console", open=True):
                debug_console = gr.HTML(value=render_debug_console([]))

    generate_btn.click(
        fn=run_inference,
        inputs=[image_input, cfg_slider, format_checkboxes],
        outputs=[plot_3d, preview_2d, status_html, result_group, download_file, debug_console, generate_btn, download_group]
    )

if __name__ == "__main__":
    # Launch parameters optimized for different platforms
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": 7860,
        "share": cfg.needs_share
    }
    
    # Notebook specific settings
    if cfg.platform in ['kaggle', 'colab']:
        launch_kwargs["inline"] = True
        launch_kwargs["height"] = 1000
    
    demo.queue().launch(**launch_kwargs)
