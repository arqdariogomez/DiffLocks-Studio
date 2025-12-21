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
    ("1/5", "📸 Preprocessing image", 15, 0.01, 0.03),
    ("2/5", "🔍 Extracting features", 45, 0.03, 0.08),
    ("3/5", "✨ Running diffusion", 360, 0.08, 0.55), # GPU: 6 min | CPU: 16h
    ("4/5", "🧶 Decoding strands", 180, 0.55, 0.75),
    ("5/5", "🏁 Finalizing inference", 10, 0.75, 0.78),
    ("preview_2d", "🎨 Creating 2D preview", 15, 0.78, 0.82),
    ("preview_3d", "🎨 Creating Interactive 3D", 20, 0.82, 0.86),
    ("obj_export", "📦 Exporting OBJ", 60, 0.86, 0.92),
    ("blender", "🟧 Blender export", 120, 0.92, 0.97),
    ("zip", "📦 Creating ZIP", 15, 0.97, 1.00),
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
        
        if self.capturing and msg and msg.strip():
            # Filter out redundant model internal prints
            skip_terms = ["cfg_val_cur", "for sigma", "tensor(", "width for this lvl", "initializing LVL", "making up cross layer"]
            
            with self.lock:
                ts = time.strftime('%H:%M:%S')
                for line in msg.strip().split('\n'):
                    line = line.strip()
                    if line and not any(term in line for term in skip_terms):
                        # Clean up tqdm lines
                        if "|" in line and "%" in line:
                            # Keep only the progress part
                            line = line.split("|")[-1].strip()
                        self.logs.append(f"[{ts}] {line}")
                
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
                return
    
    def get_progress(self):
        if self.current_phase_index >= len(self.phases):
            return 100, 0, "Complete", 100, 0
        current_phase = self.phases[self.current_phase_index]
        phase_id, phase_name, phase_duration, phase_start_pct, phase_end_pct = current_phase
        phase_elapsed = time.time() - self.current_phase_start_time
        phase_progress = min((phase_elapsed / phase_duration) * 100, 99) if phase_duration > 0 else 0
        phase_remaining = max(phase_duration - phase_elapsed, 0)
        completed_time = sum(self.phases[i][2] for i in range(self.current_phase_index))
        current_contribution = (phase_progress / 100) * phase_duration
        total_elapsed_estimated = completed_time + current_contribution
        total_progress = min((total_elapsed_estimated / self.total_estimated) * 100, 99)
        remaining_phases_time = sum(self.phases[i][2] for i in range(self.current_phase_index + 1, len(self.phases)))
        total_remaining = phase_remaining + remaining_phases_time
        return total_progress, total_remaining, phase_name, phase_progress, phase_remaining

# --- 4. HTML RENDERING FUNCTIONS ---

def format_time(seconds):
    if seconds < 0: seconds = 0
    if seconds >= 3600:
        return f"{int(seconds // 3600)}h {int((seconds % 3600) // 60)}m"
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes}m {secs}s" if minutes > 0 else f"{secs}s"

def render_debug_console(logs_list):
    if not logs_list: logs_list = ["[Waiting for job to start...]"]
    lines_html = []
    for line in logs_list[-50:]:
        escaped = escape(line)
        color = "#d4d4d8"
        if any(x in line for x in ["❌", "ERROR", "ERR]"]): color = "#f87171"
        elif any(x in line for x in ["✅", "SUCCESS", "COMPLETED"]): color = "#34d399"
        elif any(x in line for x in ["⚠️", "WARNING"]): color = "#fbbf24"
        elif any(x in line for x in ["🔄", "Diffusion:"]): color = "#818cf8"
        elif any(x in line for x in ["🟧", "[Blender]"]): color = "#fb923c"
        lines_html.append(f'<div style="color: {color}; margin: 2px 0;">{escaped}</div>')
    return f'<div style="background-color: #18181b; border: 1px solid #3f3f46; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 12px; line-height: 1.5; max-height: 400px; overflow-y: auto;">{"".join(lines_html)}</div>'

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
    <div style="background-color: #18181b; border: 1px solid #3f3f46; border-radius: 8px; padding: 12px; text-align: center;">
        <img src="data:image/png;base64,{img_data}" style="max-width: 100%; height: auto; border-radius: 6px;" alt="{title}"/>
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
                <span style="background: {main_color}33; color: {main_color}; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 700;">{int(total_pct)}%</span>
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
                <span style="color: #a5b4fc; font-size: 13px; font-weight: 600;">{int(step_pct)}%</span>
            </div>
            <div style="background: #3f3f46; height: 6px; border-radius: 3px; overflow: hidden;">
                <div style="width: {step_pct}%; height: 100%; background: linear-gradient(90deg, #a5b4fc, #818cf8); transition: width 0.3s ease-in-out;"></div>
            </div>
        </div>
    </div>
    '''

def create_complete_html():
    return f'''
    <div style="padding: 20px; background: linear-gradient(135deg, #18181b 0%, #27272a 100%); border-radius: 12px; border: 1px solid #34d39966; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 28px;">🎉</span>
                <span style="font-size: 20px; font-weight: 700; color: #34d399;">Complete!</span>
            </div>
            <span style="background: #34d39933; color: #34d399; padding: 6px 14px; border-radius: 20px; font-size: 14px; font-weight: 700;">100%</span>
        </div>
        <div style="background: #3f3f46; height: 12px; border-radius: 6px; overflow: hidden; margin-top: 12px;">
            <div style="width: 100%; height: 100%; background: linear-gradient(90deg, #34d399, #22c55e);"></div>
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

ckpt_files = list(cfg.repo_dir.rglob("scalp_*.pth"))
vae_files = list(cfg.repo_dir.rglob("strand_codec.pt"))
conf_path = cfg.repo_dir / "configs/config_scalp_texture_conditional.json"

model = None

def load_model():
    global model
    if model is not None: return

    if not ckpt_files or not vae_files:
        raise FileNotFoundError("Checkpoints missing!")
    
    print(f"Loading Model on {DEVICE} (Precision=float32): {ckpt_files[0].name}")
    model = DiffLocksInference(str(vae_files[0]), str(conf_path), str(ckpt_files[0]), DEVICE)
    print("✅ Model loaded!")

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
        if log_capture: log_capture.add_log(f"❌ 2D Preview error: {e}")
        return None

def generate_preview_3d(npz_path, log_capture=None):
    try:
        if log_capture: log_capture.add_log("🎨 3D Interactive: Loading data...")
        data = np.load(npz_path)
        positions = data['positions']
        num_strands, points_per_strand, _ = positions.shape
        
        if log_capture: log_capture.add_log(f"🎨 3D Interactive: Processing {num_strands} strands...")
        strand_step = max(1, num_strands // target_strands)
        target_points = 32
        point_step = max(1, points_per_strand // target_points)
        
        subset = positions[::strand_step, ::point_step, :]
        n_s, n_p, _ = subset.shape
        
        x = subset[:, :, 0]
        y = -subset[:, :, 2]
        z = subset[:, :, 1]
        
        theta = np.radians(180)
        c, s = np.cos(theta), np.sin(theta)
        fx = x * c + y * s
        fy = -x * s + y * c
        fz = z
        
        nan_col = np.full((n_s, 1), np.nan)
        x_plot = np.hstack([fx, nan_col]).flatten()
        y_plot = np.hstack([fy, nan_col]).flatten()
        z_plot = np.hstack([fz, nan_col]).flatten()
        
        np.random.seed(42)
        colors_flat = []
        for s_idx in range(n_s):
            brightness_var = 0.85 + np.random.random() * 0.30
            for p_idx in range(n_p):
                t = p_idx / max(1, n_p - 1)
                base_val = 0.2 + t * 0.75
                val = np.clip(base_val * brightness_var, 0.1, 1.0)
                colors_flat.append(val)
            colors_flat.append(0.5)
        
        color_array = np.array(colors_flat)
        colorscale = [
            [0.0, 'rgb(30,30,30)'],
            [0.2, 'rgb(60,60,60)'],
            [0.4, 'rgb(100,100,100)'],
            [0.6, 'rgb(160,160,160)'],
            [0.8, 'rgb(210,210,210)'],
            [1.0, 'rgb(250,250,250)']
        ]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=x_plot, y=y_plot, z=z_plot,
            mode='lines',
            line=dict(width=2, color=color_array, colorscale=colorscale, showscale=False),
            hoverinfo='none'
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showbackground=False),
                yaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showbackground=False),
                zaxis=dict(visible=False, showgrid=False, showline=False, zeroline=False, showbackground=False),
                bgcolor='rgba(0,0,0,0)',
                dragmode='orbit',
                camera=dict(eye=dict(x=0, y=-1.8, z=0.3), up=dict(x=0, y=0, z=1)),
                aspectmode='data'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=550,
            showlegend=False
        )
        
        if log_capture: log_capture.add_log("✅ 3D Interactive: Plot created successfully")
        del positions, subset, x, y, z, fx, fy, fz, x_plot, y_plot, z_plot, color_array
        gc.collect()
        return fig
    except Exception as e:
        if log_capture: log_capture.add_log(f"❌ 3D Interactive error: {e}")
        return None

def create_empty_3d_plot():
    fig = go.Figure()
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, showgrid=False, showbackground=False),
            yaxis=dict(visible=False, showgrid=False, showbackground=False),
            zaxis=dict(visible=False, showgrid=False, showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=550,
        annotations=[dict(
            text="🎨 3D Preview will appear here after generation",
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
        if log_capture: log_capture.add_log(f"❌ OBJ error: {e}")
        return False

def export_blender(npz_path, job_dir, formats, log_capture):
    format_map = {'.blend': 'blend', '.abc': 'abc', '.usd': 'usd'}
    keys = [v for k, v in format_map.items() if any(v.lower() in f.lower() for f in formats)]
    if not keys:
        log_capture.add_log("⚠️ No Blender formats selected")
        return []
    if not cfg.blender_exe.exists():
        log_capture.add_log(f"❌ Blender not found: {cfg.blender_exe}")
        return []
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

def run_inference(image, cfg_scale, export_formats, progress=gr.Progress()):
    log_capture = VerboseLogCapture()
    log_capture.start()
    tracker = ProgressTracker(is_cpu=IS_CPU)
    
    job_id = f"job_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    log_capture.add_log(f"🚀 JOB STARTED: {job_id}")
    log_capture.add_log(f"Device: {DEVICE} | Precision: float32")
    
    try:
        if image is None: raise ValueError("Upload an image first!")
        
        # Phase: Init
        tracker.set_phase("init")
        yield { 
            status_html: create_dual_progress_html(*tracker.get_progress()), 
            debug_console: render_debug_console(log_capture.get_logs()), 
            generate_btn: gr.update(interactive=False),
            plot_3d: create_empty_3d_plot(),
            preview_2d: render_image_html(None)
        }
        
        # Load model
        load_model()
        
        img_path = job_dir / "input.png"
        if isinstance(image, str): shutil.copy(image, img_path)
        else: image.save(img_path)
        
        # Run model
        model.cfg_val = float(cfg_scale)
        for update in model.file2hair(str(img_path), str(job_dir), cfg_val=float(cfg_scale), progress=progress):
            if isinstance(update, tuple):
                dtype, val = update[0], update[1]
                if dtype == "status":
                    tracker.set_phase(val)
                    yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
                elif dtype == "log":
                    log_capture.add_log(val)
                    yield { debug_console: render_debug_console(log_capture.get_logs()) }
                elif dtype == "error":
                    raise Exception(val)
            
            # Update progress tracker based on time passing
            yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }

        npz_path = job_dir / "difflocks_output_strands.npz"
        if not npz_path.exists(): raise Exception("No output generated")
        
        # Previews & Exports
        tracker.set_phase("preview_2d")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        preview_img_path = generate_preview_2d(npz_path, job_dir, log_capture)
        preview_2d_html = render_image_html(preview_img_path)
        
        tracker.set_phase("preview_3d")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        plot_3d_fig = generate_preview_3d(npz_path, log_capture)
        if plot_3d_fig is None: plot_3d_fig = create_empty_3d_plot()
        
        tracker.set_phase("obj_export")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        obj_path = job_dir / "hair.obj"
        export_obj(npz_path, obj_path, log_capture)
        
        tracker.set_phase("blender")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        blender_outputs = export_blender(npz_path, job_dir, export_formats, log_capture)
        
        tracker.set_phase("zip")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        zip_path = job_dir / "DiffLocks_Results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in [str(npz_path), preview_img_path, str(obj_path)] + blender_outputs:
                if f and Path(f).exists(): zf.write(f, Path(f).name)
        
        # Final yield with all files
        final_files = [str(zip_path), str(npz_path), preview_img_path, str(obj_path)] + blender_outputs
        final_files = [f for f in final_files if f and Path(f).exists()]
        
        yield {
            plot_3d: plot_3d_fig,
            preview_2d: preview_2d_html,
            status_html: create_complete_html(),
            result_group: gr.update(visible=True),
            download_file: final_files,
            debug_console: render_debug_console(log_capture.get_logs()),
            generate_btn: gr.update(interactive=True)
        }
        
    except Exception as e:
        log_capture.add_log(f"❌ Error: {str(e)}")
        yield { status_html: create_error_html(str(e)), debug_console: render_debug_console(log_capture.get_logs()), generate_btn: gr.update(interactive=True) }
    finally:
        log_capture.stop()

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
    gr.Markdown("## 💇‍♀️ DiffLocks Studio")
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Input Portrait", height=300)
            cfg_slider = gr.Slider(1, 7, 2.5, step=0.1, label="CFG Scale")
            format_checkboxes = gr.CheckboxGroup(choices=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"], value=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"], label="Additional Exports")
            generate_btn = gr.Button("🚀 GENERATE HAIR", variant="primary", size="lg")
        
        with gr.Column(scale=2):
            status_html = gr.HTML(value=create_dual_progress_html(0, 0, "⏳ Ready", 0, 0))
            with gr.Group(visible=False) as result_group:
                with gr.Tabs():
                    with gr.Tab("🎨 3D Interactive preview"): plot_3d = gr.Plot(value=create_empty_3d_plot())
                    with gr.Tab("📸 2D Preview"): preview_2d = gr.HTML(value=render_image_html(None))
                download_file = gr.File(label="📥 Download Results", file_count="multiple")
            with gr.Accordion("🛠️ Debug Console", open=True):
                debug_console = gr.HTML(value=render_debug_console([]))
    
    generate_btn.click(
        fn=run_inference,
        inputs=[image_input, cfg_slider, format_checkboxes],
        outputs=[plot_3d, preview_2d, status_html, result_group, download_file, debug_console, generate_btn]
    )

if __name__ == "__main__":
    demo.queue().launch(share=cfg.needs_share, server_name="0.0.0.0", server_port=7860)
