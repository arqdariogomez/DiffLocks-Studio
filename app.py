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
            with self.lock:
                ts = time.strftime('%H:%M:%S')
                for line in msg.strip().split('\n'):
                    if line.strip():
                        self.logs.append(f"[{ts}] {line.strip()}")
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
    print(f"Loading Model on {DEVICE}: {ckpt_files[0].name}")
    model = DiffLocksInference(str(vae_files[0]), str(conf_path), str(ckpt_files[0]), DEVICE)
    print("✅ Model loaded!")

# --- 6. UTILITY FUNCTIONS ---

def generate_preview_2d(npz_path, output_dir):
    try:
        import matplotlib.pyplot as plt
        data = np.load(npz_path)['positions']
        step = max(1, len(data) // 20000)
        pts = data[::step].reshape(-1, 3)
        x, y, z = pts[:, 0], -pts[:, 2], pts[:, 1]
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#111')
        views = [(x, z, y, 'FRONT'), (y, z, np.abs(x), 'SIDE'), (x, y, z, 'TOP')]
        for ax, (u, v, depth, title) in zip(axes, views):
            idx = np.argsort(depth)
            norm_depth = (depth[idx] - depth.min()) / (depth.max() - depth.min() + 1e-8)
            ax.scatter(u[idx], v[idx], c=norm_depth, cmap='copper', s=0.5, lw=0)
            ax.axis('off')
            ax.set_title(title, color='#888', fontsize=14)
        preview_path = output_dir / "preview.png"
        fig.savefig(preview_path, facecolor='#111', bbox_inches='tight', dpi=100)
        plt.close(fig)
        return str(preview_path)
    except: return None

def generate_preview_3d(npz_path):
    try:
        import plotly.graph_objects as go
        data = np.load(npz_path)['positions']
        step = max(1, len(data) // 30000)
        sample = data[::step][:, ::8, :]
        pts = sample.reshape(-1, 3)
        x, y, z = pts[:, 0], -pts[:, 2], pts[:, 1]
        colors = np.tile(np.linspace(0.3, 1, sample.shape[1]), sample.shape[0])
        def add_nan(arr): return np.hstack([arr.reshape(sample.shape[:2]), np.full((sample.shape[0], 1), np.nan)]).flatten()
        fig = go.Figure(data=[go.Scatter3d(x=add_nan(x), y=add_nan(y), z=add_nan(z), mode='lines', line=dict(width=1.5, color=np.hstack([colors, np.zeros(sample.shape[0])]), colorscale=[[0, '#505050'], [1, 'white']], showscale=False))])
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False), bgcolor='rgba(0,0,0,0)'), margin=dict(l=0, r=0, b=0, t=0), height=500)
        return fig
    except: return None

def export_obj(npz_path, obj_path):
    try:
        data = np.load(npz_path); pos = data['positions']; points_per_strand = 100
        with open(obj_path, 'w') as f:
            for p in pos: f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            num_strands = len(pos) // points_per_strand
            for s in range(num_strands):
                start_idx = s * points_per_strand + 1
                indices = range(start_idx, start_idx + points_per_strand)
                f.write("l " + " ".join(map(str, indices)) + "\n")
        return True
    except: return False

def export_blender(npz_path, output_base, formats):
    format_map = {'.blend': 'blend', '.abc': 'abc', '.usd': 'usd'}
    keys = [v for k, v in format_map.items() if any(v.lower() in f.lower() for f in formats)]
    if not keys or not cfg.blender_exe.exists(): return []
    script = cfg.repo_dir / "inference/converter_blender.py"
    cmd = [str(cfg.blender_exe), "-b", "-P", str(script), "--", str(npz_path), str(output_base)] + keys
    try:
        subprocess.run(cmd, capture_output=True, timeout=600)
        outputs = []
        for ext in ['.blend', '.abc', '.usd']:
            path = Path(f"{output_base}{ext}")
            if path.exists(): outputs.append(str(path))
        return outputs
    except: return []

# --- 7. MAIN INFERENCE FUNCTION ---

def run_inference(image, cfg_scale, export_formats, progress=gr.Progress()):
    log_capture = VerboseLogCapture()
    log_capture.start()
    tracker = ProgressTracker(is_cpu=IS_CPU)
    
    job_id = f"job_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    log_capture.add_log(f"🚀 JOB STARTED: {job_id}")
    log_capture.add_log(f"Device: {DEVICE} | Precision: {'float16' if cfg.use_half else 'float32'}")
    
    try:
        if image is None: raise ValueError("Upload an image first!")
        
        # Phase: Init
        tracker.set_phase("init")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()), generate_btn: gr.Button(interactive=False) }
        
        if model is None:
            log_capture.add_log("🧠 Loading model...")
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
        preview_img = generate_preview_2d(npz_path, job_dir)
        
        tracker.set_phase("preview_3d")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        plot_3d_fig = generate_preview_3d(npz_path)
        
        tracker.set_phase("obj_export")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        obj_path = job_dir / "hair.obj"
        export_obj(npz_path, obj_path)
        
        tracker.set_phase("blender")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        blender_outputs = export_blender(npz_path, job_dir / "hair", export_formats)
        
        tracker.set_phase("zip")
        yield { status_html: create_dual_progress_html(*tracker.get_progress()), debug_console: render_debug_console(log_capture.get_logs()) }
        zip_path = job_dir / "DiffLocks_Results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in [str(npz_path), preview_img, str(obj_path)] + blender_outputs:
                if f and Path(f).exists(): zf.write(f, Path(f).name)
        
        yield {
            plot_3d: plot_3d_fig,
            preview_2d: preview_img,
            status_html: create_complete_html(),
            result_group: gr.Group(visible=True),
            download_file: str(zip_path),
            debug_console: render_debug_console(log_capture.get_logs()),
            generate_btn: gr.Button(interactive=True)
        }
        
    except Exception as e:
        log_capture.add_log(f"❌ Error: {str(e)}")
        yield { status_html: create_error_html(str(e)), debug_console: render_debug_console(log_capture.get_logs()), generate_btn: gr.Button(interactive=True) }
    finally:
        log_capture.stop()

# --- 8. GRADIO UI ---

CSS = """
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }
.debug-console { font-family: monospace; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, title="DiffLocks Studio") as demo:
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
                    with gr.Tab("🎨 3D Preview"): plot_3d = gr.Plot()
                    with gr.Tab("📸 2D Renders"): preview_2d = gr.Image(interactive=False)
                download_file = gr.File(label="📥 Download Results")
            with gr.Accordion("🛠️ Debug Console", open=True):
                debug_console = gr.HTML(value=render_debug_console([]))
    
    generate_btn.click(
        fn=run_inference,
        inputs=[image_input, cfg_slider, format_checkboxes],
        outputs=[plot_3d, preview_2d, status_html, result_group, download_file, debug_console, generate_btn]
    )

if __name__ == "__main__":
    demo.queue().launch(share=cfg.needs_share, server_name="0.0.0.0", server_port=7860)
