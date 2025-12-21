# app.py - DiffLocks Studio (Universal V2)
# ============================================================================
# Improvements:
# - Modern UI with visual states (from Kaggle V144)
# - Integrated Debug Console
# - Progress translations with estimated times
# - Native OBJ export (no Blender required)
# - Robust error handling
# - Compatible with: HF Space, Kaggle, Colab, Docker, Local
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

if IS_CPU:
    print("\n" + "!" * 60)
    print("⚠️  WARNING: RUNNING IN CPU MODE")
    print("   Inference will be extremely slow (hours per image).")
    print("   Please apply for ZeroGPU or use Colab/Kaggle with GPU.")
    print("!" * 60 + "\n")

# --- 2. HF SPACE AUTO-SETUP ---
if cfg.platform == 'huggingface':
    token = os.environ.get("HF_TOKEN")
    
    # Install Blender if needed
    if not cfg.blender_exe.exists():
        print("📦 Installing Blender...")
        b_dir = Path("/tmp/blender")
        b_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            "wget -q -O /tmp/blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.5-linux-x64.tar.xz",
            shell=True
        )
        subprocess.run(
            f"tar -xf /tmp/blender.tar.xz -C {b_dir} --strip-components=1",
            shell=True
        )
    
    # Download checkpoints if missing
    if not list(cfg.repo_dir.rglob("*.pth")):
        print("🧠 Downloading Assets...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="arqdariogomez/difflocks-assets-hybrid",
                repo_type="dataset",
                allow_patterns=["checkpoints/*", "assets/*", "*.pt"],
                local_dir=cfg.repo_dir,
                token=token
            )
            
            # Organize assets
            src_assets = cfg.repo_dir / "assets"
            dst_assets = cfg.repo_dir / "inference/assets"
            if src_assets.exists():
                dst_assets.mkdir(parents=True, exist_ok=True)
                for f in src_assets.glob("*"):
                    shutil.move(str(f), str(dst_assets / f.name))
                shutil.rmtree(src_assets)
                
        except Exception as e:
            print(f"❌ Download Error: {e}")
            # Fallback to download script
            if (cfg.repo_dir / "download_checkpoints.py").exists():
                subprocess.run(["python", "download_checkpoints.py"])

# --- 3. MODEL LOADER ---
from inference.img2hair import DiffLocksInference

# Find checkpoints recursively
# Improved model search
ckpt_files = list(cfg.repo_dir.rglob("scalp_*.pth"))
if not ckpt_files:
    # Fallback: try to find anything that looks like a model, excluding cache and dinov2
    all_pths = list(cfg.repo_dir.rglob("*.pth"))
    ckpt_files = [p for p in all_pths if "dinov2" not in p.name and "cache" not in str(p).lower()]
    
if not ckpt_files:
    # Last resort fallback
    ckpt_files = list(cfg.repo_dir.rglob("*.pth"))
vae_files = list(cfg.repo_dir.rglob("strand_codec.pt"))
conf_path = cfg.repo_dir / "configs/config_scalp_texture_conditional.json"

model = None

def load_model():
    """Lazy load the model"""
    global model
    if model is not None:
        return
    
    if not ckpt_files or not vae_files:
        raise FileNotFoundError(
            f"Checkpoints missing!\n"
            f"PTH files: {len(ckpt_files)}, PT files: {len(vae_files)}\n"
            f"Ensure 'HF_TOKEN' is set in Settings -> Secrets."
        )
    
    print(f"Loading Model on {DEVICE}: {ckpt_files[0].name}")
    model = DiffLocksInference(
        str(vae_files[0]),
        str(conf_path),
        str(ckpt_files[0]),
        DEVICE
    )
    print("✅ Model loaded!")

# Preload if GPU available and not using ZeroGPU
if not HAS_ZEROGPU and not IS_CPU:
    try:
        load_model()
    except Exception as e:
        print(f"Preload skipped: {e}")

# --- 4. UTILITY FUNCTIONS ---

def translate_status(msg):
    """Translate status messages to user-friendly format with time estimates"""
    msg_lower = str(msg).lower()
    
    # Time estimates vary by device
    time_modifier = "" if not IS_CPU else " (CPU: much longer)"
    
    translations = [
        ("1/5", "📸 Preprocessing image...", f"~15s{time_modifier}"),
        ("2/5", "🔍 Extracting features...", f"~45s{time_modifier}"),
        ("3/5", "✨ Running diffusion...", "~6 min" if not IS_CPU else "~16 hours (with gpu it takes 6 min"),
        ("4/5", "🧶 Decoding strands...", f"~30s{time_modifier}"),
        ("5/5", "🏁 Finalizing geometry...", f"~10s{time_modifier}"),
    ]
    
    for key, text, time_est in translations:
        if key in msg_lower:
            return text, time_est
    
    return str(msg), ""

def create_status_html(message, time_est="", status_type="info"):
    """Create styled HTML for status display"""
    colors = {
        "info": "#3b82f6",
        "success": "#10b981",
        "error": "#ef4444",
        "warning": "#f59e0b"
    }
    color = colors.get(status_type, "#3b82f6")
    
    time_html = f"<span style='opacity:0.6; font-size:0.9em; margin-left:8px'>({time_est})</span>" if time_est else ""
    
    return f"""
    <div style="
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 20px;
        background: linear-gradient(135deg, #1e293b, #334155);
        border-radius: 10px;
        border-left: 4px solid {color};
        margin: 10px 0;
    ">
        <span style="font-size: 16px; font-weight: 600; color: #e2e8f0;">
            {message} {time_html}
        </span>
    </div>
    """

def generate_preview_2d(npz_path, output_dir):
    """Generate 2D preview renders"""
    try:
        import matplotlib.pyplot as plt
        
        data = np.load(npz_path)['positions']
        step = max(1, len(data) // 20000)
        pts = data[::step].reshape(-1, 3)
        x, y, z = pts[:, 0], -pts[:, 2], pts[:, 1]  # Rotate for standard view
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.patch.set_facecolor('#111')
        
        views = [
            (x, z, y, 'FRONT'),
            (y, z, np.abs(x), 'SIDE'),
            (x, y, z, 'TOP')
        ]
        
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
    except Exception as e:
        print(f"2D Preview error: {e}")
        return None

def generate_preview_3d(npz_path):
    """Generate interactive 3D preview with Plotly"""
    try:
        import plotly.graph_objects as go
        
        data = np.load(npz_path)['positions']
        step = max(1, len(data) // 30000)
        sample = data[::step][:, ::8, :]  # Subsample points per strand
        pts = sample.reshape(-1, 3)
        x, y, z = pts[:, 0], -pts[:, 2], pts[:, 1]
        
        # Color gradient along strands
        colors = np.tile(np.linspace(0.3, 1, sample.shape[1]), sample.shape[0])
        
        # Add NaN to separate line segments
        def add_nan(arr):
            return np.hstack([
                arr.reshape(sample.shape[:2]),
                np.full((sample.shape[0], 1), np.nan)
            ]).flatten()
        
        fig = go.Figure(data=[go.Scatter3d(
            x=add_nan(x), y=add_nan(y), z=add_nan(z),
            mode='lines',
            line=dict(
                width=1.5,
                color=np.hstack([colors, np.zeros(sample.shape[0])]),
                colorscale=[[0, '#505050'], [1, 'white']],
                showscale=False
            )
        )])
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(l=0, r=0, b=0, t=0),
            height=500
        )
        
        return fig
    except Exception as e:
        print(f"3D Preview error: {e}")
        return None

def export_obj(npz_path, obj_path):
    """Export to OBJ format using pure NumPy (no Blender needed)"""
    try:
        data = np.load(npz_path)
        pos = data['positions']
        points_per_strand = 100
        
        with open(obj_path, 'w') as f:
            f.write("# DiffLocks Hair Export\n")
            f.write(f"# Strands: {len(pos) // points_per_strand}\n\n")
            
            # Write vertices
            for p in pos:
                f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
            
            # Write lines (strands)
            num_strands = len(pos) // points_per_strand
            for s in range(num_strands):
                start_idx = s * points_per_strand + 1  # OBJ is 1-indexed
                indices = range(start_idx, start_idx + points_per_strand)
                f.write("l " + " ".join(map(str, indices)) + "\n")
        
        return True
    except Exception as e:
        print(f"OBJ Export error: {e}")
        return False

def export_blender(npz_path, output_base, formats):
    """Export using Blender for advanced formats"""
    format_map = {'.blend': 'blend', '.abc': 'abc', '.usd': 'usd'}
    keys = [v for k, v in format_map.items() if any(v.lower() in f.lower() for f in formats)]
    
    if not keys:
        return []

    if not cfg.blender_exe.exists():
        if cfg.platform == 'pinokio' or sys.platform == 'win32':
            print("⚠️ Blender not found. Skipping .blend/.abc/.usd export.")
            print(f"To enable, install Blender 4.2+ and place it at: {cfg.blender_exe}")
        return []
    
    script = cfg.repo_dir / "inference/converter_blender.py"
    cmd = [str(cfg.blender_exe), "-b", "-P", str(script), "--",
           str(npz_path), str(output_base)] + keys
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=600)
        
        outputs = []
        for ext in ['.blend', '.abc', '.usd']:
            path = Path(f"{output_base}{ext}")
            if path.exists():
                outputs.append(str(path))
        return outputs
    except Exception as e:
        print(f"Blender export error: {e}")
        return []

def check_for_updates():
    """Check for git updates and pull if available"""
    try:
        import subprocess
        # Fetch latest
        subprocess.run(["git", "fetch"], capture_output=True, check=True)
        # Check status
        status = subprocess.run(["git", "status", "-uno"], capture_output=True, text=True, check=True).stdout
        if "Your branch is behind" in status:
            yield "🔄 Update found! Pulling changes..."
            subprocess.run(["git", "pull"], capture_output=True, check=True)
            yield "✅ Successfully updated! Please restart the app to apply changes."
        else:
            yield "✨ You are already up to date!"
    except Exception as e:
        yield f"❌ Update failed: {str(e)}"

# --- 5. MAIN INFERENCE FUNCTION ---

def run_inference(image, cfg_scale, export_formats, debug_logs):
    """Main inference function with improved UI feedback"""
    
    # Initialize logs
    logs = debug_logs if debug_logs else ""
    
    def add_log(msg):
        nonlocal logs
        ts = time.strftime('%H:%M:%S')
        logs += f"[{ts}] {msg}\n"
        return logs
    
    # Create job directory
    job_id = f"job_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    add_log(f"Job started: {job_id}")
    add_log(f"Device: {DEVICE}")
    
    try:
        # Validate input
        if image is None:
            raise ValueError("Please upload an image first!")
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load model if needed
        if model is None:
            add_log("Loading model (first run)...")
            yield {
                plot_3d: None,
                preview_2d: None,
                status_html: create_status_html("🧠 Loading model...", "first time only"),
                result_group: gr.Group(visible=False),
                download_file: None,
                debug_console: logs,
                generate_btn: gr.Button(interactive=False)
            }
            load_model()
            add_log("Model loaded!")
        
        # Save input image
        img_path = job_dir / "input.png"
        if isinstance(image, str):
            shutil.copy(image, img_path)
        else:
            image.save(img_path)
        add_log("Input image saved")
        
        # Start inference
        add_log("Starting inference...")
        yield {
            status_html: create_status_html("🚀 Starting inference..."),
            result_group: gr.Group(visible=False),
            generate_btn: gr.Button(interactive=False),
            debug_console: logs
        }
        
        # Run model
        model.cfg_val = float(cfg_scale)
        
        for update in model.file2hair(str(img_path), str(job_dir)):
            if isinstance(update, tuple) and update[0] == "status":
                text, time_est = translate_status(update[1])
                add_log(update[1])
                yield {
                    status_html: create_status_html(text, time_est),
                    debug_console: logs
                }
        
        # Verify output
        npz_path = job_dir / "difflocks_output_strands.npz"
        if not npz_path.exists():
            raise Exception("No output generated - check debug console for errors")
        
        add_log("✅ Hair geometry generated!")
        output_files = [str(npz_path)]
        
        # Generate previews
        yield {
            status_html: create_status_html("🎨 Creating previews..."),
            debug_console: logs
        }
        
        preview_img = generate_preview_2d(npz_path, job_dir)
        if preview_img:
            output_files.append(preview_img)
            add_log("2D preview created")
        
        plot_3d_fig = generate_preview_3d(npz_path)
        if plot_3d_fig:
            add_log("3D preview created")
        
        # Export OBJ (always, it's fast)
        obj_path = job_dir / "hair.obj"
        if export_obj(npz_path, obj_path):
            output_files.append(str(obj_path))
            add_log("OBJ exported")
        
        # Export Blender formats if selected
        blender_formats = [f for f in export_formats if any(x in f.lower() for x in ['blend', 'abc', 'usd'])]
        if blender_formats:
            add_log(f"Exporting: {blender_formats}")
            yield {
                status_html: create_status_html("🟧 Blender export..."),
                debug_console: logs
            }
            blender_outputs = export_blender(npz_path, job_dir / "hair", blender_formats)
            output_files.extend(blender_outputs)
            add_log(f"Blender exports: {len(blender_outputs)} files")
        
        # Create ZIP
        zip_path = job_dir / "DiffLocks_Results.zip"
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in output_files:
                if Path(f).exists():
                    zf.write(f, Path(f).name)
        add_log("ZIP created")
        
        add_log("✅ Complete!")
        
        # Final result
        yield {
            plot_3d: plot_3d_fig,
            preview_2d: preview_img,
            status_html: create_status_html("✅ Generation complete!", status_type="success"),
            result_group: gr.Group(visible=True),
            download_file: str(zip_path),
            debug_console: logs,
            generate_btn: gr.Button(interactive=True)
        }
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        add_log(error_msg)
        add_log(traceback.format_exc())
        
        yield {
            plot_3d: None,
            preview_2d: None,
            status_html: create_status_html(error_msg, status_type="error"),
            result_group: gr.Group(visible=False),
            download_file: None,
            debug_console: logs,
            generate_btn: gr.Button(interactive=True)
        }

# Apply ZeroGPU decorator if available
if HAS_ZEROGPU:
    run_inference = spaces.GPU(duration=120)(run_inference)

# --- 6. GRADIO UI ---

CSS = """
.gradio-container { max-width: 100% !important; }
footer { display: none !important; }

.cpu-warning {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 1px solid #f59e0b;
    color: #92400e;
    padding: 15px;
    border-radius: 8px;
    margin-bottom: 20px;
    text-align: center;
}

.debug-console textarea {
    font-family: 'Consolas', 'Monaco', monospace !important;
    font-size: 11px !important;
    background: #0d1117 !important;
    color: #c9d1d9 !important;
}

/* Notification Sound */
#notification-trigger { display: none; }
"""

JS_NOTIFY = """
function() {
    // Sound notification
    const audio = new Audio('https://actions.google.com/sounds/v1/alarms/beep_short.ogg');
    audio.play().catch(e => console.log("Sound blocked by browser"));
    
    // Browser notification
    if (Notification.permission === "granted") {
        new Notification("DiffLocks Studio", { body: "✨ Generation Complete!" });
    } else if (Notification.permission !== "denied") {
        Notification.requestPermission();
    }
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), css=CSS, title="DiffLocks Studio") as demo:
    
    gr.Markdown("## 💇‍♀️ DiffLocks Studio")
    gr.Markdown("*AI-powered 3D hair generation from a single image*")
    
    # CPU Warning Banner
    if IS_CPU:
        gr.HTML("""
        <div class="cpu-warning">
            <b>⚠️ RUNNING IN CPU MODE</b><br>
            Generation will be very slow (hours). For fast results:<br>
            • Apply for <b>ZeroGPU</b> grant<br>
            • Use <a href='https://github.com/arqdariogomez/DiffLocks-Studio' target='_blank'><b>Colab/Kaggle</b></a> with GPU
        </div>
        """)
    
    with gr.Row():
        # Left Panel - Controls
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="filepath",
                label="Input Portrait",
                height=300
            )
            
            cfg_slider = gr.Slider(
                minimum=1, maximum=7, value=2.5, step=0.1,
                label="CFG Scale",
                info="Higher = stiffer hair, Lower = messier hair"
            )
            
            format_checkboxes = gr.CheckboxGroup(
                choices=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"],
                value=["Blender (.blend)", "Alembic (.abc)", "USD (.usd)"],
                label="Additional Exports",
                info="OBJ is always included"
            )
            
            generate_btn = gr.Button(
                "🚀 GENERATE HAIR",
                variant="primary",
                size="lg"
            )
            
            with gr.Accordion("⚙️ System", open=False):
                update_btn = gr.Button("🔄 Check for Updates")
                update_status = gr.Markdown("")
                
                # Hidden trigger for JS notification
                notify_trigger = gr.Button("Notify", elem_id="notification-trigger", visible=False)
        
        # Right Panel - Results
        with gr.Column(scale=2):
            status_html = gr.HTML(
                value=create_status_html("⏳ Upload an image and click Generate")
            )
            
            with gr.Group(visible=False) as result_group:
                with gr.Tabs():
                    with gr.Tab("🎨 3D Preview"):
                        plot_3d = gr.Plot(show_label=False)
                    with gr.Tab("📸 2D Renders"):
                        preview_2d = gr.Image(show_label=False, interactive=False)
                
                download_file = gr.File(label="📥 Download Results")
            
            with gr.Accordion("🛠️ Debug Console", open=False):
                debug_console = gr.Textbox(
                    lines=10,
                    max_lines=20,
                    interactive=False,
                    elem_classes=["debug-console"],
                    show_label=False,
                    value=""
                )
    
    # Connect events
    generate_btn.click(
        fn=run_inference,
        inputs=[image_input, cfg_slider, format_checkboxes, debug_console],
        outputs=[plot_3d, preview_2d, status_html, result_group, download_file, debug_console, generate_btn]
    ).then(
        fn=None,
        _js=JS_NOTIFY
    )
    
    update_btn.click(
        fn=check_for_updates,
        outputs=[update_status]
    )

# --- 7. LAUNCH ---
if __name__ == "__main__":
    demo.queue().launch(
        share=cfg.needs_share,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True
    )
