
import os
import sys
import time
import shutil
import subprocess
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
import traceback
import torch
from huggingface_hub import snapshot_download

# --- 0. DIAGNOSTICS ---
print("🔍 SYSTEM DIAGNOSTICS")
token = os.environ.get("HF_TOKEN")
if token: print(f"✅ HF_TOKEN found")
else: print("❌ FATAL: HF_TOKEN NOT FOUND")

# --- 1. SETUP ---
try:
    from platform_config import cfg
except ImportError:
    sys.path.append(".")
    from platform_config import cfg

if str(cfg.repo_dir) not in sys.path:
    sys.path.append(str(cfg.repo_dir))

ASSETS_REPO_ID = "arqdariogomez/difflocks-assets-hybrid"

if cfg.platform == 'huggingface':
    # Install Blender
    if not cfg.blender_exe.exists():
        print("📦 Installing Blender...")
        b_dir = Path("/tmp/blender"); b_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run("wget -q -O /tmp/blender.tar.xz https://download.blender.org/release/Blender4.2/blender-4.2.5-linux-x64.tar.xz", shell=True)
        subprocess.run(f"tar -xf /tmp/blender.tar.xz -C {b_dir} --strip-components=1", shell=True)

    # Download Checkpoints
    # Buscamos si ya tenemos AL MENOS el modelo principal
    if not list(cfg.repo_dir.rglob("*.pth")):
        print("🧠 Downloading Assets...")
        try:
            snapshot_download(
                repo_id=ASSETS_REPO_ID,
                repo_type="dataset",
                # Bajamos checkpoints y también la raíz por si el .pt está suelto
                allow_patterns=["checkpoints/*", "assets/*", "*.pt"],
                local_dir=cfg.repo_dir, 
                token=token
            )
            
            # Organizar assets de Blender
            src_assets = cfg.repo_dir / "assets"
            dst_assets = cfg.repo_dir / "inference/assets"
            if src_assets.exists():
                dst_assets.mkdir(parents=True, exist_ok=True)
                for f in src_assets.glob("*"):
                    shutil.move(str(f), str(dst_assets / f.name))
                shutil.rmtree(src_assets)
                
        except Exception as e:
            print(f"❌ DOWNLOAD ERROR: {e}")

# --- 2. MODEL LOADER ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IS_CPU = (DEVICE == "cpu")

from inference.img2hair import DiffLocksInference

# BÚSQUEDA PROFUNDA DE ARCHIVOS (FIX)
# Buscamos en TODO el repo, no solo en la carpeta checkpoints
ckpt_files = list(cfg.repo_dir.rglob("*diffusion*.pth")) or list(cfg.repo_dir.rglob("*.pth"))
vae_files = list(cfg.repo_dir.rglob("strand_codec.pt"))
conf_path = cfg.repo_dir / "configs/config_scalp_texture_conditional.json"

model = None

def load_model():
    global model
    if model is not None: return
    
    # Diagnóstico detallado si falta algo
    if not ckpt_files or not vae_files:
        debug_msg = f"PTH Found: {len(ckpt_files)}, PT Found: {len(vae_files)}"
        print(f"Files in repo: {list(cfg.repo_dir.rglob('*'))}")
        raise FileNotFoundError(f"Missing Files! {debug_msg}")
    
    print(f"Loading Model: {ckpt_files[0].name} / {vae_files[0].name}")
    model = DiffLocksInference(str(vae_files[0]), str(conf_path), str(ckpt_files[0]), DEVICE)

# ... (Resto del código igual: run_inference, UI, etc) ...
try:
    import spaces
    has_zerogpu = True
except: has_zerogpu = False

def run_inference(img, cfg_val, fmts):
    job_id = f"j_{int(time.time())}"
    job_dir = cfg.output_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        if img is None: raise ValueError("Image required")
        if model is None: load_model()
        
        img_p = job_dir / "input.png"
        if isinstance(img, str): shutil.copy(img, img_p)
        else: img.save(img_p)
        
        model.cfg_val = float(cfg_val)
        for update in model.file2hair(str(img_p), str(job_dir)):
            if isinstance(update, tuple) and update[0] == "status":
                yield None, None, f"⚙️ {update[1]}", gr.Group(visible=False), None
        
        npz_path = job_dir / "difflocks_output_strands.npz"
        yield None, None, "🎨 Rendering Preview...", gr.Group(visible=False), None
        
        prev_path = None; fig_3d = None
        try:
            d=np.load(npz_path)['positions']; s=d[::max(1,len(d)//20000)]; p=s.reshape(-1,3); x,y,z=p[:,0],p[:,1],p[:,2]; rx,ry,rz=x,-z,y
            plt.style.use('dark_background'); f,ax=plt.subplots(1,3,figsize=(18,6)); f.patch.set_facecolor('#111')
            for a,u,v,d_ in zip(ax,[rx,ry,rx],[rz,rz,ry],[ry,np.abs(rx),rz]):
                idx=np.argsort(d_); a.scatter(u[idx],v[idx],c=(d_[idx]-d_.min())/(d_.max()-d_.min()+1e-8),cmap='copper',s=0.5,lw=0); a.axis('off')
            prev_path = job_dir/"prev.png"
            f.savefig(prev_path, facecolor='#111', bbox_inches='tight'); plt.close(f)
            
            s3=d[::max(1,len(d)//30000)][:,::8,:]; p3=s3.reshape(-1,3); x3,y3,z3=p3[:,0],p3[:,1],p3[:,2]; rx3,ry3,rz3=x3,-z3,y3
            c3=np.hstack([np.tile(np.linspace(0.3,1,s3.shape[1]),(s3.shape[0],1)),np.zeros((s3.shape[0],1))]).flatten()
            fig_3d = go.Figure(data=[go.Scatter3d(x=np.hstack([rx3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), y=np.hstack([ry3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), z=np.hstack([rz3.reshape(s3.shape[:2]),np.full((s3.shape[0],1),np.nan)]).flatten(), mode='lines', line=dict(width=1.5,color=c3,colorscale=[[0,'#505050'],[1,'white']],showscale=False))], layout=dict(paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='rgba(0,0,0,0)',scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False),bgcolor='rgba(0,0,0,0)'),margin=dict(l=0,r=0,b=0,t=0),height=500))
        except: pass

        outputs = [str(npz_path)]
        blender_keys = [k for k,v in {'.blend':'Blender','.abc':'Alembic','.usd':'USD'}.items() if v in fmts]
        if blender_keys and cfg.blender_exe.exists():
            yield None, None, "🟧 Blender Export...", gr.Group(visible=False), None
            script = cfg.repo_dir / "inference/converter_blender.py"
            cmd = [str(cfg.blender_exe), "-b", "-P", str(script), "--", str(npz_path), str(job_dir/"hair")] + [k.replace('.','') for k in blender_keys]
            subprocess.run(cmd)
            for k in blender_keys:
                f = job_dir / f"hair{k}"; (outputs.append(str(f)) if f.exists() else None)

        import zipfile
        zip_f = job_dir / "results.zip"
        with zipfile.ZipFile(zip_f, 'w') as z: [z.write(f, Path(f).name) for f in outputs]
        
        yield fig_3d, str(prev_path), "✅ Done!", gr.Group(visible=True), str(zip_f)

    except Exception as e:
        err_msg = f"❌ Error: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        print(err_msg)
        yield None, None, f"<div style='color:red; white-space:pre-wrap'>{err_msg}</div>", gr.Group(visible=False), None

if has_zerogpu:
    run_inference = spaces.GPU(duration=120)(run_inference)

css = """
.cpu-warning { background-color: #fffbeb; border: 1px solid #f59e0b; color: #92400e; padding: 15px; border-radius: 8px; margin-bottom: 20px; font-size: 16px; text-align: center; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css, title="DiffLocks Studio") as demo:
    gr.Markdown("# 💇‍♀️ DiffLocks Studio (Universal)")
    if IS_CPU: gr.HTML('<div class="cpu-warning"><b>⚠️ RUNNING IN CPU MODE (SLOW)</b><br>For fast inference, run on Colab/Kaggle or apply for ZeroGPU.</div>')
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="filepath", label="Input Face")
            cfg_s = gr.Slider(1, 7, 2.5, label="CFG Scale")
            chk = gr.CheckboxGroup(["Blender", "Alembic", "USD"], label="Exports", value=[])
            btn = gr.Button("🚀 Generate", variant="primary")
        with gr.Column():
            status = gr.HTML("Ready")
            with gr.Group(visible=False) as res_grp:
                with gr.Tabs():
                    with gr.Tab("3D"): p3d = gr.Plot()
                    with gr.Tab("2D"): p2d = gr.Image()
                files = gr.File(label="Download ZIP")
    btn.click(run_inference, [inp, cfg_s, chk], [p3d, p2d, status, res_grp, files])

if __name__ == "__main__":
    demo.queue().launch(share=cfg.needs_share, server_name="0.0.0.0", server_port=7860)
