
print("--- SCRIPT START ---")
import gradio as gr
import numpy as np
import plotly.graph_objects as go
import os
import sys
import time
from pathlib import Path

# Add current dir to path to import app
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app import generate_preview_3d, create_empty_3d_plot, CSS, js_func, dark_theme, render_image_html

def log_to_file(msg):
    with open("mvp_debug.log", "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} - {msg}\n")

class MockLog:
    def add_log(self, msg):
        print(f"[3D LOG] {msg}")
        log_to_file(f"[3D LOG] {msg}")

def test_interface():
    log_to_file("Starting MVP interface...")
    outputs_dir = project_root / "studio_outputs"
    if not outputs_dir.exists():
        print(f"❌ studio_outputs directory not found at {outputs_dir}")
        log_to_file(f"❌ studio_outputs not found at {outputs_dir}")
        return
        
    # Prioritize full NPZ files, then fall back to previews
    # Full NPZ is usually difflocks_output_strands.npz
    # Preview is usually difflocks_output_strands_preview.npz
    all_npzs = list(outputs_dir.rglob("*.npz"))
    
    # Filter for full NPZ (the one without '_preview')
    full_npzs = [f for f in all_npzs if not f.name.endswith("_preview.npz")]
    previews = [f for f in all_npzs if f.name.endswith("_preview.npz")]
    
    if full_npzs:
        full_npzs.sort(key=os.path.getmtime, reverse=True)
        target_npz = full_npzs[0]
        print(f"🚀 Using FULL NPZ: {target_npz}")
        log_to_file(f"🚀 Using FULL NPZ: {target_npz}")
    elif previews:
        previews.sort(key=os.path.getmtime, reverse=True)
        target_npz = previews[0]
        print(f"⚠️ FULL NPZ not found, falling back to PREVIEW: {target_npz}")
        log_to_file(f"⚠️ Fallback to PREVIEW: {target_npz}")
    else:
        print("❌ No NPZ files found in studio_outputs")
        log_to_file("❌ No NPZ files found")
        return

    with gr.Blocks(theme=dark_theme, css=CSS, js=js_func) as demo:
        gr.Markdown("# 🧪 3D Preview Debugger MVP")
        gr.Markdown(f"Testing file: `{target_npz.name}`")
        
        with gr.Row():
            with gr.Column(scale=4):
                btn = gr.Button("🔄 Force Reload 3D", variant="primary")
                btn_test = gr.Button("🧪 Test Plotly 3D (Hello World)")
                btn_test_2d = gr.Button("📊 Test Plotly 2D (Hello World)")
                status_box = gr.Textbox(label="Status/Errors", interactive=False)
            
            with gr.Column(scale=6):
                with gr.Group():
                    with gr.Accordion("🎨 Interactive 3D Preview", open=True) as plot_3d_accordion:
                        plot_3d = gr.Plot(label="Gradio Plot (Standard)")
                        plot_html = gr.HTML(label="HTML Plot (Fallback)")

        def reload_3d():
            try:
                print(f"🔄 reload_3d() triggered for {target_npz}")
                log_to_file(f"🔄 reload_3d() triggered")
                fig = generate_preview_3d(str(target_npz), log_capture=MockLog())
                if fig:
                    print(f"✅ Figure generated successfully")
                    html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
                    return fig, html_str, "✅ Success"
                else:
                    print("❌ generate_preview_3d returned None")
                    return None, "", "⚠️ generate_preview_3d returned None"
            except Exception as e:
                import traceback
                err = traceback.format_exc()
                print(f"💥 Error in reload_3d: {err}")
                log_to_file(f"💥 Error: {err}")
                return None, "", f"💥 Error: {str(e)}"

        def hello_world_plot():
            print("🧪 hello_world_plot() triggered")
            fig = go.Figure(data=[go.Scatter3d(
                x=[0, 1, 0.5], y=[0, 0, 1], z=[0, 0, 0.5], 
                mode='markers+lines',
                marker=dict(size=10, color='red'),
                line=dict(width=5, color='blue')
            )])
            fig.update_layout(title="Hello World 3D", paper_bgcolor='#18181b', font=dict(color='white'))
            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return fig, html_str, "🧪 Hello World 3D Plot Displayed"

        def hello_world_plot_2d():
            print("📊 hello_world_plot_2d() triggered")
            fig = go.Figure(data=[go.Scatter(
                x=[0, 1, 2, 3], y=[10, 11, 12, 13], 
                mode='lines+markers',
                marker=dict(size=10, color='green'),
                line=dict(width=3, color='orange')
            )])
            fig.update_layout(title="Hello World 2D", paper_bgcolor='#18181b', font=dict(color='white'))
            html_str = fig.to_html(full_html=False, include_plotlyjs='cdn')
            return fig, html_str, "📊 Hello World 2D Plot Displayed"

        btn.click(fn=reload_3d, outputs=[plot_3d, plot_html, status_box])
        btn_test.click(fn=hello_world_plot, outputs=[plot_3d, plot_html, status_box])
        btn_test_2d.click(fn=hello_world_plot_2d, outputs=[plot_3d, plot_html, status_box])
        
        # Auto-load on start
        demo.load(fn=reload_3d, outputs=[plot_3d, plot_html, status_box])

    return demo

if __name__ == "__main__":
    demo = test_interface()
    if demo:
        # Use a higher port to avoid common conflicts
        demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7888,
        share=False,
        debug=True,
        show_error=True
    )
