
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import os

def debug_npz():
    project_root = Path(os.getcwd())
    outputs_dir = project_root / "studio_outputs"
    previews = list(outputs_dir.rglob("*_preview.npz"))
    if not previews:
        print("No NPZ found")
        return
    
    target_npz = sorted(previews, key=os.path.getmtime, reverse=True)[0]
    print(f"Checking: {target_npz}")
    
    data = np.load(target_npz)
    positions = data['positions']
    print(f"Shape: {positions.shape}")
    print(f"X range: {positions[:,:,0].min():.4f} to {positions[:,:,0].max():.4f}")
    print(f"Y range: {positions[:,:,1].min():.4f} to {positions[:,:,1].max():.4f}")
    print(f"Z range: {positions[:,:,2].min():.4f} to {positions[:,:,2].max():.4f}")

    # Test the transformation logic from app.py
    num_strands, points_per_strand, _ = positions.shape
    target_strands = 600
    strand_step = max(1, num_strands // target_strands)
    target_points = 24
    point_step = max(1, points_per_strand // target_points)
    subset = positions[::strand_step, ::point_step, :]
    
    n_s, n_p, _ = subset.shape
    x_base = subset[:, :, 0]
    y_base = -subset[:, :, 2]
    z_base = subset[:, :, 1]
    
    theta = np.radians(180)
    c, s = np.cos(theta), np.sin(theta)
    fx = x_base * c + y_base * s
    fy = -x_base * s + y_base * c
    fz = z_base
    
    print(f"Transformed FX range: {fx.min():.4f} to {fx.max():.4f}")
    print(f"Transformed FY range: {fy.min():.4f} to {fy.max():.4f}")
    print(f"Transformed FZ range: {fz.min():.4f} to {fz.max():.4f}")

    # Save a test HTML to see if it renders
    nan_col = np.full((n_s, 1), np.nan)
    x_plot = np.hstack([fx, nan_col]).flatten()
    y_plot = np.hstack([fy, nan_col]).flatten()
    z_plot = np.hstack([fz, nan_col]).flatten()
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='lines',
        line=dict(width=2, color='white')
    )])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.write_html("test_debug_render.html")
    print("Saved test_debug_render.html")

if __name__ == "__main__":
    debug_npz()
