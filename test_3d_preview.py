import numpy as np
import plotly.graph_objects as go
import os

def test_preview(job_id):
    # Use relative paths to be platform agnostic
    base_path = os.path.join("studio_outputs", job_id)
    full_npz = os.path.join(base_path, "difflocks_output_strands.npz")
    prev_npz = os.path.join(base_path, "difflocks_output_strands_preview.npz")
    
    print(f"--- Testing Job: {job_id} ---")
    
    if not os.path.exists(full_npz):
        print(f"❌ Full NPZ not found: {full_npz}")
        return

    # 1. Check Full NPZ
    data_full = np.load(full_npz)
    pos_full = data_full['positions']
    print(f"✅ Full NPZ shape: {pos_full.shape}")
    print(f"Full NPZ stats - Min: {pos_full.min(axis=(0,1))}, Max: {pos_full.max(axis=(0,1))}")

    # 2. Check Preview NPZ
    if os.path.exists(prev_npz):
        data_prev = np.load(prev_npz)
        pos_prev = data_prev['positions']
        print(f"✅ Preview NPZ shape: {pos_prev.shape}")
        print(f"Preview NPZ stats - Min: {pos_prev.min(axis=(0,1))}, Max: {pos_prev.max(axis=(0,1))}")
        
        # Calculate expected size for float32 (4 bytes)
        expected_bytes = pos_prev.size * 4
        actual_kb = os.path.getsize(prev_npz) / 1024
        print(f"Preview NPZ Disk Size: {actual_kb:.2f} KB (Expected raw: {expected_bytes/1024:.2f} KB)")
    else:
        print(f"⚠️ Preview NPZ not found: {prev_npz}")

    # 3. Simulate app.py transformation
    print("\n--- Simulating app.py transformation ---")
    # Take a subset
    subset = pos_full[:100] # 100 strands
    
    # Transformation: [x, y, z] -> [x, -z, y] (to make it upright and front-facing)
    # This is what app.py does: x, -z, y
    transformed = subset.copy()
    transformed = transformed[:, :, [0, 2, 1]]
    transformed[:, :, 1] = -transformed[:, :, 1]
    
    print(f"Transformed stats - Min: {transformed.min(axis=(0,1))}, Max: {transformed.max(axis=(0,1))}")
    
    if np.isnan(transformed).any():
        print("❌ CRITICAL: NaNs found in transformed data!")
    else:
        print("✅ No NaNs found in transformed data.")

    # Check for zero variance (flat model)
    ranges = transformed.max(axis=(0,1)) - transformed.min(axis=(0,1))
    print(f"Data Ranges (X, Y, Z): {ranges}")
    if any(ranges < 1e-5):
        print("⚠️ Warning: Data seems flat in one or more dimensions.")
    
    # 2. Check Preview NPZ if exists
    if os.path.exists(prev_npz):
        data_prev = np.load(prev_npz)
        pos_prev = data_prev['positions']
        print(f"Preview NPZ shape: {pos_prev.shape}")
        print(f"Preview NPZ stats - Min: {pos_prev.min(axis=(0,1))}, Max: {pos_prev.max(axis=(0,1))}")
        subset = pos_prev
    else:
        print("⚠️ Preview NPZ not found, creating from full...")
        # Downsample logic from app.py
        num_strands, points_per_strand, _ = pos_full.shape
        target_strands = 500
        strand_step = max(1, num_strands // target_strands)
        target_points = 32
        point_step = max(1, points_per_strand // target_points)
        subset = pos_full[::strand_step, ::point_step, :]
        print(f"Created subset shape: {subset.shape}")

    # 3. Simulate Plotly logic from app.py
    n_s, n_p, _ = subset.shape
    
    # Coordinate transformation
    x = subset[:, :, 0]
    y = -subset[:, :, 2]
    z = subset[:, :, 1]
    
    # Rotation
    theta = np.radians(180)
    c, s = np.cos(theta), np.sin(theta)
    fx = x * c + y * s
    fy = -x * s + y * c
    fz = z
    
    # Check for NaN/Inf in transformed data
    if np.isnan(fx).any() or np.isinf(fx).any():
        print("❌ Detected NaN/Inf in fx!")
    
    nan_col = np.full((n_s, 1), np.nan)
    x_plot = np.hstack([fx, nan_col]).flatten()
    y_plot = np.hstack([fy, nan_col]).flatten()
    z_plot = np.hstack([fz, nan_col]).flatten()
    
    print(f"Plot arrays length: {len(x_plot)}")
    
    # Colors
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
    
    # Create Figure
    fig = go.Figure(data=[go.Scatter3d(
        x=x_plot, y=y_plot, z=z_plot,
        mode='lines',
        line=dict(width=3, color=color_array, colorscale='Viridis', showscale=False),
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1]),
            aspectmode='cube'
        ),
        title=f"Debug 3D Preview - {job_id}"
    )
    
    output_html = f"debug_3d_{job_id}.html"
    fig.write_html(output_html)
    print(f"✅ Debug HTML saved to: {output_html}")

if __name__ == "__main__":
    # Test with the latest job found in the list
    test_preview("job_1766624076")
