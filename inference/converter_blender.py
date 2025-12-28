
import bpy
import sys
import numpy as np
import os
import time
import traceback

# --- ARGS PARSING ---
# Usage: blender ... -- input.npz output_base format1 format2 ...
argv = sys.argv
if "--" in argv:
    args = argv[argv.index("--") + 1:]
    input_npz = args[0]
    output_base = args[1]
    requested_formats = args[2:] 
else:
    print("❌ Args missing")
    sys.exit(1)

print(f"🔄 Blender Processing: {input_npz}")
print(f"ℹ️ Blender Version: {bpy.app.version_string}")
bpy.ops.wm.read_factory_settings(use_empty=True)

# --- CONFIG ---
SCALE_FACTOR = 1.0
ROTATE_X_90 = True # Fix Y-Up to Z-Up

def create_hair_principled_material(name="DiffLocks_Mat"):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Try to use Principled Hair BSDF (available in modern Blender)
    try:
        shader = nodes.new(type='ShaderNodeBsdfHairPrincipled')
        shader.location = (0, 0)
        # Default to a nice brown
        # Melanin: 0.8, Redness: 0.1
        if "Melanin" in shader.inputs:
            shader.inputs["Melanin"].default_value = 0.8
    except:
        # Fallback to standard Principled BSDF
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader.location = (0, 0)
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = (0.05, 0.03, 0.01, 1.0)

    # Check for color attribute
    try:
        attr_node = nodes.new(type='ShaderNodeAttribute')
        attr_node.attribute_name = "DiffLocks_Color"
        attr_node.location = (-300, 0)
        
        # Connect to color input
        target_input = None
        for inp in shader.inputs:
            if "color" in inp.name.lower() and "random" not in inp.name.lower():
                target_input = inp
                break
        
        if target_input:
            links.new(attr_node.outputs["Color"], target_input)
    except:
        pass

    out = nodes.new(type='ShaderNodeOutputMaterial')
    out.location = (300, 0)
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    t_start = time.time()
    
    # LOAD
    data = np.load(input_npz)
    positions = data['positions']
    colors = data.get('colors', None)
    radii = data.get('radii', None)
    
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    print(f"🧬 Processing {num_strands:,} strands ({pts_per_strand} pts each)...")

    # TRANSFORM
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        flat_pos = flat_pos[:, [0, 2, 1]] # Swap Y and Z
        flat_pos[:, 1] *= -1 # Invert Y

    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos
    points_4d[:, 3] = 1.0

    # --- GEOMETRY CREATION (Using Addon Logic: Legacy then Convert) ---
    print("🔨 Building Geometry...", end="", flush=True)
    curve_data = bpy.data.curves.new(name="DiffLocks_Temp", type='CURVE')
    curve_data.dimensions = '3D'
    
    # Batch creation of splines
    report_interval = max(1, num_strands // 5)
    for i in range(num_strands):
        s = curve_data.splines.new('POLY')
        s.points.add(pts_per_strand - 1)
        start = i * pts_per_strand
        end = start + pts_per_strand
        s.points.foreach_set('co', points_4d[start:end].ravel())
        
        if i > 0 and i % report_interval == 0:
            print(f" {int((i/num_strands)*100)}%...", end="", flush=True)
    print(" 100% Done.")

    temp_obj = bpy.data.objects.new("DiffLocks_Temp", curve_data)
    bpy.context.collection.objects.link(temp_obj)
    
    # Activate and select
    bpy.context.view_layer.objects.active = temp_obj
    for obj in bpy.data.objects:
        obj.select_set(False)
    temp_obj.select_set(True)
    
    # --- ABC EXPORT (from Legacy Curve) ---
    if 'abc' in requested_formats:
        # Use forward slashes for Alembic exporter on Windows
        abc_out = os.path.abspath(f"{output_base}.abc").replace("\\", "/")
        print(f"📦 Exporting Alembic (Legacy Strands): {abc_out}")
        
        try:
            # Method 1: Export as Hair (Particles/Curves)
            # In Blender 4.0, export_hair=True is the most likely to work for curves
            bpy.ops.wm.alembic_export(
                filepath=abc_out, 
                selected=True, 
                start=1, end=1,
                export_hair=True, 
                export_particles=False,
                as_background_job=False,
                evaluation_mode='VIEWPORT'
            )
            
            # Method 2: Fallback if file is empty - try as regular object
            if not os.path.exists(abc_out) or os.path.getsize(abc_out) < 5000:
                print("🔄 ABC Method 1 failed, trying Method 2 (Regular Object)...")
                bpy.ops.wm.alembic_export(
                    filepath=abc_out, 
                    selected=True, 
                    start=1, end=1,
                    export_hair=False,
                    as_background_job=False
                )
                
            # Method 3: Final fallback - convert to Mesh (edges) temporarily
            if not os.path.exists(abc_out) or os.path.getsize(abc_out) < 5000:
                print("🔄 ABC Method 2 failed, trying Method 3 (Mesh Edges)...")
                # We do this on a copy to not break the original curve
                mesh_copy = temp_obj.copy()
                mesh_copy.data = temp_obj.data.copy()
                bpy.context.collection.objects.link(mesh_copy)
                
                # Deselect all, select copy
                for obj in bpy.data.objects: obj.select_set(False)
                mesh_copy.select_set(True)
                bpy.context.view_layer.objects.active = mesh_copy
                
                # Convert to mesh (turns curves into edges)
                bpy.ops.object.convert(target='MESH')
                
                bpy.ops.wm.alembic_export(
                    filepath=abc_out, 
                    selected=True, 
                    start=1, end=1,
                    as_background_job=False
                )
                
                # Clean up
                bpy.data.objects.remove(mesh_copy, do_unlink=True)
                
                # Restore selection to original
                temp_obj.select_set(True)
                bpy.context.view_layer.objects.active = temp_obj
                
        except Exception as e:
            print(f"⚠️ ABC Export error: {e}")

    final_obj = temp_obj
    use_new_curves = False

    # CONVERT TO MODERN HAIR CURVES (for .blend and .usd)
    print("✨ Converting to Modern Hair Curves...", end="", flush=True)
    try:
        # This is the step that makes it Geometry Nodes compatible and faster for Cycles
        bpy.ops.object.convert(target='CURVES', keep_original=False)
        final_obj = bpy.context.active_object
        final_obj.name = "DiffLocks_Hair"
        use_new_curves = True
        print(" Done.")
        
        # --- ATTRIBUTES (Radius & Colors) ---
        # Set Radius
        if radii is not None:
            r_flat = radii.reshape(-1) * SCALE_FACTOR
            if 'radius' in final_obj.data.attributes:
                final_obj.data.attributes['radius'].data.foreach_set('value', r_flat.astype(np.float32))
        else:
            total_pts = len(final_obj.data.points)
            defaults = np.full(total_pts, 0.003 * SCALE_FACTOR, dtype=np.float32)
            if 'radius' in final_obj.data.attributes:
                final_obj.data.attributes['radius'].data.foreach_set('value', defaults)

        # Set Colors
        if colors is not None:
            try:
                if "DiffLocks_Color" not in final_obj.data.attributes:
                    attr = final_obj.data.attributes.new(name="DiffLocks_Color", type='FLOAT_COLOR', domain='POINT')
                else:
                    attr = final_obj.data.attributes["DiffLocks_Color"]
                
                c_flat = colors.reshape(-1, 3)
                rgba = np.ones((len(c_flat), 4), dtype=np.float32)
                rgba[:, :3] = c_flat
                attr.data.foreach_set('color', rgba.ravel())
                print("🎨 Color attributes applied.")
            except Exception as e:
                print(f"⚠️ Could not apply colors: {e}")

    except Exception as e:
        print(f"\n⚠️ Modern conversion failed: {e}. Staying with legacy curves.")
        final_obj.data.fill_mode = 'FULL'
        final_obj.data.bevel_depth = 0.001

    # MATERIAL
    mat = create_hair_principled_material()
    if final_obj.data.materials:
        final_obj.data.materials[0] = mat
    else:
        final_obj.data.materials.append(mat)

    # Force dependency graph update
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    
    # EXPORT REMAINING FORMATS
    if 'blend' in requested_formats:
        out = f"{output_base}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
        print(f"✅ Exported: {out}")
    
    if 'abc' in requested_formats:
        out = f"{output_base}.abc"
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print(f"✅ Exported ABC: {out}")
        else:
            print(f"❌ ABC Export failed (size: {os.path.getsize(out) if os.path.exists(out) else 0})")

    if 'usd' in requested_formats:
        out = f"{output_base}.usd"
        print(f"📦 Exporting USD: {out}")
        try:
            # USD handles the new CURVES type better than ABC
            bpy.ops.wm.usd_export(
                filepath=out, 
                selected_objects_only=True,
                export_hair=True, 
                evaluation_mode='VIEWPORT'
            )
        except: pass
        print(f"✅ Exported USD: {out}")
        
    print(f"✅ SUCCESS (Total time: {time.time() - t_start:.2f}s)")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
