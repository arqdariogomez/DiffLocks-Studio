
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

    # --- GEOMETRY CREATION (Modern CURVES API for Blender 3.3+) ---
    print("🔨 Building Geometry (Modern Curves)...", end="", flush=True)
    
    # Create the curves data
    curve_data = bpy.data.curves.new(name="DiffLocks_Hair", type='CURVES')
    hair_obj = bpy.data.objects.new("DiffLocks_Hair", curve_data)
    bpy.context.collection.objects.link(hair_obj)
    
    # Efficiently add curves and points
    # In Blender 4.0+, we use the curves.attributes and point attributes
    total_points = num_strands * pts_per_strand
    
    # Create points and curves in one go
    curve_data.curves.add(num_strands)
    curve_data.points.add(num_strands * pts_per_strand)
    
    # Set the number of points per curve (uniform in our case)
    # The attribute is 'points_length' in recent Blender versions
    if 'points_length' in curve_data.attributes:
        curve_data.attributes['points_length'].data.foreach_set('value', [pts_per_strand] * num_strands)
    
    # Set positions using foreach_set (very fast)
    curve_data.attributes['position'].data.foreach_set('vector', flat_pos.astype(np.float32).ravel())
    
    print(" 100% Done.")

    # Activate and select
    bpy.context.view_layer.objects.active = hair_obj
    for obj in bpy.data.objects:
        obj.select_set(False)
    hair_obj.select_set(True)
    
    # --- ABC EXPORT ---
    if 'abc' in requested_formats:
        abc_out = os.path.abspath(f"{output_base}.abc").replace("\\", "/")
        print(f"📦 Exporting Alembic: {abc_out}")
        try:
            bpy.ops.wm.alembic_export(
                filepath=abc_out, 
                selected=True, 
                start=1, end=1,
                export_hair=True, 
                evaluation_mode='VIEWPORT'
            )
        except Exception as e:
            print(f"⚠️ ABC Export error: {e}")

    final_obj = hair_obj
    
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
            # For modern CURVES, we use 'FLOAT_COLOR' on 'POINT' or 'CURVE' domain
            # We'll use POINT domain for per-vertex color
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
    created_any = False
    
    if 'blend' in requested_formats:
        out = f"{output_base}.blend"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                print(f"✅ Exported: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
                created_any = True
            else:
                print(f"❌ Failed to create valid .blend file: {out}")
        except Exception as e:
            print(f"❌ Blender .blend export error: {e}")
    
    if 'abc' in requested_formats:
        out = f"{output_base}.abc"
        # Selection for ABC
        final_obj.select_set(True)
        bpy.context.view_layer.objects.active = final_obj
        try:
            bpy.ops.wm.alembic_export(filepath=out, selected=True, visible_objects_only=False, flatten=True)
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                print(f"✅ Exported: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
                created_any = True
            else:
                print(f"❌ ABC Export failed or zero size: {out}")
        except Exception as e:
            print(f"❌ Blender ABC export error: {e}")

    if 'usd' in requested_formats:
        out = f"{output_base}.usd"
        try:
            bpy.ops.wm.usd_export(
                filepath=out, 
                selected_objects_only=True,
                export_hair=True, 
                evaluation_mode='VIEWPORT'
            )
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                print(f"✅ Exported: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
                created_any = True
            else:
                print(f"❌ USD Export failed or zero size: {out}")
        except Exception as e:
            print(f"❌ Blender USD export error: {e}")
            
    if not created_any:
        print("⚠️ No Blender files were created. This usually happens if Blender crashes or out of memory.")
        
    print(f"✅ SUCCESS (Total time: {time.time() - t_start:.2f}s)")

except Exception as e:
    traceback.print_exc()
    sys.exit(1)
