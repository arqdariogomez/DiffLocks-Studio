
import bpy
import sys
import numpy as np
import os

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

def create_material():
    mat = bpy.data.materials.new(name="DiffLocks_Mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    shader = nodes.new(type='ShaderNodeBsdfPrincipled') 
    shader.inputs["Base Color"].default_value = (0.05, 0.03, 0.01, 1.0) # Dark Brown
    shader.inputs["Roughness"].default_value = 0.5
    out = nodes.new(type='ShaderNodeOutputMaterial')
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    # LOAD
    data = np.load(input_npz)
    positions = data['positions']
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    # TRANSFORM
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        flat_pos = flat_pos[:, [0, 2, 1]] # Swap Y and Z
        flat_pos[:, 1] *= -1 # Invert Y

    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos
    points_4d[:, 3] = 1.0

    # CURVE
    curve_data = bpy.data.curves.new(name="Hair", type='CURVE')
    curve_data.dimensions = '3D'
    
    # --- CURVE SETTINGS (Legacy for compatibility) ---
    curve_data.fill_mode = 'FULL'
    curve_data.bevel_depth = 0.0001 # Give it some thickness for export
    curve_data.bevel_resolution = 0
    
    for i in range(num_strands):
        s = curve_data.splines.new('POLY') 
        s.points.add(pts_per_strand - 1)
        start = i * pts_per_strand
        end = start + pts_per_strand
        s.points.foreach_set('co', points_4d[start:end].ravel())
        
    obj = bpy.data.objects.new("DiffLocks_Hair", curve_data)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    # CONVERT TO GEOMETRY (Optional: New Curves system in Blender 3.3+)
    # We try to convert but if it fails or if we are in a version that handles ABC poorly, we stay legacy
    is_new_curves = False
    try:
        # Only convert to new CURVES if we're not explicitly asked for legacy compatibility in ABC
        # For now, let's keep it legacy for better ABC/USD support across platforms
        # bpy.ops.object.convert(target='CURVES', keep_original=False)
        # is_new_curves = True
        # print("✨ Converted to new CURVES system")
        print("ℹ️ Keeping as legacy CURVE for maximum compatibility with ABC/USD")
    except:
        print("⚠️ Conversion to CURVES failed, keeping as legacy CURVE")
    
    obj = bpy.context.active_object
    obj.select_set(True)
    
    # MATERIAL
    mat = create_material()
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)

    # Force dependency graph update
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    
    def verify_and_retry_abc(filepath):
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 5000: # Increased threshold
            print(f"⚠️ {filepath} is missing or too small ({os.path.getsize(filepath) if os.path.exists(filepath) else 0} bytes).")
            
            # TRY 1: Export without export_hair (treat as mesh/curve)
            print("🔄 Retry 1: Exporting without 'export_hair'...")
            try:
                bpy.ops.wm.alembic_export(
                    filepath=filepath, 
                    selected=True, 
                    start=1, end=1,
                    export_hair=False,
                    export_particles=False,
                    as_background_job=False
                )
            except: pass
            
            # TRY 2: Convert to MESH as absolute last resort
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 5000:
                print("🔄 Retry 2: Converting to MESH for export...")
                try:
                    bpy.ops.object.convert(target='MESH')
                    bpy.ops.wm.alembic_export(
                        filepath=filepath, 
                        selected=True, 
                        start=1, end=1,
                        as_background_job=False
                    )
                except Exception as e:
                    print(f"❌ Mesh conversion export failed: {e}")

    # EXPORT
    if 'blend' in requested_formats:
        out = f"{output_base}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
        print(f"✅ Exported: {out}")
    
    if 'abc' in requested_formats:
        out = f"{output_base}.abc"
        print(f"📦 Exporting Alembic: {out}")
        try:
            # For legacy CURVE objects with bevel, we want standard curve export.
            # 'export_hair' in Blender Alembic refers to Particle Systems.
            # 'curves' refers to Curve objects.
            bpy.ops.wm.alembic_export(
                filepath=out, 
                selected=True, 
                start=1, end=1,
                export_hair=False, # We use actual Curve objects, not Particles
                export_particles=False,
                as_background_job=False,
                evaluation_mode='VIEWPORT'
            )
        except:
            pass
            
        verify_and_retry_abc(out)
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print(f"✅ Exported ABC: {out} ({os.path.getsize(out)} bytes)")
        else:
            print(f"❌ Failed to export valid ABC: {out}")

    if 'usd' in requested_formats:
        out = f"{output_base}.usd"
        print(f"📦 Exporting USD: {out}")
        try:
            bpy.ops.wm.usd_export(
                filepath=out, 
                selected_objects_only=True,
                export_hair=True,
                evaluation_mode='VIEWPORT'
            )
        except:
            pass
            
        if not os.path.exists(out) or os.path.getsize(out) < 5000:
            print(f"⚠️ USD too small. Retrying as mesh...")
            try:
                bpy.ops.object.convert(target='MESH')
                bpy.ops.wm.usd_export(filepath=out, selected_objects_only=True)
            except: pass
            
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print(f"✅ Exported USD: {out} ({os.path.getsize(out)} bytes)")
        else:
            print(f"❌ Failed to export valid USD: {out}")
        
    if 'obj' in requested_formats:
        out = f"{output_base}.obj"
        # Fallback for OBJ if needed via Blender
        bpy.ops.wm.obj_export(filepath=out, export_selected_objects=True)
        print(f"✅ Exported: {out}")

    print("✅ SUCCESS")

except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
