
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

    # --- HAIR CREATION (Optimized) ---
    print(f"🧬 Creating {num_strands} strands...")
    
    obj = None
    # Try 1: Modern CURVES object (Blender 3.3+)
    try:
        # Some Blender builds have issues with the 'CURVES' enum even if version >= 3.3
        print("🧪 Attempting modern CURVES object...")
        curve_data = bpy.data.curves.new(name="Hair", type='CURVES')
        curve_data.curves.add(num_strands)
        curve_data.points.add(num_strands * pts_per_strand)
        curve_data.points.foreach_set('position', flat_pos.ravel())
        
        offsets = np.arange(0, (num_strands + 1) * pts_per_strand, pts_per_strand, dtype=np.int32)
        curve_data.curve_offsets.foreach_set(offsets)
        
        obj = bpy.data.objects.new("DiffLocks_Hair", curve_data)
        use_new_curves = True
        print("🚀 Success: Using modern CURVES object")
    except Exception as e:
        print(f"ℹ️ Modern CURVES failed or not supported: {e}")
        use_new_curves = False

    # Try 2: Fast Mesh-to-Curve conversion (Fallback for all versions)
    if obj is None:
        print("🔄 Fallback: Creating via Mesh-to-Curve conversion (Fast)...")
        try:
            mesh_data = bpy.data.meshes.new("HairMesh")
            
            # Create edges for all strands
            # Each strand has pts_per_strand points: (0,1), (1,2), ... (n-1, n)
            edges = []
            for s in range(num_strands):
                offset = s * pts_per_strand
                for p in range(pts_per_strand - 1):
                    edges.append((offset + p, offset + p + 1))
            
            mesh_data.from_pydata(flat_pos, edges, [])
            obj = bpy.data.objects.new("DiffLocks_Hair", mesh_data)
            bpy.context.collection.objects.link(obj)
            bpy.context.view_layer.objects.active = obj
            obj.select_set(True)
            
            # Convert Mesh to Curve
            bpy.ops.object.convert(target='CURVE')
            obj = bpy.context.active_object
            
            # Set curve settings for the converted object
            obj.data.dimensions = '3D'
            obj.data.fill_mode = 'FULL'
            obj.data.bevel_depth = 0.0
            print("✅ Success: Created via Mesh conversion")
        except Exception as e:
            print(f"❌ Mesh conversion failed: {e}")
            
    # Try 3: Legacy Spline creation (Absolute last resort, very slow)
    if obj is None:
        print("⚠️ Final Fallback: Legacy CURVE object (Very slow for high strand counts)")
        curve_data = bpy.data.curves.new(name="Hair", type='CURVE')
        curve_data.dimensions = '3D'
        curve_data.fill_mode = 'FULL'
        curve_data.bevel_depth = 0.0
        
        for i in range(num_strands):
            s = curve_data.splines.new('POLY') 
            s.points.add(pts_per_strand - 1)
            start = i * pts_per_strand
            end = start + pts_per_strand
            s.points.foreach_set('co', points_4d[start:end].ravel())
            
        obj = bpy.data.objects.new("DiffLocks_Hair", curve_data)
        bpy.context.collection.objects.link(obj)

    # Finalize object
    if obj:
        if obj.name not in bpy.context.collection.objects:
            bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
    else:
        raise Exception("Failed to create hair object in Blender")

    # MATERIAL
    mat = create_material()
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)

    # Force dependency graph update
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    
    def verify_and_retry_abc(filepath, is_new_curves):
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 5000: 
            print(f"⚠️ {filepath} is missing or too small. Retrying with alternative settings...")
            
            # TRY 1: Toggle export_hair
            try:
                bpy.ops.wm.alembic_export(
                    filepath=filepath, 
                    selected=True, 
                    start=1, end=1,
                    export_hair=not is_new_curves, # Try the opposite of what we tried first
                    export_particles=False,
                    as_background_job=False
                )
            except: pass
            
            # TRY 2: Convert to MESH as absolute last resort
            if not os.path.exists(filepath) or os.path.getsize(filepath) < 5000:
                print("🔄 Retry 2: Converting to MESH for export...")
                try:
                    # For new CURVES, we might need to convert to legacy curve first then mesh, 
                    # or just mesh. bpy.ops.object.convert handles it.
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
            # For new CURVES system, export_hair=True is often required.
            # For legacy CURVE, export_hair=False (it's a curve object).
            bpy.ops.wm.alembic_export(
                filepath=out, 
                selected=True, 
                start=1, end=1,
                export_hair=use_new_curves, 
                export_particles=False,
                as_background_job=False,
                evaluation_mode='VIEWPORT'
            )
        except:
            pass
            
        verify_and_retry_abc(out, use_new_curves)
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            print(f"✅ Exported ABC: {out} ({os.path.getsize(out)} bytes)")
        else:
            print(f"❌ Failed to export valid ABC: {out}")

    if 'usd' in requested_formats:
        out = f"{output_base}.usd"
        print(f"📦 Exporting USD: {out}")
        try:
            # USD exporter usually handles new Curves and legacy Curves well with export_hair=True
            bpy.ops.wm.usd_export(
                filepath=out, 
                selected_objects_only=True,
                export_hair=True, 
                evaluation_mode='VIEWPORT'
            )
        except:
            pass
            
        if not os.path.exists(out) or os.path.getsize(out) < 5000:
            print(f"⚠️ USD too small. Retrying with mesh conversion...")
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
