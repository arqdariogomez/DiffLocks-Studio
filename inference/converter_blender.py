
import bpy
import sys
import numpy as np
import os
import time
import traceback

# --- ARGS PARSING ---
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
ROTATE_X_90 = True 

def purge_orphans():
    import gc
    for i in range(2):
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    gc.collect()

def create_hair_principled_material(name="DiffLocks_Mat", colors=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    try:
        shader = nodes.new(type='ShaderNodeBsdfHairPrincipled')
        shader.location = (300, 300)
        if "Melanin" in shader.inputs:
            shader.inputs["Melanin"].default_value = 0.65
        if "Redness" in shader.inputs:
            shader.inputs["Redness"].default_value = 0.5
    except:
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader.location = (300, 300)
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = (0.05, 0.03, 0.01, 1.0)

    if colors is not None:
        try:
            attr_node = nodes.new(type='ShaderNodeAttribute')
            attr_node.attribute_name = "DiffLocks_Color"
            attr_node.location = (0, 400)
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
    out.location = (600, 300)
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    t_start = time.time()
    created_any = False
    
    # LOAD
    data = np.load(input_npz)
    positions = data['positions']
    colors = data.get('colors', None)
    radii = data.get('radii', None)
    del data
    import gc
    gc.collect()
    
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    print(f"🧬 Processing {num_strands:,} strands ({pts_per_strand} pts each)...")

    # TRANSFORM
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        flat_pos = flat_pos[:, [0, 2, 1]] 
        flat_pos[:, 1] *= -1 

    del positions
    gc.collect()

    # --- GEOMETRY CREATION ---
    print("🔨 Building Geometry: ", end="", flush=True)
    
    chunk_size = 10000
    legacy_objs = []
    
    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos.astype(np.float32)
    points_4d[:, 3] = 1.0
    
    for i in range(0, num_strands, chunk_size):
        end_i = min(i + chunk_size, num_strands)
        current_num = end_i - i
        
        try:
            chunk_curve = bpy.data.curves.new(f"Hair_{i}", type='CURVE')
        except:
            chunk_curve = bpy.data.curves.new(f"Hair_{i}", 'CURVE')
        chunk_curve.dimensions = '3D'
        
        for j in range(current_num):
            strand_idx = i + j
            s = chunk_curve.splines.new('POLY')
            s.points.add(pts_per_strand - 1)
            
            start_p = strand_idx * pts_per_strand
            end_p = start_p + pts_per_strand
            chunk_pts = points_4d[start_p:end_p].ravel()
            s.points.foreach_set('co', chunk_pts)
        
        chunk_obj = bpy.data.objects.new(f"Chunk_{i}", chunk_curve)
        bpy.context.collection.objects.link(chunk_obj)
        legacy_objs.append(chunk_obj)
        
        print(f"{int((end_i/num_strands)*100)}%...", end=" ", flush=True)
        # purge_orphans() removed from loop to avoid hang
    
    print("100% Done.")
    del points_4d
    del flat_pos
    gc.collect()

    # --- ABC EXPORT (From Legacy Curves) ---
    if 'abc' in requested_formats:
        out_abc = f"{output_base}.abc"
        print(f"📦 Exporting Alembic (.abc)...")
        try:
            bpy.ops.object.select_all(action='DESELECT')
            for obj in legacy_objs:
                obj.select_set(True)
                # Add bevel for volume
                obj.data.bevel_depth = 0.0005 * SCALE_FACTOR
                obj.data.bevel_resolution = 0
                obj.data.fill_mode = 'FULL'
            
            # Use temp_override for export to be safe
            ctx = {
                "selected_objects": legacy_objs,
                "selected_editable_objects": legacy_objs
            }
            with bpy.context.temp_override(**ctx):
                bpy.ops.wm.alembic_export(
                    filepath=out_abc,
                    selected=True,
                    flatten=True,
                    as_background_job=False,
                    export_hair=True,
                    export_particles=True
                )
            
            if os.path.exists(out_abc) and os.path.getsize(out_abc) > 1000:
                print(f"✅ Exported ABC: {out_abc} ({os.path.getsize(out_abc)/1024/1024:.1f} MB)")
                created_any = True
            else:
                print(f"⚠️ ABC export produced no file or empty file: {out_abc}")
        except Exception as e:
            print(f"❌ ABC export error: {e}")

    # --- CONVERT TO MODERN CURVES & JOIN ---
    print("🔄 Converting to Modern Curves...")
    modern_objs = []
    for obj in legacy_objs:
        if not obj or obj.name not in bpy.data.objects:
            continue
            
        bpy.ops.object.select_all(action='DESELECT')
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
        
        try:
            # Use temp_override for conversion too
            ctx = {
                "active_object": obj,
                "selected_objects": [obj],
                "selected_editable_objects": [obj]
            }
            with bpy.context.temp_override(**ctx):
                if bpy.ops.object.convert.poll():
                    bpy.ops.object.convert(target='CURVES')
                    new_obj = bpy.context.active_object
                    if new_obj and new_obj.type == 'CURVES':
                        modern_objs.append(new_obj)
                    else:
                        modern_objs.append(obj)
                else:
                    print(f"⚠️ Convert poll failed for {obj.name}, using as is")
                    modern_objs.append(obj)
        except Exception as e:
            print(f"⚠️ Conversion failed for {obj.name}: {e}")
            modern_objs.append(obj)
    
    # JOIN (Only if they are the same type)
    if len(modern_objs) > 0:
        # Separate by type to avoid join errors
        objs_by_type = {}
        for obj in modern_objs:
            t = obj.type
            if t not in objs_by_type: objs_by_type[t] = []
            objs_by_type[t].append(obj)
            
        final_main_obj = None
        
        for t, o_list in objs_by_type.items():
            if len(o_list) > 1:
                bpy.ops.object.select_all(action='DESELECT')
                for o in o_list:
                    o.select_set(True)
                
                # Robust active object setting
                active_obj = o_list[0]
                bpy.context.view_layer.objects.active = active_obj
                
                print(f"🔗 Joining {len(o_list)} objects of type {t}...")
                try:
                    # Use temp_override for Blender 4.x+ headless reliability
                    ctx = {
                        "active_object": active_obj,
                        "selected_objects": o_list,
                        "selected_editable_objects": o_list
                    }
                    with bpy.context.temp_override(**ctx):
                        if bpy.ops.object.join.poll():
                            bpy.ops.object.join()
                            final_main_obj = bpy.context.active_object
                            print(f"✅ Join successful for type {t}")
                        else:
                            print(f"⚠️ Join poll failed for type {t}")
                            final_main_obj = o_list[0]
                except Exception as e:
                    print(f"⚠️ Join failed for type {t}: {e}")
                    final_main_obj = o_list[0]
            else:
                final_main_obj = o_list[0]
        
        final_obj = final_main_obj
        final_obj.name = "DiffLocks_Hair"
    else:
        raise Exception("No geometry objects were created/converted successfully")
    
    # --- ATTRIBUTES ---
    print("🎨 Applying Attributes...")
    try:
        # Radius
        if radii is not None:
            if 'radius' not in final_obj.data.attributes:
                final_obj.data.attributes.new(name="radius", type='FLOAT', domain='POINT')
            rad_attr = final_obj.data.attributes['radius']
            if len(radii.shape) == 1:
                # Per strand radius -> expand to per point
                full_radii = np.repeat(radii, pts_per_strand).astype(np.float32)
                rad_attr.data.foreach_set('value', full_radii)
            else:
                rad_attr.data.foreach_set('value', radii.astype(np.float32).ravel())
        
        # Colors
        if colors is not None:
            if 'DiffLocks_Color' not in final_obj.data.attributes:
                final_obj.data.attributes.new(name="DiffLocks_Color", type='FLOAT_COLOR', domain='POINT')
            col_attr = final_obj.data.attributes['DiffLocks_Color']
            if len(colors.shape) == 2: # (N, 3)
                # Per strand color -> expand to per point
                full_colors = np.repeat(colors, pts_per_strand, axis=0)
                # Add alpha
                rgba = np.ones((full_colors.shape[0], 4), dtype=np.float32)
                rgba[:, :3] = full_colors.astype(np.float32)
                col_attr.data.foreach_set('color', rgba.ravel())
            else:
                # Per point color (N, P, 3)
                flat_colors = colors.reshape(-1, 3).astype(np.float32)
                rgba = np.ones((flat_colors.shape[0], 4), dtype=np.float32)
                rgba[:, :3] = flat_colors
                col_attr.data.foreach_set('color', rgba.ravel())
    except Exception as e:
        print(f"⚠️ Could not apply attributes: {e}")

    # --- MATERIAL ---
    mat = create_hair_principled_material("DiffLocks_Mat", colors)
    if final_obj.data.materials:
        final_obj.data.materials[0] = mat
    else:
        final_obj.data.materials.append(mat)

    # --- FINAL EXPORTS ---
    if 'blend' in requested_formats:
        out_blend = f"{output_base}.blend"
        bpy.ops.wm.save_as_mainfile(filepath=out_blend)
        print(f"✅ Exported Blend: {out_blend}")
        created_any = True
        
    if 'usd' in requested_formats:
        out_usd = f"{output_base}.usd"
        try:
            ctx = {
                "selected_objects": [final_obj],
                "selected_editable_objects": [final_obj]
            }
            with bpy.context.temp_override(**ctx):
                bpy.ops.wm.usd_export(filepath=out_usd, selected_objects_only=True)
            
            if os.path.exists(out_usd):
                print(f"✅ Exported USD: {out_usd}")
                created_any = True
            else:
                print(f"⚠️ USD export produced no file: {out_usd}")
        except Exception as e:
            print(f"❌ USD export error: {e}")

    if created_any:
        print("✅ SUCCESS: Blender export process finished")
    else:
        print("⚠️ No Blender files were created")

except Exception as e:
    print(f"❌ ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup memory
    try:
        del colors
        del radii
    except: pass
    purge_orphans()
