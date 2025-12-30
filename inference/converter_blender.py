
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

def purge_orphans():
    """Liberar memoria de bloques de datos no utilizados en Blender y forzar GC de Python"""
    import gc
    # Eliminar bloques de datos huérfanos (mallas, curvas, materiales que no se usan)
    for i in range(2): # A veces requiere un par de pasadas para limpiar dependencias
        bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    gc.collect()

def create_hair_principled_material(name="DiffLocks_Mat", colors=None):
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Try to use Principled Hair BSDF (available in modern Blender)
    try:
        shader = nodes.new(type='ShaderNodeBsdfHairPrincipled')
        shader.location = (300, 300)
        # Default to a nice brown (Melanin: 0.65, Redness: 0.5)
        if "Melanin" in shader.inputs:
            shader.inputs["Melanin"].default_value = 0.65
        if "Redness" in shader.inputs:
            shader.inputs["Redness"].default_value = 0.5
    except:
        # Fallback to standard Principled BSDF
        shader = nodes.new(type='ShaderNodeBsdfPrincipled')
        shader.location = (300, 300)
        if "Base Color" in shader.inputs:
            shader.inputs["Base Color"].default_value = (0.05, 0.03, 0.01, 1.0)

    # Check for color attribute
    if colors is not None:
        try:
            attr_node = nodes.new(type='ShaderNodeAttribute')
            attr_node.attribute_name = "DiffLocks_Color"
            attr_node.location = (0, 400)
            
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
    out.location = (600, 300)
    links.new(shader.outputs[0], out.inputs[0])
    return mat

try:
    t_start = time.time()
    
    # LOAD
    data = np.load(input_npz)
    positions = data['positions']
    colors = data.get('colors', None)
    radii = data.get('radii', None)
    
    # Liberar el objeto data de npz
    del data
    import gc
    gc.collect()
    
    num_strands = int(positions.shape[0])
    pts_per_strand = int(positions.shape[1])
    
    print(f"🧬 Processing {num_strands:,} strands ({pts_per_strand} pts each)...")

    # TRANSFORM
    flat_pos = positions.reshape(-1, 3) * SCALE_FACTOR
    if ROTATE_X_90:
        flat_pos = flat_pos[:, [0, 2, 1]] # Swap Y and Z
        flat_pos[:, 1] *= -1 # Invert Y

    # Liberar memoria de positions ya que tenemos flat_pos
    del positions
    import gc
    gc.collect()

    # --- GEOMETRY CREATION (Addon-style: Legacy -> Convert) ---
    print("🔨 Building Geometry: ", end="", flush=True)
    
    # 1. Create legacy curve
    try:
        curve_data = bpy.data.curves.new("DiffLocks_Temp", type='CURVE')
    except:
        curve_data = bpy.data.curves.new("DiffLocks_Temp", 'CURVE')
    
    curve_data.dimensions = '3D'
    
    # 2. Add splines
    # Prepare 4D points for foreach_set
    points_4d = np.empty((num_strands * pts_per_strand, 4), dtype=np.float32)
    points_4d[:, :3] = flat_pos.astype(np.float32)
    points_4d[:, 3] = 1.0
    
    report_interval = max(1, num_strands // 10)
    for i in range(num_strands):
        s = curve_data.splines.new('POLY')
        s.points.add(pts_per_strand - 1)
        start = i * pts_per_strand
        end = start + pts_per_strand
        s.points.foreach_set('co', points_4d[start:end].ravel())
        
        if i > 0 and i % report_interval == 0:
            print(f"{int((i/num_strands)*100)}%...", end=" ", flush=True)
    
    print("100% Done.")
    
    # Liberar memoria de arrays de numpy que ya no necesitamos
    del points_4d
    del flat_pos
    import gc
    gc.collect()

    # 3. Create Object
    temp_obj = bpy.data.objects.new("DiffLocks_Temp", curve_data)
    bpy.context.collection.objects.link(temp_obj)
    
    # Deselect all and select temp_obj
    bpy.ops.object.select_all(action='DESELECT')
    temp_obj.select_set(True)
    bpy.context.view_layer.objects.active = temp_obj
    
    # --- ABC EXPORT (BEFORE CONVERSION) ---
    # Exportamos ABC usando la curva legacy con bevel para asegurar volumen y compatibilidad
    if 'abc' in requested_formats:
        out_abc = f"{output_base}.abc"
        try:
            print(f"📦 Exporting Alembic (.abc) from Legacy Curve...")
            # Temporalmente damos volumen para la exportación
            temp_obj.data.bevel_depth = 0.001 * SCALE_FACTOR
            temp_obj.data.bevel_resolution = 0
            temp_obj.data.fill_mode = 'FULL'
            
            bpy.ops.wm.alembic_export(
                filepath=out_abc,
                selected=True,
                flatten=True,
                as_background_job=False,
                export_hair=True,
                export_particles=True
            )
            
            if os.path.exists(out_abc) and os.path.getsize(out_abc) > 10000:
                print(f"✅ Exported ABC: {out_abc} ({os.path.getsize(out_abc)/1024/1024:.1f} MB)")
                created_any = True
            
            # Quitamos el bevel para que la conversión a Hair Curves sea limpia
            temp_obj.data.bevel_depth = 0
            
            # Limpieza post-ABC
            purge_orphans()
        except Exception as e:
            print(f"❌ ABC export error: {e}")

    final_obj = temp_obj
    
    print("✨ Converting to Modern Hair Curves for .blend and .usd...", end=" ", flush=True)
    try:
        # This is the magic part from the addon
        bpy.ops.object.convert(target='CURVES', keep_original=False)
        final_obj = bpy.context.active_object
        final_obj.name = "DiffLocks_Hair"
        print("Done.")
        
        # --- ATTRIBUTES (Radius & Colors) ---
        # Set Radius
        if radii is not None:
            r_flat = radii.reshape(-1) * SCALE_FACTOR
            if 'radius' in final_obj.data.attributes:
                final_obj.data.attributes['radius'].data.foreach_set('value', r_flat.astype(np.float32))
        else:
            # Default radius
            if 'radius' in final_obj.data.attributes:
                total_pts = len(final_obj.data.attributes['radius'].data)
                defaults = np.full(total_pts, 0.003 * SCALE_FACTOR, dtype=np.float32)
                final_obj.data.attributes['radius'].data.foreach_set('value', defaults)

        # Set Colors
        if colors is not None:
            try:
                attr_name = "DiffLocks_Color"
                if attr_name not in final_obj.data.attributes:
                    attr = final_obj.data.attributes.new(name=attr_name, type='FLOAT_COLOR', domain='POINT')
                else:
                    attr = final_obj.data.attributes[attr_name]
                
                # Prepare color data (handle per-strand or per-point)
                c_flat = colors.reshape(-1, 3)
                num_colors = len(c_flat)
                total_pts = len(attr.data)
                
                if num_colors == num_strands and num_colors != total_pts:
                    # Per-strand colors: repeat for each point
                    print(f"ℹ️ Broadcasting {num_colors} strand colors to {total_pts} points...")
                    c_full = np.repeat(c_flat, pts_per_strand, axis=0)
                elif num_colors == total_pts:
                    c_full = c_flat
                else:
                    print(f"⚠️ Color count mismatch: {num_colors} colors for {total_pts} points. Skipping.")
                    c_full = None
                
                if c_full is not None:
                    rgba = np.ones((len(c_full), 4), dtype=np.float32)
                    rgba[:, :3] = c_full
                    attr.data.foreach_set('color', rgba.ravel())
                    print("🎨 Color attributes applied.")
                    del rgba
                    if 'c_full' in locals(): del c_full
                    import gc
                    gc.collect()
            except Exception as e:
                print(f"⚠️ Could not apply colors: {e}")

    except Exception as e:
        print(f"\n⚠️ Modern conversion failed: {e}. Keeping as legacy curve.")
        final_obj.data.bevel_depth = 0.003 * SCALE_FACTOR
        final_obj.data.bevel_resolution = 0
        final_obj.data.fill_mode = 'FULL'

    # MATERIAL
    mat = create_hair_principled_material(colors=colors)
    if final_obj.data.materials:
        final_obj.data.materials[0] = mat
    else:
        final_obj.data.materials.append(mat)
    
    # Ahora sí podemos borrar los colores originales
    if 'colors' in locals(): del colors
    if 'radii' in locals(): del radii
    import gc
    gc.collect()

    # Force dependency graph update
    dg = bpy.context.evaluated_depsgraph_get()
    dg.update()
    
    # Final cleanup before exports
    purge_orphans()
    
    # EXPORT REMAINING FORMATS
    created_any = False
    
    # Ensure active and selected for exports
    bpy.ops.object.select_all(action='DESELECT')
    final_obj.select_set(True)
    bpy.context.view_layer.objects.active = final_obj
    
    if 'blend' in requested_formats:
        out = f"{output_base}.blend"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=out, compress=True)
            if os.path.exists(out) and os.path.getsize(out) > 1024:
                print(f"✅ Exported: {out} ({os.path.getsize(out)/1024/1024:.1f} MB)")
                created_any = True
            purge_orphans()
        except Exception as e:
            print(f"❌ Blender .blend export error: {e}")
    
    if 'abc' in requested_formats and not any(f.endswith('.abc') for f in [output_base]):
        # Si por alguna razón no se creó arriba, podríamos intentar aquí, 
        # pero la lógica principal ahora está antes de la conversión.
        pass
    
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
            purge_orphans()
        except Exception as e:
            print(f"❌ Blender USD export error: {e}")
            
    if not created_any:
        print("⚠️ No Blender files were created.")
        
    print(f"✅ SUCCESS (Total time: {time.time() - t_start:.2f}s)")


except Exception as e:
    traceback.print_exc()
    sys.exit(1)
