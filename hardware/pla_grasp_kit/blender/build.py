import bpy
import bmesh
import json
import math
from pathlib import Path
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[1]
STL = ROOT / 'stl'
DOCS = ROOT / 'docs'
REVIEW = DOCS / 'review'
for folder in (STL, DOCS, REVIEW):
    folder.mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
scene.render.engine = 'CYCLES'
scene.cycles.samples = 48
scene.render.resolution_x = 1200
scene.render.resolution_y = 850
scene.render.resolution_percentage = 100
scene.world.color = (0.12, 0.12, 0.12)


def mm(v):
    return v / 1000.0


def material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = rgba
    bsdf.inputs['Roughness'].default_value = 0.48
    return mat


blue = material('抓取块 · 蓝色 PLA', (0.035, 0.32, 0.72, 1))
orange = material('圆柱销 · 橙色 PLA', (0.95, 0.27, 0.045, 1))
gray = material('插孔底座 · 浅灰 PLA', (0.56, 0.62, 0.68, 1))
green = material('分拣托盘 · 绿色 PLA', (0.06, 0.48, 0.30, 1))


def activate(obj):
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def apply_modifier(obj, modifier):
    activate(obj)
    bpy.ops.object.modifier_apply(modifier=modifier.name)


def bevel(obj, width, segments=3):
    mod = obj.modifiers.new('打印安全倒角', 'BEVEL')
    mod.width = width
    mod.segments = segments
    apply_modifier(obj, mod)
    mod = obj.modifiers.new('平面法线', 'WEIGHTED_NORMAL')
    mod.keep_sharp = True
    apply_modifier(obj, mod)


def cube(name, dims_mm, mat, bevel_mm=0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, mm(dims_mm[2]) / 2))
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = tuple(mm(v) for v in dims_mm)
    activate(obj)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel_mm:
        bevel(obj, mm(bevel_mm), 4)
    obj.data.materials.append(mat)
    return obj


def lathe(name, profile_mm, mat=None, segments=96):
    verts, rings, faces = [], [], []
    for radius, z in profile_mm:
        if radius == 0:
            rings.append([len(verts)])
            verts.append((0, 0, mm(z)))
        else:
            ring = []
            for i in range(segments):
                a = 2 * math.pi * i / segments
                ring.append(len(verts))
                verts.append((mm(radius) * math.cos(a), mm(radius) * math.sin(a), mm(z)))
            rings.append(ring)
    for first, second in zip(rings[:-1], rings[1:]):
        if len(first) == 1 and len(second) > 1:
            for i in range(segments):
                faces.append((first[0], second[(i + 1) % segments], second[i]))
        elif len(second) == 1 and len(first) > 1:
            for i in range(segments):
                faces.append((first[i], first[(i + 1) % segments], second[0]))
        elif len(first) > 1 and len(second) > 1:
            for i in range(segments):
                j = (i + 1) % segments
                faces.append((first[i], first[j], second[j], second[i]))
    mesh = bpy.data.meshes.new(name + ' · 封闭网格')
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    scene.collection.objects.link(obj)
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bm.to_mesh(mesh)
    bm.free()
    if mat:
        mesh.materials.append(mat)
    for poly in mesh.polygons:
        poly.use_smooth = True
    return obj


def boolean_difference(obj, cutter):
    mod = obj.modifiers.new('减材特征', 'BOOLEAN')
    mod.operation = 'DIFFERENCE'
    mod.solver = 'EXACT'
    mod.object = cutter
    apply_modifier(obj, mod)
    bpy.data.objects.remove(cutter, do_unlink=True)


def clean_mesh(obj):
    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.remove_doubles(bm, verts=list(bm.verts), dist=1e-8)
    bmesh.ops.recalc_face_normals(bm, faces=list(bm.faces))
    bad = sum(not edge.is_manifold for edge in bm.edges)
    volume = abs(bm.calc_volume(signed=True))
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return bad, volume


def export_stl(obj, filename):
    activate(obj)
    bpy.ops.wm.stl_export(
        filepath=str(STL / filename),
        export_selected_objects=True,
        global_scale=1000.0,
        apply_modifiers=True,
        ascii_format=False,
    )


# 1. Opposing curved finger grooves locate the two gripper pads.
block = cube('01_带指槽抓取块', (36, 30, 20), blue, 1.6)
for x in (-18, 18):
    bpy.ops.mesh.primitive_cylinder_add(vertices=64, radius=mm(4), depth=mm(38),
                                      location=(mm(x), 0, mm(11)),
                                      rotation=(math.pi / 2, 0, 0))
    cutter = bpy.context.object
    boolean_difference(block, cutter)
bevel(block, mm(0.35), 3)

# 2. Single manifold lathed pin: shaft, grasping head and insertion chamfers.
pin = lathe('02_带头导向圆柱销', [
    (0, 0), (6.5, 0), (8, 1.5), (8, 28.5), (8, 30),
    (13, 30), (13, 40.5), (11.5, 42), (0, 42),
], orange)

# 3. Blind socket: 16.4 mm diameter, 14 mm depth, 4 mm bottom.
socket = cube('03_盲孔插装底座', (50, 50, 18), gray, 2.0)
cutter = lathe('临时孔刀', [
    (0, 4), (8.2, 4), (8.2, 16.5), (9.2, 18), (9.2, 21), (0, 21),
], segments=96)
boolean_difference(socket, cutter)
bevel(socket, mm(0.3), 3)

# 4. Open tray with 3 mm floor and walls, no support required.
tray = cube('04_低边分拣托盘', (80, 65, 12), green, 2.0)
bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, mm(10.5)))
cutter = bpy.context.object
cutter.dimensions = (mm(74), mm(59), mm(15))
activate(cutter)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
bevel(cutter, mm(2), 4)
boolean_difference(tray, cutter)
bevel(tray, mm(0.5), 3)

parts = [
    (block, '01_grip_block_36x30x20mm.stl', (36, 30, 20), [-90, 15, 0]),
    (pin, '02_peg_d16_h42mm.stl', (26, 26, 42), [-28, 15, 0]),
    (socket, '03_socket_50x50x18mm_hole16p4.stl', (50, 50, 18), [35, 15, 0]),
    (tray, '04_sorting_tray_80x65x12mm.stl', (80, 65, 12), [-15, -72, 0]),
]
report = {'units': 'mm', 'blender': bpy.app.version_string, 'parts': []}
for obj, filename, expected, location_mm in parts:
    bad, volume = clean_mesh(obj)
    dims = [round(v * 1000, 5) for v in obj.dimensions]
    if bad:
        raise RuntimeError(f'{obj.name}: {bad} non-manifold edges')
    if any(abs(a - b) > 0.03 for a, b in zip(dims, expected)):
        raise RuntimeError(f'{obj.name}: dimensions {dims} differ from {expected}')
    export_stl(obj, filename)
    report['parts'].append({
        'name': obj.name, 'file': 'stl/' + filename, 'dimensions_mm': dims,
        'nonmanifold_edges': bad, 'volume_mm3': round(volume * 1e9, 3),
        'triangles': sum(len(p.vertices) - 2 for p in obj.data.polygons),
    })
    obj.location = tuple(mm(v) for v in location_mm)

# Keep a clean, editable source and a GLB for viewers that support it.
scene['打印单位'] = 'STL coordinates are millimeters; scene geometry is meters'
scene['销孔直径间隙_mm'] = 0.4
scene['模型用途'] = 'B601-RS 低速抓取、搬运与销孔插装演示'
bpy.ops.wm.save_as_mainfile(filepath=str(ROOT / 'pla_grasp_kit.blend'))
bpy.ops.object.select_all(action='SELECT')
bpy.ops.export_scene.gltf(filepath=str(ROOT / 'pla_grasp_kit.glb'),
                          export_format='GLB', use_selection=True,
                          export_apply=True, export_yup=True)

# Studio preview; ground, camera and lights are not part of the printable GLB.
scene.view_settings.view_transform = 'AgX'
scene.view_settings.look = 'AgX - Medium High Contrast'
scene.view_settings.exposure = -0.35
ground_mat = material('预览台 · 中性灰', (0.12, 0.15, 0.19, 1))
bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -mm(0.15)))
ground = bpy.context.object
ground.name = '仅供预览的台面'
ground.scale = (0.002, 0.002, 0.002)
ground.data.materials.append(ground_mat)
bpy.ops.object.light_add(type='AREA', location=(-0.15, -0.10, 0.28))
key = bpy.context.object
key.name = '主光 · 大面积柔光'
key.data.energy = 4
key.data.shape = 'DISK'
key.data.size = 0.22
key.rotation_euler = (Vector((0, -0.025, 0.01)) - key.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.light_add(type='AREA', location=(0.18, 0.08, 0.18))
fill = bpy.context.object
fill.name = '补光 · 轮廓'
fill.data.energy = 2
fill.data.size = 0.16
fill.rotation_euler = (Vector((0, -0.025, 0.01)) - fill.location).to_track_quat('-Z', 'Y').to_euler()
bpy.ops.object.camera_add(location=(0.19, -0.27, 0.22))
camera = bpy.context.object
camera.name = '打印件总览相机'
target = Vector((-0.01, -0.025, 0.012))
camera.rotation_euler = (target - camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.type = 'ORTHO'
camera.data.ortho_scale = 0.265
scene.camera = camera
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(REVIEW / 'overview.png')
bpy.ops.render.render(write_still=True)
(REVIEW / 'mesh_validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
