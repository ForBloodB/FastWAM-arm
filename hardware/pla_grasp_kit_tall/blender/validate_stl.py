import json
import struct
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    '01_grip_block_36x30x70mm.stl': (36.0, 30.0, 70.0),
    '02_peg_d16_h80mm.stl': (26.0, 26.0, 80.0),
    '03_socket_50x50x50mm_hole16p4.stl': (50.0, 50.0, 50.0),
    '04_sorting_tray_80x65x40mm.stl': (80.0, 65.0, 40.0),
}
report = {'units': 'mm', 'method': 'independent binary STL parse', 'parts': []}
for filename, expected in EXPECTED.items():
    path = ROOT / 'stl' / filename
    data = path.read_bytes()
    count = struct.unpack_from('<I', data, 80)[0]
    if len(data) != 84 + 50 * count:
        raise RuntimeError(f'{filename}: invalid binary STL length')
    triangles = np.empty((count, 3, 3), dtype=np.float64)
    raw_vertices = []
    offset = 84
    for index in range(count):
        values = struct.unpack_from('<12fH', data, offset)
        triangle = np.asarray(values[3:12], dtype=np.float64).reshape(3, 3)
        triangles[index] = triangle
        raw_vertices.extend(struct.pack('<3f', *values[i:i + 3]) for i in (3, 6, 9))
        offset += 50
    dimensions = np.ptp(triangles.reshape(-1, 3), axis=0)
    if not np.allclose(dimensions, expected, atol=0.001, rtol=0):
        raise RuntimeError(f'{filename}: dimensions {dimensions} != {expected}')
    edges = Counter()
    for start in range(0, len(raw_vertices), 3):
        a, b, c = raw_vertices[start:start + 3]
        for u, v in ((a, b), (b, c), (c, a)):
            edges[tuple(sorted((u, v)))] += 1
    nonmanifold = sum(value != 2 for value in edges.values())
    cross = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    degenerate = int(np.count_nonzero(np.linalg.norm(cross, axis=1) <= 1e-10))
    signed_volume = float(np.einsum('ij,ij->i', triangles[:, 0], cross[:, :]).sum() / 6.0)
    if nonmanifold or degenerate or signed_volume <= 0:
        raise RuntimeError(f'{filename}: nonmanifold={nonmanifold}, degenerate={degenerate}, volume={signed_volume}')
    report['parts'].append({
        'file': 'stl/' + filename,
        'triangles': count,
        'dimensions_mm': dimensions.tolist(),
        'min_z_mm': float(triangles[:, :, 2].min()),
        'max_z_mm': float(triangles[:, :, 2].max()),
        'nonmanifold_edges': nonmanifold,
        'degenerate_triangles': degenerate,
        'signed_volume_mm3': round(signed_volume, 3),
    })
report['success'] = True
output = ROOT / 'docs' / 'review' / 'stl_validation.json'
output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps(report, indent=2, ensure_ascii=False))
