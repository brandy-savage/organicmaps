"""Measure polygon coverage using the generator's even/odd filling rule.

Callers provide projected coordinates. These measurement geometries are never
used to replace source coordinate records.
"""

import numpy as np
from shapely.geometry import GeometryCollection, LineString, Polygon
from shapely.ops import polygonize, unary_union


def geometry_parts(geometry):
    """Yield atomic geometries without mixing point, line and area dimensions."""
    if geometry.is_empty:
        return
    if hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from geometry_parts(part)
    else:
        yield geometry


def measurement_difference(geometry, other):
    """Subtract coverage without GEOS's mixed-dimension collection difference bug.

    Point/line collections can retain a point already covered by their lines in
    GEOS 3.14. Compare homogeneous components instead; coordinates are unchanged.
    """
    if geometry.geom_type != "GeometryCollection" and other.geom_type != "GeometryCollection":
        return geometry.difference(other)
    by_dimension = {}
    for part in geometry_parts(other):
        by_dimension.setdefault(part.geom_type, []).append(part)
    occupants = [unary_union(parts) for parts in by_dimension.values()]
    remaining = []
    for part in geometry_parts(geometry):
        for occupant in occupants:
            if part.is_empty:
                break
            if part.geom_type in ("Point", "LineString") and occupant.covers(part):
                part = GeometryCollection()
            else:
                part = part.difference(occupant)
        if not part.is_empty:
            remaining.append(part)
    return unary_union(remaining)


def even_odd_contains(points, point):
    """Classify a face interior using the original traversal, including repeats."""
    current = points - (point.x, point.y)
    previous = np.roll(current, 1, axis=0)
    right = (current[:, 1] > 0) != (previous[:, 1] > 0)
    cross = current[:, 0] * previous[:, 1] - current[:, 1] * previous[:, 0]
    previous_above = previous[:, 1] > current[:, 1]
    # Representative points lie inside faces. Boundary membership tolerances can
    # misclassify a large face whose representative point lies in a narrow neck.
    return bool(np.count_nonzero(right & ((cross > 0) == previous_above)) % 2)


def ring_coverage(points, valid):
    if valid:
        return Polygon(points)
    if not np.array_equal(points[0], points[-1]):
        points = np.concatenate((points, points[:1]))
    # Noding deduplicates segments; classify faces against the original ring so
    # repeated traversals still cancel under the generator's even/odd rule.
    faces = polygonize(unary_union(LineString(points)))
    return unary_union([face for face in faces if even_odd_contains(points, face.representative_point())])


def measured_geometry(border, points, geometry):
    if geometry.is_valid:
        return geometry
    rings = [ring_coverage(p, ring.valid) for ring, p in zip(border.rings, points)]
    outers = []
    for index, ring in enumerate(border.rings):
        if ring.hole:
            continue
        holes = [rings[hole] for hole, owner in border.hole_owners.items() if owner == index]
        outers.append(rings[index].difference(unary_union(holes)) if holes else rings[index])
    return unary_union(outers)
