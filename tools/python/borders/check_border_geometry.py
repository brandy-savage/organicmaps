#!/usr/bin/env python3
"""Fail on source or decoded packed geometry regressions against an explicit baseline.

A region split or rename is allowed when its rings are valid and global coverage
and physical ambiguity are preserved. Areas use the audit tool's approximate
latitude correction. This command never generates or writes border data.
"""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
from shapely import STRtree, prepare
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon, box
from shapely.ops import unary_union
from shapely.validation import explain_validity

from audit_borders import area_m2, geographic_bounds, geographic_point, load, uncovered_loss
from border_geometry import geometry_parts, measurement_difference, ring_coverage
from packed_borders import PackedBorders, mercator
from maintain_borders import project


LOG = logging.getLogger(__name__)
# Two outer-path scale-10 tolerances, matching GetEpsilonForSimplify(10).
# This is an inspection limit for inherited contact movement, not an area tolerance.
PACKED_CONTACT_SHIFT = 2 * 360 * 1.3 / (256 << 10)


@dataclass
class Dataset:
    names: list
    shapes: list
    invalid_rings: list
    invalid_files: list
    metadata: dict
    contacts: dict
    file_contacts: dict = field(default_factory=dict)
    contact_kinds: dict = field(default_factory=dict)


def ring_contacts(points):
    """Exact nonadjacent edge contacts, including inherited invalid ring defects."""
    return unary_union(list(ring_contact_kinds(points).values()))


def ring_contact_kinds(points):
    """Separate proper crossings, point touches and retraced segments."""
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    edges = [LineString((point, points[(i + 1) % len(points)])) for i, point in enumerate(points)]
    tree = STRtree(edges)
    contacts = {"proper_crossings": [], "point_touches": [], "backtracks": []}
    for i, edge in enumerate(edges):
        if not edge.length:
            continue
        for index in tree.query(edge, predicate="intersects"):
            j = int(index)
            if j <= i or not edges[j].length:
                continue
            intersection = edge.intersection(edges[j])
            adjacent = j == i + 1 or (i == 0 and j == len(edges) - 1)
            for part in geometry_parts(intersection):
                if adjacent and not part.length:
                    continue
                kind = "backtracks" if part.geom_type == "LineString" else (
                    "proper_crossings" if edge.crosses(edges[j]) else "point_touches")
                contacts[kind].append(part)
    kinds = {kind: unary_union(parts) for kind, parts in contacts.items()}
    kinds["point_touches"] = measurement_difference(
        kinds["point_touches"], unary_union([kinds["backtracks"], kinds["proper_crossings"]]))
    return kinds


def contact_count(geometry):
    """Count connected contacts, independent of retraced segment subdivision."""
    parts = list(geometry_parts(geometry))
    parent = list(range(len(parts)))

    def owner(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    tree = STRtree(parts)
    for i, part in enumerate(parts):
        for value in tree.query(part, predicate="intersects"):
            j = int(value)
            if j > i:
                parent[owner(j)] = owner(i)
    return len({owner(index) for index in range(len(parts))})


def point_contacts_fit(current, baseline):
    """Match every current point to a distinct nearby inherited contact."""
    old = list(geometry_parts(baseline))
    choices = [[index for index, point in enumerate(old) if point.distance(current_point) <= PACKED_CONTACT_SHIFT]
               for current_point in geometry_parts(current)]
    matches = {}

    def match(index, visited):
        for owner in choices[index]:
            if owner in visited:
                continue
            visited.add(owner)
            if owner not in matches or match(matches[owner], visited):
                matches[owner] = index
                return True
        return False

    return all(match(index, set()) for index in range(len(choices)))


def new_ring_contacts(candidate, baseline, key):
    """Packed point contacts may shift locally; source and backtrack geometry may not."""
    kinds = candidate.contact_kinds.get(key)
    old_kinds = baseline.contact_kinds.get(key)
    old = baseline.contacts.get(key, GeometryCollection())
    if kinds is None or old_kinds is None:
        return measurement_difference(candidate.contacts[key], old), {}
    increased, added = {}, []
    for kind, contacts in kinds.items():
        if kind == "point_touches":
            # Repairing a retraced segment can leave its endpoints as touches.
            contacts = measurement_difference(contacts, old_kinds["backtracks"])
        before, after = contact_count(old_kinds[kind]), contact_count(contacts)
        if kind == "backtracks":
            extra = measurement_difference(contacts, old_kinds[kind])
        else:
            extra = contacts if after > before or not point_contacts_fit(contacts, old_kinds[kind]) else GeometryCollection()
        if not extra.is_empty:
            increased[kind] = {"baseline": before, "current": after}
            added.append(extra)
    return unary_union(added), increased


def cross_ring_contacts(rings):
    """Boundary contacts and unintended interior overlap between ring pairs."""
    if len(rings) < 2:
        return {}
    lines, coverage = [], []
    for _, points, _, valid in rings:
        closed = points if np.array_equal(points[0], points[-1]) else np.vstack((points, points[0]))
        lines.append(LineString(closed))
        coverage.append(ring_coverage(points, valid))
    tree = STRtree(lines)
    contacts = {}
    for i, line in enumerate(lines):
        for value in tree.query(line):
            j = int(value)
            if j <= i:
                continue
            contact = line.intersection(lines[j])
            # An outer intentionally contains its hole's interior.
            if rings[i][2] == rings[j][2]:
                contact = contact.union(coverage[i].intersection(coverage[j]))
            if not contact.is_empty:
                contacts[rings[i][0], rings[j][0]] = contact
    return contacts


def source_dataset(directory):
    borders, shapes, metadata = load(directory)
    contacts, file_contacts = {}, {}
    for border in borders:
        for ring in border.rings:
            if not ring.valid:
                key = border.name, ring.name
                contacts[key] = ring_contacts(project(ring.points))
        if not border.valid:
            for pair, contact in cross_ring_contacts([
                    (ring.name, project(ring.points), ring.hole, ring.valid) for ring in border.rings]).items():
                file_contacts[border.name, *pair] = contact
    metadata["sha256"] = {border.name: hashlib.sha256(border.source.read_bytes()).hexdigest() for border in borders}
    metadata["file_identity"] = {border.name: {"title": border.title, "rings": [ring.name for ring in border.rings]}
                                 for border in borders}
    return Dataset([border.name for border in borders], shapes, metadata["invalid_rings"],
                   metadata["invalid_files"], metadata, contacts, file_contacts)


def rectangle_mask(bounds):
    left, bottom, right, top = bounds
    # CountryDef::IsPointInsideRect accepts all three canonical longitude shifts.
    return unary_union([box(left + shift, bottom, right + shift, top) for shift in (-360, 0, 360)])


def packed_datasets(path, bounds_format="wrapped"):
    packed = PackedBorders(path, bounds_format)
    names, shapes, invalid_rings, invalid_files = [], [], [], []
    contacts, file_contacts, contact_kinds, clipped = {}, {}, {}, {}
    path_points = 0
    for index, name in enumerate(packed.names):
        filename = name + ".poly"
        names.append(filename)
        parts, polygons, rings = [], [], []
        for ring_index, integers in enumerate(packed.paths(index), 1):
            points = mercator(integers)
            path_points += len(points)
            ring_name = str(ring_index)
            polygon = Polygon(points) if len(points) >= 3 else Polygon()
            valid = polygon.is_valid and not polygon.is_empty
            if len(points) >= 3:
                rings.append((ring_name, points, False, valid))
            polygons.append(polygon)
            if not valid:
                invalid_rings.append({"file": filename, "ring": ring_name, "reason": explain_validity(polygon)})
                key = filename, ring_name
                if len(points) >= 3:
                    contact_kinds[key] = ring_contact_kinds(points)
                    contacts[key] = unary_union(list(contact_kinds[key].values()))
                else:
                    contacts[key] = GeometryCollection()
            parts.append(ring_coverage(points, valid) if len(points) >= 3 else Polygon())
        assembled = MultiPolygon(polygons) if len(polygons) > 1 else polygons[0]
        if not assembled.is_valid or assembled.is_empty:
            invalid_files.append(filename)
            for pair, contact in cross_ring_contacts(rings).items():
                file_contacts[filename, *pair] = contact
        measured = unary_union(parts)
        mask = rectangle_mask(packed.bounds[name])
        clipped[filename] = measured.difference(mask)
        shapes.append(measured.intersection(mask))
        if index % 100 == 0:
            LOG.info("Decoded outer paths for %s/%s regions", index, len(packed.names))
    metadata = {"sha256": hashlib.sha256(packed.data).hexdigest(), "size_bytes": len(packed.data),
                "bounds_format": bounds_format,
                "files": len(names), "path_points": path_points,
                "bounds": packed.bounds}
    outer = Dataset(names, shapes, invalid_rings, invalid_files, metadata, contacts, file_contacts, contact_kinds)
    return outer, clipped


def finding(geometry, **details):
    return {**details, "approx_m2": area_m2(geometry), "bounds": geographic_bounds(geometry),
            "point": geographic_point(geometry)}


def coverage_levels(shapes, tree, window):
    """Return local geometry covered by at least 1, 2, ... distinct region owners."""
    levels = []
    for value in tree.query(window, predicate="intersects"):
        piece = shapes[int(value)].intersection(window)
        if not piece.area:
            continue
        for index in range(len(levels) - 1, -1, -1):
            added = levels[index].intersection(piece)
            if not added.area:
                continue
            if index + 1 == len(levels):
                levels.append(added)
            else:
                levels[index + 1] = levels[index + 1].union(added)
        if levels:
            levels[0] = levels[0].union(piece)
        else:
            levels.append(piece)
    return levels


def introduced_ambiguity(window, candidate, current_tree, baseline, old_tree):
    current = coverage_levels(candidate.shapes, current_tree, window)
    old = coverage_levels(baseline.shapes, old_tree, window)
    # A third owner is a regression even where the baseline already had two.
    return unary_union([measurement_difference(level, old[index] if index < len(old) else Polygon())
                        for index, level in enumerate(current) if index >= 1])


def compare(candidate, baseline, minimum_area):
    old_invalid = {(item["file"], item["ring"]) for item in baseline.invalid_rings}
    new_invalid = [item for item in candidate.invalid_rings if (item["file"], item["ring"]) not in old_invalid]
    new_contacts = []
    for key, contacts in candidate.contacts.items():
        added, increased = new_ring_contacts(candidate, baseline, key)
        if not added.is_empty:
            new_contacts.append({"file": key[0], "ring": key[1], "bounds": geographic_bounds(added),
                                 "point": geographic_point(added), "increased_kinds": increased})
    new_file_contacts = []
    for key, contacts in candidate.file_contacts.items():
        added = measurement_difference(contacts, baseline.file_contacts.get(key, GeometryCollection()))
        if not added.is_empty:
            new_file_contacts.append({"file": key[0], "rings": list(key[1:]),
                                      "bounds": geographic_bounds(added), "point": geographic_point(added)})
    old_tree = STRtree(baseline.shapes)
    new_tree = STRtree(candidate.shapes)
    prepare(candidate.shapes)
    overlaps, losses = [], []
    old_by_name = dict(zip(baseline.names, baseline.shapes))
    unchanged = {name for name, shape in zip(candidate.names, candidate.shapes)
                 if name in old_by_name and shape.equals_exact(old_by_name[name], 0)}
    for i, shape in enumerate(candidate.shapes):
        for value in new_tree.query(shape, predicate="intersects"):
            j = int(value)
            if j <= i or (candidate.names[i] in unchanged and candidate.names[j] in unchanged):
                continue
            overlap = shape.intersection(candidate.shapes[j])
            if area_m2(overlap) <= minimum_area:
                continue
            introduced = introduced_ambiguity(overlap, candidate, new_tree, baseline, old_tree)
            first, second = candidate.names[i], candidate.names[j]
            if first in old_by_name and second in old_by_name:
                # Established owners must not gain a new pair-specific overlap.
                old_overlap = old_by_name[first].intersection(old_by_name[second])
                introduced = introduced.union(measurement_difference(overlap, old_overlap))
            if area_m2(introduced) > minimum_area:
                overlaps.append(finding(introduced, owners=[candidate.names[i], candidate.names[j]]))
        if i % 100 == 0:
            LOG.info("Compared overlap for %s/%s regions", i, len(candidate.names))
    current_by_name = dict(zip(candidate.names, candidate.shapes))
    for index, (name, old) in enumerate(zip(baseline.names, baseline.shapes)):
        lost = uncovered_loss(old, current_by_name.get(name, Polygon()), candidate.shapes, new_tree)
        if area_m2(lost) > minimum_area:
            losses.append(finding(lost, file=name))
        if index % 100 == 0:
            LOG.info("Compared retained coverage for %s/%s baseline regions", index, len(baseline.names))
    old_identity = baseline.metadata.get("file_identity", {})
    changed_metadata = [name for name, identity in candidate.metadata.get("file_identity", {}).items()
                        if name in old_identity and identity != old_identity[name]]
    report = {"changed_metadata": sorted(changed_metadata), "added_names": sorted(set(candidate.names) - set(baseline.names)),
              "removed_names": sorted(set(baseline.names) - set(candidate.names)),
              "invalid_rings": candidate.invalid_rings, "inherited_invalid_rings": [
                  item for item in candidate.invalid_rings if (item["file"], item["ring"]) in old_invalid],
              "new_invalid_rings": new_invalid,
              "new_invalid_files": sorted(set(candidate.invalid_files) - set(baseline.invalid_files)),
              "new_invalid_ring_contacts": new_contacts,
              "new_invalid_file_contacts": new_file_contacts,
              "introduced_overlap_parts": sorted(overlaps, key=lambda item: -item["approx_m2"]),
              "introduced_uncovered_parts": sorted(losses, key=lambda item: -item["approx_m2"])}
    report["regressions"] = any(report[key] for key in ("new_invalid_rings", "new_invalid_files",
                                                       "new_invalid_ring_contacts", "introduced_overlap_parts",
                                                       "new_invalid_file_contacts",
                                                       "introduced_uncovered_parts"))
    return report


def check(source, packed_path, baseline_source, baseline_packed_path, minimum_area=0.01,
          bounds_format="wrapped", baseline_bounds_format="wrapped"):
    LOG.info("Reading current source borders: %s", source)
    current_source = source_dataset(source)
    LOG.info("Reading baseline source borders: %s", baseline_source)
    old_source = source_dataset(baseline_source)
    LOG.info("Reading current packed geometry: %s", packed_path)
    current, clips = packed_datasets(packed_path, bounds_format)
    LOG.info("Reading baseline packed geometry: %s", baseline_packed_path)
    old, old_clips = packed_datasets(baseline_packed_path, baseline_bounds_format)
    if set(current.names) != set(current_source.names):
        raise ValueError("Packed names must match the current .poly filenames")
    if set(old.names) != set(old_source.names):
        raise ValueError("Baseline packed names must match the baseline .poly filenames")
    LOG.info("Comparing source geometry against baseline")
    source_report = compare(current_source, old_source, minimum_area)
    LOG.info("Comparing packed outer paths against baseline")
    path_report = compare(current, old, minimum_area)
    report = {"minimum_area_m2": minimum_area,
              "measurement": "Original ring even/odd fill, OR across outers, Mercator segments; physical ambiguity compares owner multiplicity; packed coverage uses wrapped CountryDef bounds",
              "source": source_report,
              "packed_paths": path_report,
              "current_source_metadata": current_source.metadata, "baseline_source_metadata": old_source.metadata,
              "current_packed_metadata": current.metadata, "baseline_packed_metadata": old.metadata}
    new_clips = []
    for name, geometry in clips.items():
        added = measurement_difference(geometry, old_clips.get(name, Polygon()))
        if area_m2(added) > minimum_area:
            new_clips.append(finding(added, file=name))
    old_unclosed = {(item["file"], item["ring"]) for item in old_source.metadata["unclosed_rings"]}
    new_unclosed = [item for item in current_source.metadata["unclosed_rings"] if (item["file"], item["ring"]) not in old_unclosed]
    report.update(new_metadata_clipping=new_clips, new_unclosed_source_rings=new_unclosed)
    report["regressions"] = any(report[label]["regressions"] for label in ("source", "packed_paths")) or bool(
        new_clips or new_unclosed)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--packed", type=Path, required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--baseline-packed", type=Path, required=True)
    parser.add_argument("--bounds-format", choices=("legacy", "wrapped"), default="wrapped",
                        help="Current CountryDef serialization: legacy signed or wrapped unsigned bounds")
    parser.add_argument("--baseline-bounds-format", choices=("legacy", "wrapped"), default="wrapped")
    parser.add_argument("--minimum-area", type=float, default=0.01)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if not math.isfinite(args.minimum_area) or args.minimum_area < 0:
        parser.error("Minimum area must be finite and nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    report = check(args.source, args.packed, args.baseline_source, args.baseline_packed, args.minimum_area,
                   args.bounds_format, args.baseline_bounds_format)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["regressions"]:
        for label in ("source", "packed_paths"):
            for category in ("new_invalid_rings", "new_invalid_files", "new_invalid_ring_contacts",
                             "new_invalid_file_contacts", "introduced_overlap_parts", "introduced_uncovered_parts"):
                findings = report[label][category]
                if findings:
                    examples = [item if isinstance(item, str) else item.get("file", "/".join(item.get("owners", [])))
                                for item in findings[:5]]
                    LOG.error("%s: %s: %s (%s)", label, category, len(findings), ", ".join(examples))
        for category in ("new_metadata_clipping", "new_unclosed_source_rings"):
            if report[category]:
                LOG.error("%s: %s", category, len(report[category]))
        LOG.error("Border geometry regressions; details: %s", args.report)
    else:
        LOG.info("No border geometry regressions; report: %s", args.report)
    raise SystemExit(1 if report["regressions"] else 0)


if __name__ == "__main__":
    main()
