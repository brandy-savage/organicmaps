import unittest

from shapely import STRtree, prepare
from shapely.geometry import GeometryCollection, LineString, MultiPoint, Point, Polygon

from audit_borders import area_m2, uncovered_loss
from border_geometry import measurement_difference


class MeasurementTests(unittest.TestCase):
    def test_point_on_line_in_mixed_collection_is_inherited(self):
        point = Point(13.58948, 56.31274395355062)
        old = GeometryCollection([LineString([(point.x, point.y), (13.58949, point.y)]), Point(13, 56)])
        self.assertTrue(old.covers(point))
        self.assertTrue(measurement_difference(point, old).is_empty)
        new = GeometryCollection([point, Point(12, 55)])
        self.assertTrue(measurement_difference(new, old).equals(Point(12, 55)))

    def test_mixed_collection_preserves_new_points_and_line_parts(self):
        old = GeometryCollection([LineString([(0, 0), (2, 0)]), Point(5, 0)])
        new = GeometryCollection([LineString([(1, 0), (3, 0)]), MultiPoint([(1, 0), (5, 0), (6, 0)])])
        difference = measurement_difference(new, old)
        self.assertTrue(difference.equals(GeometryCollection([LineString([(2, 0), (3, 0)]), Point(6, 0)])))

    def test_near_collinear_source_area_is_retained_by_exact_shared_neighbor(self):
        # A legacy overlay reported a 0.36 m² sliver here despite the current
        # regions using the exact same A-C segment in opposite directions.
        a = (31.21306, -14.84761133688672)
        b = (31.21999, -14.846587911840558)
        c = (31.23056, -14.845026939127916)
        below = [(c[0], c[1] - .01), (a[0], a[1] - .01)]
        old = Polygon([a, b, c, *below])
        current = Polygon([a, c, *below])
        neighbor = Polygon([c, a, (a[0], a[1] + .01), (c[0], c[1] + .01)])
        occupants = [current, neighbor]
        prepare(occupants)
        lost = uncovered_loss(old, current, occupants, STRtree(occupants))
        self.assertLess(area_m2(lost), .01)

    def test_real_submillimetre_gap_is_not_rounded_away(self):
        old = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        current = Polygon([(0, 0), (.5, 0), (.5, 1), (0, 1)])
        neighbor = Polygon([(.5 + 1e-9, 0), (1, 0), (1, 1), (.5 + 1e-9, 1)])
        occupants = [current, neighbor]
        lost = uncovered_loss(old, current, occupants, STRtree(occupants))
        self.assertGreater(area_m2(lost), .01)
        self.assertAlmostEqual(lost.area, 1e-9, delta=1e-16)
