"""Logical index ordering safeguards; no detector/replay or network."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from index_draco_collections import chronological
from download_draco_terminal import StopAcquisition


class IndexTests(unittest.TestCase):
    def test_orders_by_xml_utc_instead_of_product_name(self):
        rows=[dict(product_stem='a',acquisition_utc='2022-09-26T23:10:01Z'),
              dict(product_stem='z',acquisition_utc='2022-09-26T23:09:59Z')]
        self.assertEqual([r['product_stem'] for r in chronological(rows)],['z','a'])

    def test_duplicate_timestamp_is_not_silently_tie_broken(self):
        with self.assertRaises(StopAcquisition):
            chronological([dict(product_stem='a',acquisition_utc='2022-09-26T23:10:01Z'),
                           dict(product_stem='b',acquisition_utc='2022-09-26T23:10:01.000Z')])

    def test_duplicate_product_is_rejected(self):
        with self.assertRaises(StopAcquisition):
            chronological([dict(product_stem='a',acquisition_utc='2022-09-26T23:10:01Z'),
                           dict(product_stem='a',acquisition_utc='2022-09-26T23:10:02Z')])
