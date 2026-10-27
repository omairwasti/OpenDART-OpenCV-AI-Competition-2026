"""Annotation gate tests use synthetic attestations, never real human labels."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_draco_review as r


class ReviewGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=r.construct(r.frozen.load_index(r.ROOT/'data/draco/draco_terminal_to_impact_manifest.json'))

    def reviewed(self):
        item=copy.deepcopy(self.rows[0])
        item.update(review_status='HUMAN_REVIEWED',human_inspection_attested=True,
            reviewer='SYNTHETIC UNIT TEST',reviewed_at_utc='2026-10-26T00:00:00Z',
            label_class='TARGET_NOT_CONFIDENTLY_IDENTIFIABLE',confidence='low',note='Unit test fixture, not ground truth')
        return item

    def test_sample_unique_ordered_and_stratified(self):
        self.assertEqual(len(self.rows),63)
        self.assertEqual(len({x['product_stem'] for x in self.rows}),63)
        self.assertEqual(len({x['stratum'] for x in self.rows}),9)
        self.assertEqual([x['frame_index'] for x in self.rows],sorted(x['frame_index'] for x in self.rows))
        self.assertEqual(self.rows[-1]['product_stem'],r.frozen.PARTIAL)

    def test_deterministic_split_and_special_frames(self):
        self.assertEqual(r.split_members(['a','b','c']),r.split_members(['c','b','a']))
        counts=r.Counter(x['split'] for x in self.rows)
        self.assertEqual(counts,dict(development=41,held_out=20,analysis_only=2))
        self.assertTrue(all(x['split']=='analysis_only' for x in self.rows if x['stratum'].startswith('final_')))

    def test_initial_annotations_are_not_ground_truth(self):
        self.assertEqual(r.validate_reviews(self.rows,self.rows),[])
        self.assertTrue(all(x['label_class'] is None and not x['human_inspection_attested'] for x in self.rows))

    def test_draft_proposal_is_never_counted_as_reviewed(self):
        item=copy.deepcopy(self.rows[0]);item['label_class']='DIMORPHOS_POINT_OR_SMALL_BODY';item['target_center_uv']=[100,200]
        self.assertEqual(r.validate_reviews([item],self.rows),[])

    def test_review_requires_explicit_attestation_reviewer_time_note(self):
        for key,value in [('human_inspection_attested',False),('reviewer',''),('reviewed_at_utc',''),('note','')]:
            item=self.reviewed();item[key]=value
            with self.assertRaises(ValueError):r.validate_reviews([item],self.rows)

    def test_tampered_split_or_source_metadata_rejected(self):
        for key,value in [('split','held_out'),('is_partial',True),('png_sha256','bad'),('acquisition_utc','2022-09-26T22:01:00Z')]:
            item=self.reviewed();item[key]=value
            with self.assertRaises(ValueError):r.validate_reviews([item],self.rows)

    def test_resolved_region_does_not_require_point(self):
        item=self.reviewed();item.update(label_class='DIMORPHOS_RESOLVED_BODY',target_visible_box_xywh=[300,200,400,500])
        self.assertEqual(len(r.validate_reviews([item],self.rows)),1)
        self.assertIsNone(item['target_center_uv'])

    def test_compact_target_requires_defensible_point(self):
        item=self.reviewed();item['label_class']='DIMORPHOS_POINT_OR_SMALL_BODY'
        with self.assertRaises(ValueError):r.validate_reviews([item],self.rows)
        item['target_center_uv']=[511.5,612.2]
        self.assertEqual(len(r.validate_reviews([item],self.rows)),1)

    def test_ambiguous_class_has_no_target_geometry(self):
        item=self.reviewed();item['target_center_uv']=[100,200]
        with self.assertRaises(ValueError):r.validate_reviews([item],self.rows)

    def test_partial_remains_separate_and_no_full_extent_required(self):
        item=copy.deepcopy(self.rows[-1]);item.update(review_status='HUMAN_REVIEWED',human_inspection_attested=True,
            reviewer='SYNTHETIC UNIT TEST',reviewed_at_utc='2026-10-26T00:00:00Z',
            label_class='PARTIAL_FRAME',confidence='high',note='Synthetic fixture; transmitted strip only')
        self.assertEqual(len(r.validate_reviews([item],self.rows)),1)
        item['label_class']='DIMORPHOS_RESOLVED_BODY'
        with self.assertRaises(ValueError):r.validate_reviews([item],self.rows)

    def test_geometry_finite_inside_image_and_nonzero_box(self):
        for value,box in [([float('nan'),5],False),([True,1],False),([1024,0],False),([0,0,0,1],True),([1000,1000,30,30],True)]:
            with self.assertRaises(ValueError):r.geometry(value,box)

    def test_duplicate_and_roundtrip(self):
        item=self.reviewed();encoded=json.loads(json.dumps([item],allow_nan=False))
        self.assertEqual(r.validate_reviews(encoded,self.rows),encoded)
        with self.assertRaises(ValueError):r.validate_reviews(encoded*2,self.rows)


if __name__=='__main__':unittest.main()
