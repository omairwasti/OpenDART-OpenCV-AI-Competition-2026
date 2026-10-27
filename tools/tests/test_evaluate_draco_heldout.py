"""Synthetic protocol/metric tests; no real held-out outcomes are inspected."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate_draco_heldout as e


def label(index,category,point=None,box=None):
    return dict(frame_index=index,product_stem=f'fixture{index}',label_class=category,
        confidence='medium',target_center_uv=point,target_visible_box_xywh=box,
        split='held_out',review_status='HUMAN_REVIEWED')


def prediction(point=None,box=None):
    return dict(selection_available=point is not None or box is not None,state='TRACKING',
        target_center_uv=point,visible_region_xywh=box)


class HeldoutEvaluationTests(unittest.TestCase):
    def setUp(self):self.config=e.dev.method.load_config()

    def test_unreviewed_boundary_entries_never_become_zero_iou(self):
        benchmark=[r for r in json.loads((e.PARENT/'benchmark.json').read_text()) if r['split']=='held_out']
        rows=copy.deepcopy(benchmark[:2]);rows[0].update(review_status='HUMAN_REVIEWED',
            human_inspection_attested=True,reviewer='SYNTHETIC TEST',reviewed_at_utc='2026-10-26T00:00:00Z',
            label_class='TARGET_NOT_CONFIDENTLY_IDENTIFIABLE',confidence='medium',note='Synthetic fixture')
        valid=e.reviewed_only(rows,benchmark)
        self.assertEqual(len(valid),1);self.assertEqual(valid[0]['frame_index'],rows[0]['frame_index'])
        self.assertEqual(rows[1]['review_status'],'UNREVIEWED')

    def test_non_heldout_or_analysis_only_rejected(self):
        for split in ('development','analysis_only'):
            with self.assertRaisesRegex(ValueError,'HELD-OUT'):e.reviewed_only([dict(split=split)],[])

    def test_scoring_refuses_unreviewed_input(self):
        a=label(1,'DIMORPHOS_RESOLVED_BODY',box=[0,0,10,10]);a['review_status']='UNREVIEWED'
        with self.assertRaisesRegex(ValueError,'reviewed HELD-OUT'):e.measure([a],{1:prediction(box=[0,0,10,10])},self.config)

    def test_point_counts_error_and_missing_measurement(self):
        labels=[label(i,'DIMORPHOS_POINT_OR_SMALL_BODY',[10,10]) for i in (1,2,3)]
        pred={1:prediction([13,14]),2:prediction([30,10]),3:prediction()}
        rows,summary=e.measure(labels,pred,self.config)
        self.assertEqual(summary['compact']['n'],3)
        self.assertEqual(summary['compact']['within_5px_n'],1)
        self.assertEqual(summary['compact']['selected_outside_5px_n'],1)
        self.assertEqual(summary['compact']['unavailable_n'],1)
        self.assertEqual(summary['compact']['centroid_error_n'],2)
        self.assertEqual(summary['compact']['centroid_error_median_px'],12.5)
        self.assertEqual(summary['label'],e.LABEL)
        self.assertIsNone(summary['p90_centroid_error_px'])

    def test_point_region_and_uncertain_metrics_are_separate(self):
        ann=[label(1,'DIMORPHOS_POINT_OR_SMALL_BODY',[10,10]),
             label(2,'DIMORPHOS_RESOLVED_BODY',box=[0,0,10,10]),label(3,'TARGET_NOT_CONFIDENTLY_IDENTIFIABLE')]
        pred={1:prediction([12,10]),2:prediction(box=[0,0,5,10]),3:prediction([50,50])}
        rows,s=e.measure(ann,pred,self.config)
        self.assertEqual(s['compact']['n'],1);self.assertEqual(s['regions']['n'],1)
        self.assertEqual(s['regions']['median_iou'],.5);self.assertEqual(s['uncertain']['selected_n'],1)
        self.assertIsNone(rows[1]['centroid_error_px']);self.assertIsNone(rows[2]['region_iou'])

    def test_designation_and_all_earlier_frames_excluded(self):
        ann=[label(i,'DIMORPHOS_RESOLVED_BODY',box=[0,0,10,10]) for i in (1,2,3)]
        pred={i:prediction(box=[0,0,10,10]) for i in (1,2,3)}
        rows,s=e.measure(ann,pred,self.config,2)
        self.assertEqual(s['excluded_pre_or_at_designation'],2)
        self.assertEqual(s['regions']['n'],1)
        self.assertEqual(s['regions']['median_iou'],1)

    def test_freeze_verification_rejects_modified_source_before_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);source=root/'method.py';source.write_text('frozen')
            cfg=root/'frozen_configuration.json';cfg.write_text('{}')
            (root/'FREEZE.json').write_text(json.dumps(dict(source_hashes={'method.py':e.dev.io.digest(source)},configuration_sha256=e.dev.io.digest(cfg))))
            e.verify_freeze(root,root);source.write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'source mismatch'):e.verify_freeze(root,root)

    def test_oneshot_guard_does_not_allow_repeated_evaluation(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);(out/'EVALUATION_STARTED.json').write_text('{}')
            with self.assertRaisesRegex(FileExistsError,'One-shot'):e.evaluate(out)


if __name__=='__main__':unittest.main()
