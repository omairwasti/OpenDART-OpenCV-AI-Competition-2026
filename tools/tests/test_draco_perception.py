"""Synthetic unit fixtures only; held-out annotations are never read."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import draco_perception as d
import draco_development as dev


def candidate(x,y,area=50,width=10,index=1):
    return dict(centroid_px=[x,y],area_px2=area,radius_px=float(np.sqrt(area/np.pi)),
        bbox_xywh=[x-width/2,y-width/2,width,width],feature_index=index)


def observation(candidates,shift=(0,0)):
    return dict(candidates=candidates,image_shift_px=list(shift))


class DracoSpecificTests(unittest.TestCase):
    def setUp(self):self.cfg=d.load_config()

    def pair(self,x=100):return [candidate(30,50,1000,36,0),candidate(x,80,50,10,1)]

    def test_autonomous_needs_confirmed_unique_pair_not_single_bright_blob(self):
        track=d.Track(self.cfg)
        self.assertFalse(track.update(observation([candidate(50,50,1000,36)]),0,0)['selection_available'])
        for i in range(1,4):value=track.update(observation(self.pair()),i,i)
        self.assertTrue(value['selection_available']);self.assertIn('AUTONOMOUS_ACQUIRED',value['events'])
        self.assertIn('UNVERIFIED',value['identity_status'])

    def test_ambiguous_secondaries_do_not_force_semantic_acquisition(self):
        track=d.Track(self.cfg);c=self.pair()+[candidate(100,130,45,10,2)]
        for i in range(5):value=track.update(observation(c),i,i)
        self.assertFalse(value['selection_available'])

    def test_one_time_designation_and_no_earlier_annotation_injection(self):
        click=dict(frame_index=2,visible_box_xywh=[95,75,10,10],center_uv=[100,80])
        track=d.Track(self.cfg,'human_designated',click)
        for i in range(2):self.assertEqual(track.update(observation(self.pair()),i,i)['state'],'AWAITING_HUMAN_DESIGNATION')
        value=track.update(observation(self.pair()),2,2)
        self.assertEqual(value['events'],['HUMAN_DESIGNATION'])
        self.assertEqual(value['state'],'TARGET_DESIGNATED')
        value=track.update(observation(self.pair(103)),3,3)
        self.assertEqual(value['target_center_uv'],[103.,80.]);self.assertNotIn('HUMAN_DESIGNATION',value['events'])

    def test_stale_gate_reset_exposes_reacquisition_identity_risk(self):
        track=d.Track(self.cfg)
        for i in range(3):track.update(observation(self.pair()),i,i)
        for i in range(3,10):value=track.update(observation([]),i,i)
        self.assertEqual(value['state'],'REACQUIRING');self.assertFalse(value['selection_available'])
        for i in range(10,13):value=track.update(observation(self.pair(350)),i,i)
        self.assertIn('REACQUIRED',value['events']);self.assertIn('STALE_GATE_RESET_IDENTITY_LINK_UNCERTAIN',value['events'])
        self.assertEqual(value['target_center_uv'],[350.,80.])

    def test_current_image_shift_used_without_future_positions(self):
        click=dict(frame_index=0,visible_box_xywh=[95,75,10,10],center_uv=[100,80])
        track=d.Track(self.cfg,'human_designated',click);track.update(observation(self.pair()),0,0)
        value=track.update(observation([candidate(400,380)],(300,300)),1,1)
        self.assertTrue(value['selection_available']);self.assertEqual(value['target_center_uv'],[400.,380.])

    def test_resolved_transition_and_collective_fragment_support(self):
        click=dict(frame_index=0,visible_box_xywh=[110,110,80,80],center_uv=[150,150])
        track=d.Track(self.cfg,'human_designated',click)
        track.update(observation([candidate(150,150,5000,80)]),0,0)
        value=track.update(observation([candidate(150,150,5000,80)]),1,1)
        self.assertEqual(value['state'],'RESOLVED_BODY')
        fragments=[candidate(x,y,500,15,i) for i,(x,y) in enumerate(( (x,y) for x in (120,135,150,165,180) for y in (135,165)))]
        value=track.update(observation(fragments),2,2)
        self.assertEqual(value['state'],'RESOLVED_BODY');self.assertTrue(value['selection_available'])
        self.assertEqual(value['region_component_count'],10);self.assertIsNone(value['selected_feature_index'])

    def test_partial_reports_only_transmitted_region_not_full_target(self):
        track=d.Track(self.cfg)
        value=track.update(observation([candidate(500,600,5000,80)]),0,0,True)
        self.assertEqual(value['state'],'PARTIAL_FRAME');self.assertFalse(value['selection_available'])
        self.assertIsNone(value['target_center_uv']);self.assertEqual(value['representation'],'TRANSMITTED_VISIBLE_REGION_ONLY')

    def test_features_are_past_only_and_reject_small_stationary_component(self):
        cfg=dict(self.cfg,persistence_alpha=.5,persistence_warmup_frames=2,persistence_probability=.4)
        feature=d.Features(cfg);cv=d.require_opencv();base=np.zeros((1024,1024),np.uint8)
        # A one-pixel-radius source becomes a 13px2 measured component, inside
        # the documented <=24px2 persistence-rejection scope.
        cv.circle(base,(100,100),1,220,-1)
        outputs=[]
        for i in range(4):
            image=base.copy();cv.circle(image,(400+20*i,400),15,220,-1)
            original=image.copy();outputs.append(feature.update(image,float(i)))
            self.assertTrue(np.array_equal(image,original))
        self.assertEqual(len(outputs[0]['rejected_persistent']),0)
        self.assertGreater(len(outputs[-1]['rejected_persistent']),0)
        self.assertTrue(any(c['area_px2']>24 for c in outputs[-1]['candidates']))

    def test_deterministic_prefix_outputs_do_not_depend_on_future_observations(self):
        a=d.Track(self.cfg);b=d.Track(self.cfg)
        for i in range(8):
            x=a.update(observation(self.pair(100+i)),i,i);y=b.update(observation(self.pair(100+i)),i,i)
            x.pop('association_wall_s');y.pop('association_wall_s');self.assertEqual(x,y)
        b.update(observation([candidate(900,900)]),8,8)
        self.assertEqual(a.position.tolist(),[107.,80.])

    def test_strict_time_order_and_lossless_serialization(self):
        track=d.Track(self.cfg);value=track.update(observation([]),0,0)
        self.assertEqual(json.loads(json.dumps(value,allow_nan=False)),value)
        with self.assertRaises(ValueError):track.update(observation([]),0,1)

    def test_metrics_development_only_and_designation_sample_excluded(self):
        a=dict(frame_index=1,product_stem='test',label_class='DIMORPHOS_POINT_OR_SMALL_BODY',
            confidence='high',target_center_uv=[100,80],target_visible_box_xywh=None)
        p={1:dict(selection_available=True,state='TRACKING',target_center_uv=[101,80],visible_region_xywh=[95,75,10,10])}
        _,metrics=dev.evaluate([a],p,self.cfg)
        self.assertEqual(metrics['compact']['within_5px_n'],1);self.assertEqual(metrics['compact']['n'],1)
        _,metrics=dev.evaluate([a],p,self.cfg,1)
        self.assertEqual(metrics['compact']['n'],0);self.assertEqual(metrics['excluded_pre_or_at_designation'],1)

    def test_bbox_iou_and_source_file_preservation(self):
        self.assertAlmostEqual(dev.iou([0,0,10,10],[5,0,10,10]),1/3)
        cv=d.require_opencv()
        with tempfile.TemporaryDirectory() as folder:
            png=Path(folder)/'source.png';xml=Path(folder)/'source.xml'
            image=np.zeros((1024,1024),np.uint8);cv.circle(image,(300,300),10,200,-1)
            cv.imwrite(str(png),image);xml.write_text('<original/>')
            before=[dev.io.digest(p) for p in (png,xml)]
            d.Features(self.cfg).update(cv.imread(str(png),cv.IMREAD_UNCHANGED),0)
            self.assertEqual(before,[dev.io.digest(p) for p in (png,xml)])


if __name__=='__main__':unittest.main()
