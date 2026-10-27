"""Adapter invariants; real detector/tracker defaults are never changed."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import replay_draco_frozen as r


def candidate(u=70,v=60,radius=4):
    return r.Candidate(np.array([u,v],float),radius,np.eye(2)*2.25,
        np.pi*radius**2,(int(u-radius),int(v-radius),2*radius,2*radius),.8).to_dict()


def frame(index,candidates,partial=False):
    return dict(frame_index=index,product_stem=r.PARTIAL if partial else f'frame{index}',
        acquisition_utc=f'2022-09-26T22:00:{index:02d}Z',elapsed_s=float(index),
        is_partial=partial,relative_to_impact_s=-4400+index,detector_wall_s=.01,
        candidate_count=len(candidates),candidates=candidates)


class FrozenReplayTests(unittest.TestCase):
    def test_index_requires_strict_utc_order_not_just_filename_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.json'
            rows=[frame(1,[]),frame(0,[])]
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'strictly ordered'):
                r.load_index(path,expected_count=None)
            rows.reverse();path.write_text(json.dumps(rows))
            self.assertEqual([x['frame_index'] for x in r.load_index(path,None)],[0,1])

    def test_duplicate_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.json'; rows=[frame(0,[]),frame(1,[])]
            rows[1]['acquisition_utc']=rows[0]['acquisition_utc']
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'strictly ordered'):r.load_index(path,None)

    def test_partial_flag_cannot_be_dropped(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'index.json'; rows=[frame(0,[],partial=True)]
            rows[0]['is_partial']=False;path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError,'partial status'):r.load_index(path,None)

    def test_empty_candidates_are_explicit_search_failure(self):
        rows=[frame(0,[])];r.associate(rows)
        self.assertEqual(rows[0]['track_status'],'SEARCHING')
        self.assertIsNone(rows[0]['selected_candidate_index'])
        self.assertEqual(rows[0]['association_failure_reason'],'NO_CANDIDATES')
        self.assertEqual(r.statistics(rows)['no_candidate_frames'],1)

    def test_single_unknown_candidate_does_not_force_identity(self):
        rows=[frame(0,[candidate()])];r.associate(rows)
        self.assertIsNone(rows[0]['selected_candidate_index'])
        self.assertEqual(rows[0]['association_failure_reason'],'NO_IMAGE_ONLY_ACQUISITION_HYPOTHESIS')

    def test_multi_candidate_serialization_roundtrip_is_lossless(self):
        values=[candidate(30,30,15),candidate(70,60,4)]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'data.json';r.save(path,values)
            loaded=json.loads(path.read_text())
            for a,b in zip(values,loaded):self.assertEqual(r.decode_candidate(b).to_dict(),a)
            table=Path(folder)/'frames.csv';r.table(table,[dict(candidates=loaded)])
            with table.open() as stream:
                self.assertEqual(json.loads(next(csv.DictReader(stream))['candidates']),loaded)
            with self.assertRaises(FileExistsError):r.save(path,[])

    def test_association_deterministic_loss_and_reacquisition(self):
        original=[frame(0,[candidate(30,30,15),candidate(70,60,4)])]
        original += [frame(i,[]) for i in range(1,8)]
        original += [frame(8,[candidate(30,30,15),candidate(70,60,4)])]
        a=copy.deepcopy(original);b=copy.deepcopy(original)
        self.assertEqual(r.associate(a),r.associate(b))
        for x,y in zip(a,b):
            x.pop('association_wall_s');y.pop('association_wall_s');self.assertEqual(x,y)
        self.assertEqual(a[0]['selected_candidate_index'],1)
        self.assertEqual(a[6]['events'],['ENTERED_LOST'])
        self.assertEqual(a[-1]['track_status'],'REACQUIRED')

    def test_partial_excluded_from_normal_aggregate_and_windows(self):
        rows=[frame(0,[candidate()]),frame(1,[],partial=True)];r.associate(rows)
        normal=r.statistics([x for x in rows if not x['is_partial']])
        self.assertEqual(normal['frames'],1);self.assertEqual(normal['candidate_frame_fraction'],1)
        self.assertEqual(sum(w['frames'] for w in r.windows(rows).values()),1)

    def test_detection_does_not_modify_input_or_source_png_xml(self):
        cv=r.require_opencv();image=np.zeros((100,100),np.uint8)
        cv.circle(image,(60,60),10,220,-1)
        original=image.copy()
        with tempfile.TemporaryDirectory() as folder:
            png=Path(folder)/'source.png';xml=Path(folder)/'source.xml'
            self.assertTrue(cv.imwrite(str(png),image));xml.write_text('<label/>')
            before=[r.digest(p) for p in (png,xml)]
            detector=r.AsteroidDetector();a=detector.detect(image);b=detector.detect(image)
            self.assertTrue(np.array_equal(image,original))
            self.assertEqual([x.to_dict() for x in a],[x.to_dict() for x in b])
            self.assertEqual(before,[r.digest(p) for p in (png,xml)])


if __name__=='__main__': unittest.main()
