"""Offline acquisition safety/selection tests; no OpenDART runtime imports."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
from datetime import timedelta

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import download_draco_terminal as d

STEM='dart_0401925587_30267_02_iof'
ITEM=dict(stem=STEM,png=STEM+'.png',xml=STEM+'.xml',filename_sclk_seconds=401925587,
          filename_sclk_subticks=30267,revision=2,png_listed_upper_bytes=100,xml_listed_upper_bytes=1000)

def label(timestamp=d.START):
    stop=(d.utc(timestamp)+timedelta(milliseconds=5)).isoformat().replace('+00:00','Z')
    return f'''<Product_Observational xmlns="urn:pds"><Observation_Area><Time_Coordinates>
    <start_date_time>{timestamp}</start_date_time><stop_date_time>{stop}</stop_date_time>
    </Time_Coordinates><acquisition_time>401925587:25332</acquisition_time></Observation_Area>
    <file_name>{STEM}.png</file_name><file_name>{STEM}.fits</file_name></Product_Observational>'''.encode()


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        d.configure('terminal')

    def tearDown(self):
        d.configure('terminal')

    def test_listing_deduplicates_anchors_and_never_enumerates_fits(self):
        html='\n'.join(f'<a href="{STEM}.{ext}">icon</a> <a href="{STEM}.{ext}">name</a> 2023-08-04 07:07 18K'
                       for ext in ('png','xml','fits'))
        parsed=d.parse_index(html)
        self.assertEqual(len(parsed['products']),1)
        self.assertEqual(parsed['matching_anchor_occurrences'],4)
        self.assertNotIn('fits',parsed['products'][0])

    def test_namespace_utc_and_clock_are_separate_from_filename(self):
        parsed=d.parse_label(label(),ITEM)
        self.assertEqual(parsed['acquisition_utc'],d.START)
        self.assertEqual(parsed['label_sclk'],'401925587:25332')
        self.assertNotEqual(parsed['label_sclk_subticks'],ITEM['filename_sclk_subticks'])

    def test_missing_utc_stops(self):
        with self.assertRaises(d.MissingUTC):
            d.parse_label(label().replace(b'start_date_time',b'creation_date_time'),ITEM)

    def test_nested_soc_clock_is_preserved_without_filename_conversion(self):
        payload=label().replace(b'<acquisition_time>401925587:25332</acquisition_time>',
            b'<acquisition_time>\n<soc_acquisition_time>0401925587:25332</soc_acquisition_time>\n</acquisition_time>')
        parsed=d.parse_label(payload,ITEM)
        self.assertEqual(parsed['label_sclk'],'0401925587:25332')
        self.assertEqual(parsed['label_sclk_source'],'soc_acquisition_time')
        self.assertEqual(parsed['label_sclk_seconds'],401925587)
        self.assertEqual(parsed['label_sclk_subticks'],25332)

    def test_label_must_reference_matching_preview(self):
        with self.assertRaises(d.StopAcquisition):
            d.parse_label(label().replace((STEM+'.png').encode(),b'unrelated.png'),ITEM)

    def test_url_guard_refuses_fits_before_any_http_request(self):
        with tempfile.TemporaryDirectory() as folder,patch('urllib.request.urlopen') as opened:
            with self.assertRaises(d.StopAcquisition): d.Client(folder).get(d.SOURCE+STEM+'.fits',999)
            opened.assert_not_called()

    def test_atomic_write_and_managed_destination_guard(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'sample.xml'; d.atomic(path,b'complete')
            self.assertEqual(path.read_bytes(),b'complete')
            self.assertFalse(list(Path(folder).glob('*.part-*')))
            with self.assertRaises(d.StopAcquisition): d.prepare(folder)
            with self.assertRaises(d.StopAcquisition): d.prepare(Path(folder)/'269')

    def test_complete_pair_resume_uses_no_network(self):
        with tempfile.TemporaryDirectory() as folder:
            out=d.prepare(Path(folder)/'owned'); products=out/'products'
            xml=label(); png=d.PNG_SIGNATURE+b'preview'+d.PNG_END
            (products/ITEM['xml']).write_bytes(xml);(products/ITEM['png']).write_bytes(png)
            client=d.Client(out,interval=0)
            with patch.object(client,'get',side_effect=AssertionError('Unexpected network')):
                rows=d.acquire(out,client,dict(candidates=[ITEM]),{},1,10000)
            self.assertEqual(rows[0]['download_status'],'SKIPPED_COMPLETE')
            self.assertEqual(rows[0]['png_sha256'],hashlib.sha256(png).hexdigest())

    def test_end_utc_is_exclusive(self):
        with tempfile.TemporaryDirectory() as folder:
            out=d.prepare(Path(folder)/'owned'); client=d.Client(out,interval=0)
            with patch.object(client,'get',return_value=label(d.END)) as get:
                rows=d.acquire(out,client,dict(candidates=[ITEM]),{},1,10000)
            self.assertEqual(rows,[])
            self.assertEqual(get.call_count,1)
            self.assertFalse(list((out/'products').iterdir()))

    def test_budget_stops_before_file_commit(self):
        with tempfile.TemporaryDirectory() as folder:
            out=d.prepare(Path(folder)/'owned'); client=d.Client(out,interval=0)
            with patch.object(client,'get',return_value=label()):
                with self.assertRaises(d.StopAcquisition):
                    d.acquire(out,client,dict(candidates=[ITEM]),{},1,10)
            self.assertFalse(list((out/'products').iterdir()))

    def test_repeated_throttling_stops_after_conservative_retries(self):
        with tempfile.TemporaryDirectory() as folder:
            client=d.Client(folder,interval=0)
            error=urllib.error.HTTPError(d.SOURCE,429,'throttle',{},None)
            with patch('urllib.request.urlopen',side_effect=error),patch('time.sleep'):
                with self.assertRaises(d.StopAcquisition): client.get(d.SOURCE,999)
            self.assertEqual(client.requests,3)
            self.assertEqual(client.retries,2)
            self.assertTrue(client.stop.is_set())

    def test_final_scope_is_entire_collection_without_utc_clipping(self):
        d.configure('final')
        self.assertTrue(d.SOURCE.endswith('/final/2022/269/'))
        self.assertIsNone(d.START)
        self.assertIsNone(d.END)
        self.assertTrue(d.in_scope('2022-09-26T23:14:23.328Z'))
        self.assertEqual(d.DEFAULT_OUT.name,'final_2022-269')

    def test_final_cannot_reuse_a_terminal_destination(self):
        with tempfile.TemporaryDirectory() as folder:
            out=d.prepare(Path(folder)/'owned')
            before=(out/'acquisition.json').read_bytes()
            d.configure('final')
            with self.assertRaises(d.StopAcquisition): d.prepare(out)
            self.assertEqual((out/'acquisition.json').read_bytes(),before)

    def test_final_url_guard_rejects_other_collection(self):
        terminal_url=d.SOURCE+STEM+'.xml'
        d.configure('final')
        with tempfile.TemporaryDirectory() as folder,patch('urllib.request.urlopen') as opened:
            with self.assertRaises(d.StopAcquisition): d.Client(folder).get(terminal_url,999)
            opened.assert_not_called()

    def test_bad_image_false_does_not_hide_curated_final_partial(self):
        stem=d.LAST_PARTIAL
        payload=label('2022-09-26T23:14:23.328Z').replace(STEM.encode(),stem.encode())
        payload=payload.replace(b'</Observation_Area>',b'<bad_image>false</bad_image><mission_phase>final</mission_phase></Observation_Area>')
        parsed=d.parse_label(payload,dict(stem=stem,png=stem+'.png'))
        self.assertFalse(parsed['pds_bad_image'])
        self.assertFalse(parsed['pds_partial_flag_present'])
        self.assertTrue(parsed['is_partial'])
        self.assertEqual(parsed['image_completeness'],'PARTIAL_CURATED')

    def test_generic_missing_constant_is_not_a_partial_flag(self):
        payload=label().replace(b'</Observation_Area>',b'<missing_constant>1E10</missing_constant><bad_image>false</bad_image></Observation_Area>')
        parsed=d.parse_label(payload,ITEM)
        self.assertFalse(parsed['is_partial'])
        self.assertNotIn('missing_constant',parsed['pds_quality_metadata'])

    def test_xml_bad_flag_is_preserved(self):
        payload=label().replace(b'</Observation_Area>',b'<bad_image>true</bad_image></Observation_Area>')
        parsed=d.parse_label(payload,ITEM)
        self.assertTrue(parsed['pds_bad_image'])
        self.assertEqual(parsed['pds_quality_metadata']['bad_image'],'true')


if __name__=='__main__': unittest.main()
