"""Read-only source collection inspection and a logical UTC-sorted DRACO index.

No images are copied, detected or replayed. No OpenDART runtime is imported.
"""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path

import download_draco_terminal as draco

ROOT=Path(__file__).resolve().parents[1]


def chronological(rows):
    ordered=sorted(rows,key=lambda r:draco.utc(r['acquisition_utc']))
    if len({r['product_stem'] for r in ordered})!=len(ordered):
        raise draco.StopAcquisition('Combined index has duplicate product stems')
    if any(draco.utc(a['acquisition_utc'])>=draco.utc(b['acquisition_utc']) for a,b in zip(ordered,ordered[1:])):
        raise draco.StopAcquisition('Combined index is not strictly chronological; duplicate UTC or ordering anomaly')
    return ordered


def read_collection(directory, phase):
    directory=Path(directory).resolve()
    manifest=json.loads((directory/'manifest.json').read_text())
    summary=json.loads((directory/'summary.json').read_text())
    if not summary['complete']:
        raise draco.StopAcquisition('Source collection did not pass acquisition validation')
    result=[]
    for original in manifest:
        if original['validation_status']!='VALID':
            raise draco.StopAcquisition('Source manifest contains an undecoded image')
        png=directory/original['png_relative_path']; xml=directory/original['xml_relative_path']
        if not png.is_file() or not xml.is_file() or not png.stat().st_size or not xml.stat().st_size:
            raise draco.StopAcquisition('Missing or empty source pair')
        parsed=draco.parse_label(xml.read_bytes(),dict(stem=original['product_stem'],png=png.name))
        if parsed['acquisition_utc']!=original['acquisition_utc'] or parsed['label_mission_phase']!=phase:
            raise draco.StopAcquisition('Original manifest UTC/phase disagrees with actual PDS XML')
        h,w=original['image_shape'][:2]
        validity='PARTIAL' if parsed['is_partial'] else 'XML_BAD' if parsed['pds_bad_image'] else 'DECODED_NO_BAD_FLAG'
        result.append(dict(mission_phase=phase,product_stem=original['product_stem'],**parsed,
            png_local_path=str(png),xml_local_path=str(xml),width=w,height=h,
            dtype=original['image_dtype'],channels=original['image_channels'],
            png_sha256=original['png_sha256'],xml_sha256=original['xml_sha256'],
            filename_sclk_seconds=original['filename_sclk_seconds'],filename_sclk_subticks=original['filename_sclk_subticks'],
            product_validity=validity,decode_status=original['validation_status']))
    return chronological(result)


def build(terminal, final, output):
    output=Path(output).resolve()
    json_path=output.with_suffix('.json')
    summary_path=output.with_name(output.stem+'_summary.json')
    if any(p.exists() for p in (output,json_path,summary_path)):
        raise draco.StopAcquisition('Combined index output already exists; preserve it and select a new output name')
    terminal_rows=read_collection(terminal,'terminal')
    final_rows=read_collection(final,'final')
    if terminal_rows[-1]['product_stem']!='dart_0401929787_00585_02_iof' or terminal_rows[-1]['acquisition_utc']!='2022-09-26T23:09:59.503Z':
        raise draco.StopAcquisition('Terminal endpoint differs from the approved dataset')
    rows=chronological(terminal_rows+final_rows)
    gap=(draco.utc(final_rows[0]['acquisition_utc'])-draco.utc(terminal_rows[-1]['acquisition_utc'])).total_seconds()
    if gap<=0: raise draco.StopAcquisition('Source phases overlap or are reversed')
    def structures(values):
        return sorted({(r['height'],r['width'],r['dtype'],r['channels']) for r in values})
    terminal_structures=structures(terminal_rows); final_structures=structures(final_rows)
    result=dict(products=len(rows),phase_counts=dict(Counter(r['mission_phase'] for r in rows)),
        first_product=rows[0]['product_stem'],first_utc=rows[0]['acquisition_utc'],
        last_product=rows[-1]['product_stem'],last_utc=rows[-1]['acquisition_utc'],strictly_chronological=True,
        duplicate_products=0,duplicate_timestamps=0,partial_products=[r['product_stem'] for r in rows if r['is_partial']],
        xml_bad_products=[r['product_stem'] for r in rows if r['pds_bad_image']],
        terminal_last=dict(product=terminal_rows[-1]['product_stem'],utc=terminal_rows[-1]['acquisition_utc']),
        final_first=dict(product=final_rows[0]['product_stem'],utc=final_rows[0]['acquisition_utc']),
        boundary_gap_s=gap,significant_gap_threshold_s=5.,significant_boundary_gap=gap>5,
        terminal_structures=terminal_structures,final_structures=final_structures,
        native_dimensions_dtype_channels_change=terminal_structures!=final_structures,
        filename_conventions_match=all(draco.PRODUCT.fullmatch(Path(r['png_local_path']).name) for r in rows),
        pds_product_versions=dict(Counter(r['pds_product_version'] for r in rows)),
        no_source_files_moved_or_copied=True,
        scientific_boundary='Recorded real imagery for perception/replay; not a causal closed-loop environment. Synthetic/AWS simulation remains causal evidence.')
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
    writer.writeheader()
    for row in rows:
        writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    draco.atomic(output,stream.getvalue().encode())
    draco.save_json(json_path,rows)
    draco.save_json(summary_path,result)
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--terminal',type=Path,default=ROOT/'data/draco/terminal_2022-269_2200-2310')
    parser.add_argument('--final',type=Path,default=ROOT/'data/draco/final_2022-269')
    parser.add_argument('--out',type=Path,default=ROOT/'data/draco/draco_terminal_to_impact_manifest.csv')
    args=parser.parse_args()
    build(args.terminal,args.final,args.out)
