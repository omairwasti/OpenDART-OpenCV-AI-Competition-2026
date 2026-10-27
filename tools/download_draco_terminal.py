"""Bounded acquisition/inspection of official calibrated-collection PNG previews.

Standalone: never imported by OpenDART. No FITS request is permitted. Prerecorded
images support later perception/replay evidence, not a causal control environment.
Acquisition uses stdlib; local validation/contact-sheet export uses existing cv2.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import http.client
from html import unescape
import io
import json
import math
from pathlib import Path
import re
import os
import statistics
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET

SOURCE = 'https://pds-smallbodies.astro.umd.edu/holdings/pds4-dart:data_dracocal-v3.0/terminal/2022/269/'
START = '2022-09-26T22:00:00Z'
END = '2022-09-26T23:10:00Z'
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT/'data/draco/terminal_2022-269_2200-2310'
PRODUCT = re.compile(r'^(dart_(\d{10})_(\d{5})_(\d{2})_iof)\.(png|xml)$')
USER_AGENT = 'OpenDART bounded scientific acquisition/1.0'
TARGETS=['2022-09-26T22:00:00Z','2022-09-26T22:24:00Z','2022-09-26T22:45:00Z',
         '2022-09-26T23:00:00Z','2022-09-26T23:09:00Z']
PNG_SIGNATURE=b'\x89PNG\r\n\x1a\n'
PNG_END=b'\x00\x00\x00\x00IEND\xaeB`\x82'
# Display-only, manually chosen after viewing these five full PNGs. No detector.
PREVIEW_CENTRES=[(512,628),(629,615),(466,725),(462,739),(390,335)]
COLLECTION='terminal'
IMPACT_UTC='2022-09-26T23:14:24.183Z'  # User-supplied approximate reference; not a clock conversion.
LAST_COMPLETE='dart_0401930049_43695_02_iof'
LAST_PARTIAL='dart_0401930050_41838_02_iof'
NASA_FINAL='https://www.nasa.gov/solar-system/darts-final-images-prior-to-impact/'


def configure(collection):
    """Select one collection for a standalone invocation; never merge source files."""
    global COLLECTION,SOURCE,START,END,DEFAULT_OUT,TARGETS
    COLLECTION=collection
    SOURCE=f'https://pds-smallbodies.astro.umd.edu/holdings/pds4-dart:data_dracocal-v3.0/{collection}/2022/269/'
    if collection=='terminal':
        START,END='2022-09-26T22:00:00Z','2022-09-26T23:10:00Z'
        DEFAULT_OUT=ROOT/'data/draco/terminal_2022-269_2200-2310'
        TARGETS=['2022-09-26T22:00:00Z','2022-09-26T22:24:00Z','2022-09-26T22:45:00Z',
                 '2022-09-26T23:00:00Z','2022-09-26T23:09:00Z']
    elif collection=='final':
        START=END=None  # Entire published final collection; no additional UTC clipping.
        DEFAULT_OUT=ROOT/'data/draco/final_2022-269'
        TARGETS=['2022-09-26T23:10:00Z']+[(utc(IMPACT_UTC)-timedelta(seconds=s)).isoformat().replace('+00:00','Z')
                                       for s in (180,120,60,30,10)]
    else:
        raise ValueError('Unknown DRACO collection')


def in_scope(acquisition_utc):
    value=utc(acquisition_utc)
    return (START is None or value>=utc(START)) and (END is None or value<utc(END))


class StopAcquisition(RuntimeError):
    pass


class MissingUTC(StopAcquisition):
    pass


def utc(text):
    value = datetime.fromisoformat(text.replace('Z','+00:00'))
    if value.tzinfo is None or value.utcoffset().total_seconds() != 0:
        raise MissingUTC('Expected an explicit UTC timestamp')
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, data):
    path = Path(path)
    temporary = path.with_name(path.name+'.part-'+uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path, data):
    atomic(path, json.dumps(data,indent=2,allow_nan=False).encode())


def file_info(path):
    data = Path(path).read_bytes()
    return dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())


def size_upper(text):
    match = re.fullmatch(r'([\d.]+)([KMG]?)',text)
    if not match:
        raise StopAcquisition('Directory listing has an unrecognized product size')
    number, unit = match.groups()
    scale = dict(K=1024,M=1024**2,G=1024**3).get(unit,1)
    # Apache sizes are rounded: conservatively add one displayed precision unit.
    increment = 10**(-len(number.split('.')[1])) if '.' in number else 1
    return math.ceil((float(number)+(increment if unit else 0))*scale)


def parse_index(html):
    groups = {}
    link_occurrences = 0
    for line in html.splitlines():
        names = [unescape(v) for v in re.findall(r'href="([^"]+)"',line)]
        names = [v for v in names if PRODUCT.fullmatch(v)]
        if not names:
            continue
        link_occurrences += len(names)
        tail = line.rsplit('</a>',1)[-1].split()
        if len(tail) < 3:
            raise StopAcquisition('Unexpected directory row structure')
        for name in set(names):
            match = PRODUCT.fullmatch(name)
            stem, seconds, fraction, revision, extension = match.groups()
            if int(fraction)>=65536:
                raise StopAcquisition('Invalid filename spacecraft-clock fractional field')
            item = groups.setdefault(stem,dict(stem=stem,filename_sclk_seconds=int(seconds),
                filename_sclk_subticks=int(fraction),revision=int(revision)))
            item[extension] = name
            item[extension+'_listed_upper_bytes'] = size_upper(tail[2])
    if not groups:
        raise StopAcquisition('No calibrated iof PNG/XML product groups found')
    complete = sorted((v for v in groups.values() if 'png' in v and 'xml' in v),
        key=lambda v:(v['filename_sclk_seconds'],v['filename_sclk_subticks'],v['revision']))
    unpaired = [v for v in groups.values() if 'png' not in v or 'xml' not in v]
    return dict(products=complete,unpaired=unpaired,product_groups=len(groups),
                matching_anchor_occurrences=link_occurrences)


def parse_label(data, product):
    try:
        root = ET.fromstring(data)
    except ET.ParseError as error:
        raise StopAcquisition('Invalid PDS4 XML: '+product['stem']) from error
    def values(name):
        return [e.text.strip() for e in root.iter() if e.tag.rsplit('}',1)[-1]==name and e.text and e.text.strip()]
    # PDS4 observation time, not modification/creation time or filename conversion.
    times = root.findall('.//{*}Observation_Area/{*}Time_Coordinates/{*}start_date_time')
    if len(times) != 1 or not times[0].text:
        raise MissingUTC('Missing/ambiguous PDS4 observation UTC: '+product['stem'])
    start = times[0].text.strip()
    utc(start)
    stops = values('stop_date_time')
    if len(stops) != 1 or utc(stops[0]) < utc(start):
        raise StopAcquisition('Invalid observation stop time: '+product['stem'])
    if product['png'] not in values('file_name'):
        raise StopAcquisition('XML does not reference its matching PNG: '+product['stem'])
    # Flight labels can use acquisition_time as a container whose scalar clock
    # is soc_acquisition_time. Preserve that value without converting it to UTC.
    clock = None
    clock_source = None
    for name in ('acquisition_time','soc_acquisition_time'):
        clocks = [v for v in values(name) if re.fullmatch(r'\d+:\d+',v)]
        if len(clocks)==1:
            clock, clock_source = clocks[0], name
            break
    match = re.fullmatch(r'(\d+):(\d+)',clock or '')
    def scalar(name):
        items=values(name)
        return items[0] if len(items)==1 else None
    def boolean(value):
        return True if str(value).lower()=='true' else False if str(value).lower()=='false' else None
    quality={}
    for element in root.iter():
        name=element.tag.rsplit('}',1)[-1]
        if not list(element) and element.text and re.fullmatch(r'bad_image|.*partial.*|.*incomplete.*|missing_(?:lines|pixels|packets|count).*',name):
            quality[name]=element.text.strip()
    explicit_partial=any(boolean(v) is True for k,v in quality.items() if 'partial' in k or 'incomplete' in k)
    is_partial=explicit_partial or product['stem']==LAST_PARTIAL
    if product['stem']==LAST_PARTIAL:
        completeness='PARTIAL_CURATED'
        basis='Final chronological browse has a truncated strip; NASA documents final image interrupted by impact. XML bad_image is not a completeness flag.'
    elif product['stem']==LAST_COMPLETE:
        completeness='FINAL_COMPLETE_CURATED'
        basis='Penultimate chronological browse; visual full-window content, consistent with NASA last-complete image about two seconds before impact.'
    else:
        completeness='PARTIAL_BY_XML' if explicit_partial else 'NO_PARTIAL_FLAG'
        basis='PDS per-image flags; absence of a partial flag is not a guarantee of completeness.'
    return dict(acquisition_utc=start,stop_utc=stops[0],label_sclk=clock,label_sclk_source=clock_source,
        label_sclk_seconds=int(match[1]) if match else None,
        label_sclk_subticks=int(match[2]) if match else None,
        selection_basis='PDS4 Observation_Area/Time_Coordinates/start_date_time',
        label_mission_phase=scalar('mission_phase'),pds_logical_identifier=scalar('logical_identifier'),
        pds_product_version=(root.findtext('./{*}Identification_Area/{*}version_id')),
        pds_bad_image=boolean(scalar('bad_image')),pds_quality_metadata=quality,
        pds_partial_flag_present=any('partial' in k or 'incomplete' in k for k in quality),
        is_partial=is_partial,image_completeness=completeness,completeness_basis=basis,
        completeness_reference=NASA_FINAL if product['stem'] in (LAST_PARTIAL,LAST_COMPLETE) else None,
        window2={key:scalar('window2_'+key) for key in ('x_start','y_start','x_end','y_end')})


class Client:
    def __init__(self, out, interval=.2):
        self.out, self.interval = Path(out), interval
        self.lock = threading.Lock()
        self.next_request = 0.
        self.stop = threading.Event()
        self.recent_errors = deque()
        self.requests = self.retries = self.received_bytes = 0
        self.events = []

    def get(self, url, limit):
        if url != SOURCE:
            if not url.startswith(SOURCE) or not PRODUCT.fullmatch(url[len(SOURCE):]):
                raise StopAcquisition('Refusing non-PNG/XML archive URL')
        for attempt in range(3):
            if self.stop.is_set():
                raise StopAcquisition('Acquisition stopped')
            with self.lock:
                delay=max(0,self.next_request-time.monotonic())
                self.next_request=max(self.next_request,time.monotonic())+self.interval
            if delay:
                time.sleep(delay)
            request=urllib.request.Request(url,headers={'User-Agent':USER_AGENT})
            event=dict(utc=now(),url=url,attempt=attempt+1)
            try:
                with self.lock:
                    self.requests += 1
                with urllib.request.urlopen(request,timeout=35) as response:
                    if response.status != 200 or response.geturl() != url:
                        raise StopAcquisition('Unexpected HTTP status or redirected archive URL')
                    expected=response.headers.get('Content-Length')
                    if expected and int(expected)>limit:
                        raise StopAcquisition('HTTP object exceeds the bounded file-size limit')
                    data=response.read(limit+1)
                    if len(data)>limit:
                        raise StopAcquisition('Oversized HTTP object')
                    if not data or (expected and len(data)!=int(expected)):
                        raise OSError('Empty or incomplete HTTP object')
                    event.update(status=200,bytes=len(data))
                    with self.lock:
                        self.received_bytes += len(data)
                        self.events.append(event)
                    return data
            except (urllib.error.URLError,TimeoutError,ConnectionError,OSError,http.client.HTTPException) as error:
                code=getattr(error,'code',None)
                retryable=code in (429,500,502,503,504) or code is None
                event.update(status=code,error=str(error),retryable=retryable)
                with self.lock:
                    self.events.append(event)
                    if code in (429,500,502,503,504):
                        current=time.monotonic()
                        self.recent_errors.append(current)
                        while self.recent_errors and current-self.recent_errors[0]>60:
                            self.recent_errors.popleft()
                        if len(self.recent_errors)>=3:
                            self.stop.set()
                if self.stop.is_set() or not retryable or attempt==2:
                    self.stop.set()
                    raise StopAcquisition('HTTP acquisition stopped: '+str(error)) from error
                with self.lock:
                    self.retries += 1
                retry_after=getattr(error,'headers',{}).get('Retry-After','0')
                pause=max(2**(attempt+1),int(retry_after) if str(retry_after).isdigit() else 0)
                if pause>60:
                    self.stop.set()
                    raise StopAcquisition('Server requested a long backoff; stopped')
                time.sleep(pause)
        raise StopAcquisition('Retries exhausted')

    def statistics(self):
        return dict(requests=self.requests,retries=self.retries,received_bytes=self.received_bytes,
                    failures=[e for e in self.events if 'error' in e])


def prepare(out):
    out=Path(out).resolve()
    marker=out/'acquisition.json'
    if out.name=='269' or (out.exists() and any(out.iterdir()) and not marker.exists()):
        raise StopAcquisition('Refusing existing unmanaged data directory')
    out.mkdir(parents=True,exist_ok=True)
    identity=dict(source_url=SOURCE,start_utc=START,end_utc_exclusive=END)
    if marker.exists():
        if json.loads(marker.read_text())['scope']!=identity:
            raise StopAcquisition('Existing acquisition scope differs')
    else:
        save_json(marker,dict(scope=identity,created_utc=now()))
    (out/'products').mkdir(exist_ok=True)
    return out


def plan(out, client, maximum):
    html=client.get(SOURCE,16*1024**2).decode('iso-8859-1')
    index=parse_index(html)
    products=index['products']
    cache={}
    probes=[]
    def label(i):
        if i not in cache:
            item=products[i]
            local=out/'products'/item['xml']
            data=local.read_bytes() if local.is_file() else client.get(SOURCE+item['xml'],512*1024)
            parsed=parse_label(data,item)
            cache[i]=(data,parsed)
            probes.append(dict(stem=item['stem'],**parsed))
        return utc(cache[i][1]['acquisition_utc'])
    def lower_bound(target):
        lo,hi=0,len(products)
        while lo<hi:
            mid=(lo+hi)//2
            if label(mid)<target: lo=mid+1
            else: hi=mid
        return lo
    method='XML_UTC_binary_boundaries_then_individual_XML_selection'
    seeds=[]
    try:
        first,end=(0,len(products)) if COLLECTION=='final' else (lower_bound(utc(START)),lower_bound(utc(END)))
        if first==len(products) or end==0 or end<=first:
            raise StopAcquisition('Requested UTC interval has no archive products')
        if COLLECTION=='terminal':
            if first and label(first-1)>=utc(START): raise StopAcquisition('Invalid lower UTC boundary')
            if label(first)<utc(START) or label(end-1)>=utc(END): raise StopAcquisition('Invalid UTC boundaries')
            if end<len(products) and label(end)<utc(END): raise StopAcquisition('Invalid upper UTC boundary')
        else:
            method='ENTIRE_FINAL_COLLECTION_individual_XML_UTC_ordering'
            label(0); label(len(products)-1)
        # A few neighboring products are checked individually, never retained outside UTC bounds.
        candidates=products[max(0,first-2):min(len(products),end+2)]
        for target in TARGETS:
            position=lower_bound(utc(target))
            choices=[i for i in (position-1,position) if first<=i<end]
            nearest=min(choices,key=lambda i:abs((label(i)-utc(target)).total_seconds()))
            seeds.append(products[nearest]['stem'])
        if COLLECTION=='final':
            for stem in (LAST_COMPLETE,LAST_PARTIAL):
                positions=[i for i,item in enumerate(products) if item['stem']==stem]
                if positions:
                    label(positions[0]); seeds.append(stem)
    except MissingUTC:
        if COLLECTION=='final': raise
        method='BOUNDED_SCLK_FALLBACK_FOR_CANDIDATE_INDEX_ONLY'
        candidates=[v for v in products if 401925585<=v['filename_sclk_seconds']<401929790]
        # Individual retained products still require reliable XML UTC; otherwise STOP.
    unpaired=index['unpaired'] if COLLECTION=='final' else [v for v in index['unpaired'] if 401925585<=v['filename_sclk_seconds']<401929790]
    if unpaired: raise StopAcquisition('Unpaired archive products near the requested interval')
    estimate=sum(v['png_listed_upper_bytes']+v['xml_listed_upper_bytes'] for v in candidates)
    if estimate>maximum:
        raise StopAcquisition(f'Requested PNG/XML interval estimate {estimate:,} bytes exceeds bounded budget {maximum:,}')
    result=dict(source_url=SOURCE,start_utc=START,end_utc_exclusive=END,created_utc=now(),
                archive_product_groups=index['product_groups'],archive_pairs=len(products),
                archive_unpaired=index['unpaired'],candidate_count=len(candidates),estimated_upper_bytes=estimate,
                maximum_bytes=maximum,index_method=method,utc_probes=probes,candidates=candidates,
                representative_seed_stems=seeds,
                listing_sha256=hashlib.sha256(html.encode('iso-8859-1')).hexdigest(),http=client.statistics())
    save_json(out/'plan.json',result)
    return result, {products[i]['stem']:value for i,value in cache.items()}


def acquire(out, client, planned, cache, workers, maximum):
    products_dir=out/'products'
    previous={}
    manifest_path=out/'manifest.json'
    if manifest_path.exists(): previous={r['product_stem']:r for r in json.loads(manifest_path.read_text())}
    rows={}
    excluded=[]
    errors=[]
    budget_lock=threading.Lock()
    total=sum(p.stat().st_size for p in products_dir.iterdir() if p.suffix in ('.png','.xml'))
    def store(path, payload):
        nonlocal total
        with budget_lock:
            replacement=path.stat().st_size if path.exists() else 0
            if total+len(payload)-replacement>maximum:
                client.stop.set()
                raise StopAcquisition('Actual product bytes exceed the dataset budget')
            atomic(path,payload)
            total += len(payload)-replacement
    def one(item):
        png,xml=products_dir/item['png'],products_dir/item['xml']
        old=previous.get(item['stem'])
        xml_data=xml.read_bytes() if xml.is_file() and xml.stat().st_size else None
        if old and xml_data is not None and hashlib.sha256(xml_data).hexdigest()!=old['xml_sha256']:
            xml_data=None
        if xml_data is None:
            xml_data=cache[item['stem']][0] if item['stem'] in cache else client.get(SOURCE+item['xml'],512*1024)
        label=parse_label(xml_data,item)
        if label['label_mission_phase'] and label['label_mission_phase']!=COLLECTION:
            raise StopAcquisition('XML mission phase disagrees with selected source directory')
        if not in_scope(label['acquisition_utc']):
            return None,dict(stem=item['stem'],**label)
        existing_png=png.read_bytes() if png.is_file() else b''
        png_complete=existing_png.startswith(PNG_SIGNATURE) and existing_png.endswith(PNG_END)
        if old and png_complete:
            png_complete=hashlib.sha256(existing_png).hexdigest()==old['png_sha256']
        skipped=bool(png_complete and xml.is_file() and xml_data==xml.read_bytes())
        if not skipped:
            store(xml,xml_data)
            if png_complete:
                png_data=existing_png
            else:
                png_data=client.get(SOURCE+item['png'],16*1024**2)
                if not png_data.startswith(PNG_SIGNATURE) or not png_data.endswith(PNG_END):
                    raise StopAcquisition('HTTP payload is not PNG: '+item['stem'])
                store(png,png_data)
        row=dict(product_stem=item['stem'],png_filename=item['png'],xml_filename=item['xml'],
                 png_url=SOURCE+item['png'],xml_url=SOURCE+item['xml'],**label,
                 filename_sclk_seconds=item['filename_sclk_seconds'],filename_sclk_subticks=item['filename_sclk_subticks'],
                 png_path=str(png.resolve()),xml_path=str(xml.resolve()),
                 png_relative_path=png.relative_to(out).as_posix(),xml_relative_path=xml.relative_to(out).as_posix(),
                 png_bytes=png.stat().st_size,xml_bytes=xml.stat().st_size,
                 png_sha256=file_info(png)['sha256'],xml_sha256=file_info(xml)['sha256'],
                 download_status='SKIPPED_COMPLETE' if skipped else 'DOWNLOADED',validation_status='PENDING')
        return row,None
    pool=ThreadPoolExecutor(max_workers=workers)
    seeds=set(planned.get('representative_seed_stems',[]))
    scheduled=sorted(planned['candidates'],key=lambda p:p['stem'] not in seeds)
    futures={pool.submit(one,item):item for item in scheduled}
    try:
        for number,future in enumerate(as_completed(futures),1):
            item=futures[future]
            try:
                row,other=future.result()
                if row: rows[row['product_stem']]=row
                if other: excluded.append(other)
            except Exception as error:
                errors.append(dict(stem=item['stem'],error=str(error)))
                client.stop.set()
                for pending in futures: pending.cancel()
                break
            if number%100==0 or number==len(futures):
                save_json(manifest_path,sorted(rows.values(),key=lambda r:(utc(r['acquisition_utc']),r['product_stem'])))
                print(json.dumps(dict(phase='download',processed=number,total=len(futures),pairs=len(rows),
                                     product_mib=round(total/1024**2,2),http=client.statistics())),flush=True)
    finally:
        pool.shutdown(wait=True,cancel_futures=True)
        ordered=sorted(rows.values(),key=lambda r:(utc(r['acquisition_utc']),r['product_stem']))
        save_json(manifest_path,ordered)
        save_json(out/'download_report.json',dict(completed_utc=now(),complete=not errors,
                  pairs=len(rows),excluded_boundary_labels=excluded,errors=errors,http=client.statistics()))
    if errors:
        raise StopAcquisition(errors[0]['error'])
    return ordered


def final_contact_sheet(out, rows, cv2, np):
    """Full native-frame views at a common scale, with no detection or recentering."""
    choices=[]
    for target in TARGETS:
        row=min(rows,key=lambda r:abs((utc(r['acquisition_utc'])-utc(target)).total_seconds()))
        choices.append((target,row))
    for stem in (LAST_COMPLETE,LAST_PARTIAL):
        row=next((r for r in rows if r['product_stem']==stem),None)
        if row: choices.append((None,row))
    if len(choices)!=8:
        raise StopAcquisition('Expected final complete/partial browse products are absent; inspect archive')
    panels=[]; representatives=[]
    for target,row in choices:
        image=cv2.imread(str(out/row['png_relative_path']),cv2.IMREAD_UNCHANGED)
        if image.ndim==2: display=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        elif image.shape[2]==4: display=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        else: display=image.copy()
        if display.dtype!=np.uint8:
            display=cv2.normalize(display,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        seconds=(utc(IMPACT_UTC)-utc(row['acquisition_utc'])).total_seconds()
        label='PARTIAL: curated, XML bad_image=false' if row['is_partial'] else (
              'Final complete browse' if row['product_stem']==LAST_COMPLETE else 'Full PNG at common scale')
        panel=np.full((530,450,3),22,np.uint8)
        color=(110,150,255) if row['is_partial'] else (230,230,230)
        cv2.putText(panel,row['acquisition_utc'][11:23]+' UTC',(12,26),cv2.FONT_HERSHEY_SIMPLEX,.65,color,1,cv2.LINE_AA)
        cv2.putText(panel,f'T - {seconds:.3f} s (approx. impact reference)',(12,51),cv2.FONT_HERSHEY_SIMPLEX,.45,(190,190,190),1,cv2.LINE_AA)
        panel[70:500,10:440]=cv2.resize(display,(430,430),interpolation=cv2.INTER_AREA)
        cv2.putText(panel,label,(12,522),cv2.FONT_HERSHEY_SIMPLEX,.44,color,1,cv2.LINE_AA)
        panels.append(panel)
        representatives.append(dict(requested_utc=target,acquisition_utc=row['acquisition_utc'],
             product_stem=row['product_stem'],png_relative_path=row['png_relative_path'],
             approximate_seconds_before_impact=seconds,is_partial=row['is_partial'],
             image_completeness=row['image_completeness'],display='Entire PNG; identical native-pixel scale across panels'))
    canvas=np.full((1160,1800,3),22,np.uint8)
    cv2.putText(canvas,'REAL DRACO FINAL PHASE | 2022-09-26 UTC | calibrated collection v3.0',(16,30),cv2.FONT_HERSHEY_SIMPLEX,.75,(235,235,235),1,cv2.LINE_AA)
    cv2.putText(canvas,'Dimorphos grows from the small companion to filling the active image window. Original frames are preserved.',(16,59),cv2.FONT_HERSHEY_SIMPLEX,.52,(195,195,195),1,cv2.LINE_AA)
    cv2.putText(canvas,'Recorded imagery: perception/replay only; commands cannot alter these frames. Partial final transmission is retained and flagged.',(16,84),cv2.FONT_HERSHEY_SIMPLEX,.48,(160,210,230),1,cv2.LINE_AA)
    for i,panel in enumerate(panels):
        y=100+(i//4)*530; x=(i%4)*450
        canvas[y:y+530,x:x+450]=panel
    ok,encoded=cv2.imencode('.png',canvas)
    if not ok: raise StopAcquisition('Final contact sheet encoding failed')
    atomic(out/'contact_sheet.png',encoded.tobytes())
    save_json(out/'representatives.json',representatives)


def validate(out):
    import cv2
    import numpy as np
    if cv2.__version__.split('.')[0]!='5': raise StopAcquisition('Use the existing OpenCV 5 validation environment')
    rows=sorted(json.loads((out/'manifest.json').read_text()),key=lambda r:(utc(r['acquisition_utc']),r['product_stem']))
    if not rows: raise StopAcquisition('No acquired products to validate')
    actual_png={p.name for p in (out/'products').glob('*.png')}
    actual_xml={p.name for p in (out/'products').glob('*.xml')}
    shapes=Counter(); dtypes=Counter(); channels=Counter(); errors=[]
    for row in rows:
        png=out/row['png_relative_path']; xml=out/row['xml_relative_path']
        image=cv2.imread(str(png),cv2.IMREAD_UNCHANGED)
        if image is None or not xml.is_file():
            row['validation_status']='INVALID'
            errors.append(row['product_stem']+': image decode failed or label missing'); continue
        parsed=parse_label(xml.read_bytes(),dict(stem=row['product_stem'],png=row['png_filename']))
        if parsed['acquisition_utc']!=row['acquisition_utc'] or not in_scope(parsed['acquisition_utc']):
            errors.append(row['product_stem']+': UTC selection mismatch')
        if file_info(png)['sha256']!=row['png_sha256'] or file_info(xml)['sha256']!=row['xml_sha256']:
            errors.append(row['product_stem']+': hash mismatch')
        shape='x'.join(map(str,image.shape)); channel=1 if image.ndim==2 else image.shape[2]
        shapes[shape]+=1; dtypes[str(image.dtype)]+=1; channels[str(channel)]+=1
        row.update(parsed,image_shape=list(image.shape),image_dtype=str(image.dtype),image_channels=channel,validation_status='VALID')
    times=[utc(r['acquisition_utc']) for r in rows]
    duplicates=[k.isoformat() for k,v in Counter(utc(r['acquisition_utc']) for r in rows).items() if v>1]
    duplicate_products=[k for k,v in Counter(r['product_stem'] for r in rows).items() if v>1]
    filename_order=sorted(rows,key=lambda r:(r['filename_sclk_seconds'],r['filename_sclk_subticks'],r['product_stem']))
    monotonic=all(utc(a['acquisition_utc'])<=utc(b['acquisition_utc']) for a,b in zip(filename_order,filename_order[1:]))
    unpaired=sorted((actual_png-{r['png_filename'] for r in rows}) | (actual_xml-{r['xml_filename'] for r in rows}))
    missing=sorted(({r['png_filename'] for r in rows}-actual_png) | ({r['xml_filename'] for r in rows}-actual_xml))
    gaps=[(b-a).total_seconds() for a,b in zip(times,times[1:])]
    representatives=[]; panels=[]
    for target,centre in ([] if COLLECTION=='final' else zip(TARGETS,PREVIEW_CENTRES)):
        row=min(rows,key=lambda r:abs((utc(r['acquisition_utc'])-utc(target)).total_seconds()))
        image=cv2.imread(str(out/row['png_relative_path']),cv2.IMREAD_UNCHANGED)
        if image is None: continue
        # Display-only layout: full PNG and a manually specified preview ROI.
        if image.ndim==2: display=cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        elif image.shape[2]==4: display=cv2.cvtColor(image,cv2.COLOR_BGRA2BGR)
        else: display=image.copy()
        if display.dtype!=np.uint8:
            display=cv2.normalize(display,None,0,255,cv2.NORM_MINMAX).astype(np.uint8)
        height,width=display.shape[:2]; cx,cy=centre
        x0=min(max(0,cx-128),max(0,width-256)); y0=min(max(0,cy-128),max(0,height-256))
        crop=display[y0:min(height,y0+256),x0:min(width,x0+256)].copy()
        overview=display.copy()
        cv2.rectangle(overview,(x0,y0),(min(width-1,x0+255),min(height-1,y0+255)),(120,200,240),1)
        panel=np.full((920,420,3),22,np.uint8)
        cv2.putText(panel,row['acquisition_utc'][11:23]+' UTC',(10,26),cv2.FONT_HERSHEY_SIMPLEX,.61,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(panel,'Full PNG; box = manual preview ROI',(10,52),cv2.FONT_HERSHEY_SIMPLEX,.48,(190,190,190),1,cv2.LINE_AA)
        panel[70:470,10:410]=cv2.resize(overview,(400,400),interpolation=cv2.INTER_AREA)
        cv2.putText(panel,'Manual 256 px ROI; nearest-neighbor',(10,499),cv2.FONT_HERSHEY_SIMPLEX,.47,(190,190,190),1,cv2.LINE_AA)
        panel[515:915,10:410]=cv2.resize(crop,(400,400),interpolation=cv2.INTER_NEAREST)
        panels.append(panel)
        representatives.append(dict(requested_utc=target,acquisition_utc=row['acquisition_utc'],
            offset_s=(utc(row['acquisition_utc'])-utc(target)).total_seconds(),product_stem=row['product_stem'],
            png_relative_path=row['png_relative_path'],preview_roi_xywh=[x0,y0,crop.shape[1],crop.shape[0]],
            preview_roi_method='Manual display-only crop after visual inspection; no detector'))
    if len(panels)==5:
        sheet=np.vstack((np.full((74,2100,3),22,np.uint8),np.hstack(panels)))
        cv2.putText(sheet,'REAL DRACO PNG PREVIEWS | calibrated collection v3.0 | XML UTC selection',(18,27),cv2.FONT_HERSHEY_SIMPLEX,.75,(235,235,235),1,cv2.LINE_AA)
        cv2.putText(sheet,'Prerecorded imagery: perception/replay evidence only; not a causal closed-loop environment.',(18,57),cv2.FONT_HERSHEY_SIMPLEX,.6,(160,210,230),1,cv2.LINE_AA)
        ok,encoded=cv2.imencode('.png',sheet)
        if not ok: raise StopAcquisition('Contact sheet encoding failed')
        atomic(out/'contact_sheet.png',encoded.tobytes())
    save_json(out/'manifest.json',rows)
    csv_rows=[{k:(json.dumps(v) if isinstance(v,(list,dict)) else v) for k,v in row.items()} for row in rows]
    stream=io.StringIO(newline=''); writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in csv_rows for k in r)))
    writer.writeheader(); writer.writerows(csv_rows); atomic(out/'manifest.csv',stream.getvalue().encode())
    plan_data=json.loads((out/'plan.json').read_text()); download=json.loads((out/'download_report.json').read_text())
    summary=dict(source_url=SOURCE,start_utc=START,end_utc_exclusive=END,validated_utc=now(),opencv_version=cv2.__version__,
        complete=not(errors or missing or unpaired or duplicate_products or not monotonic),products=len(rows),pairs=len(rows),
        first_product=rows[0]['product_stem'],last_product=rows[-1]['product_stem'],
        first_acquisition_utc=rows[0]['acquisition_utc'],last_acquisition_utc=rows[-1]['acquisition_utc'],
        total_product_bytes=sum(r['png_bytes']+r['xml_bytes'] for r in rows),
        png_bytes=sum(r['png_bytes'] for r in rows),xml_bytes=sum(r['xml_bytes'] for r in rows),
        actual_png_files=len(actual_png),actual_xml_files=len(actual_xml),missing=missing,unpaired=unpaired,
        validation_errors=errors,duplicate_timestamps=duplicates,duplicate_products=duplicate_products,
        utc_nondecreasing_in_sclk_order=monotonic,image_shapes=dict(shapes),image_dtypes=dict(dtypes),image_channels=dict(channels),
        cadence_s=dict(min=min(gaps),median=statistics.median(gaps),max=max(gaps)),
        gaps_over_5s=[dict(before=rows[i]['acquisition_utc'],after=rows[i+1]['acquisition_utc'],seconds=g) for i,g in enumerate(gaps) if g>5],
        representatives=representatives,index_method=plan_data['index_method'],
        utility_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        sclk_fallback_used='FALLBACK' in plan_data['index_method'],plan_http=plan_data['http'],download_http=download['http'],
        preflight_diagnostics=(json.loads((out/'preflight_diagnostics.json').read_text())
                               if (out/'preflight_diagnostics.json').exists() else {}),
        scientific_boundary='Prerecorded real PNG browse imagery; not a causal closed-loop environment. PNG pixels are previews, not the FITS numerical I/F arrays.')
    summary.update(mission_phase=COLLECTION,scope='entire_collection' if COLLECTION=='final' else 'bounded_UTC_interval',
                   pds_bad_image_counts=dict(Counter(str(r['pds_bad_image']) for r in rows)),
                   partial_products=[r['product_stem'] for r in rows if r['is_partial']],
                   explicit_pds_partial_flags=sum(r['pds_partial_flag_present'] for r in rows),
                   duplicate_png_hash_groups=sum(n>1 for n in Counter(r['png_sha256'] for r in rows).values()))
    if COLLECTION=='final':
        final_contact_sheet(out,rows,cv2,np)
        summary['representatives']=json.loads((out/'representatives.json').read_text())
        summary['final_complete_product']=next((r['product_stem'] for r in reversed(rows) if r['image_completeness']=='FINAL_COMPLETE_CURATED'),None)
        summary['final_partial_product']=next((r['product_stem'] for r in reversed(rows) if r['is_partial']),None)
    save_json(out/'summary.json',summary)
    if not summary['complete']: raise StopAcquisition('Dataset validation failed; see summary.json')
    print(json.dumps(summary,indent=2),flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['plan','download','validate'],default='download')
    parser.add_argument('--collection',choices=['terminal','final'],default='terminal')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-mib',type=int,default=300)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or not 1<=args.max_mib<=300: parser.error('workers 1–8; budget 1–300 MiB')
    configure(args.collection)
    out=prepare(args.out or DEFAULT_OUT); client=Client(out); maximum=args.max_mib*1024**2
    try:
        if args.phase=='validate': validate(out); return
        if args.phase=='plan' or not (out/'plan.json').exists():
            planned,cache=plan(out,client,maximum)
        else:
            planned=json.loads((out/'plan.json').read_text()); cache={}
            if planned['source_url']!=SOURCE or planned['start_utc']!=START or planned['end_utc_exclusive']!=END:
                raise StopAcquisition('Cached plan scope mismatch')
            if planned['estimated_upper_bytes']>maximum: raise StopAcquisition('Cached plan exceeds the byte budget')
        if args.phase=='plan':
            print(json.dumps({k:v for k,v in planned.items() if k not in ('candidates','utc_probes')},indent=2)); return
        acquire(out,client,planned,cache,args.workers,maximum)
        validate(out)
    except Exception as error:
        save_json(out/'STOPPED.json',dict(utc=now(),reason=str(error),http=client.statistics()))
        raise


if __name__=='__main__':
    main()
