"""Separate causal DRACO recorded-image perception. Never imports navigation.

Frozen image measurements are reused, not modified. New image-only persistence,
acquisition, association and multi-component visible-region handling live here.
No human annotations are read by the online classes; optional designation is one
explicit initialization record only. No future observation is consulted.
"""
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from opendart.vision import AsteroidDetector,require_opencv


def load_config(path=None):
    return json.loads(Path(path or Path(__file__).with_name('draco_perception_config_v1.json')).read_text())


def center(candidate):return np.asarray(candidate['centroid_px'],float)


def box_union(boxes):
    a=np.asarray(boxes,float);left=a[:,:2].min(axis=0);right=(a[:,:2]+a[:,2:]).max(axis=0)
    return [*left.tolist(),*(right-left).tolist()]


def in_box(point,box):
    x,y,w,h=box
    return x<=point[0]<=x+w and y<=point[1]<=y+h


class Features:
    """Current image plus past-only component persistence and translation cue."""
    def __init__(self,config):
        self.config=config;self.detector=AsteroidDetector();self.past=np.zeros((1024,1024),np.float32)
        self.frames=0;self.previous=None;self.last_time=None
        cv=require_opencv();n=config['phase_size'];self.window=cv.createHanningWindow((n,n),cv.CV_32F)

    def update(self,image,timestamp_s):
        if self.last_time is not None and timestamp_s<=self.last_time:raise ValueError('Strict chronological input required')
        self.last_time=timestamp_s
        if image.shape!=(1024,1024) or image.dtype!=np.uint8:raise ValueError('Original 1024x1024 uint8 grayscale required')
        start=time.perf_counter();cv=require_opencv();cfg=self.config
        candidates=[c.to_dict() for c in self.detector.detect(image)]
        occupancy=np.zeros_like(self.past);kept=[];rejected=[]
        for i,c in enumerate(candidates):
            x,y,w,h=map(int,c['bbox_xywh']);prob=float(np.mean(self.past[y:y+h,x:x+w]>=cfg['persistence_probability']))
            # Read prior occupancy BEFORE updating with this frame. Large bodies
            # are never rejected merely for staying at the same image location.
            fixed=(self.frames>=cfg['persistence_warmup_frames'] and c['area_px2']<=cfg['persistent_max_area_px2']
                   and prob>=cfg['persistent_fraction'])
            value=dict(c,feature_index=i,prior_persistent_fraction=prob)
            (rejected if fixed else kept).append(value)
            occupancy[y:y+h,x:x+w]=1
        self.past*=(1-cfg['persistence_alpha']);self.past+=cfg['persistence_alpha']*occupancy
        n=cfg['phase_size'];small=cv.resize(self.detector.last_normalized,(n,n),interpolation=cv.INTER_AREA).astype(np.float32)
        shift=[0.,0.];response=0.
        if self.previous is not None:
            delta,response=cv.phaseCorrelate(self.previous.copy(),small.copy(),self.window)
            delta=np.asarray(delta)*(1024/n)
            if np.isfinite(response) and np.all(np.isfinite(delta)) and response>=cfg['phase_min_response'] and np.linalg.norm(delta)<=cfg['phase_max_shift_px']:
                shift=delta.tolist()
        self.previous=small;self.frames+=1
        return dict(candidates=kept,rejected_persistent=rejected,raw_candidate_count=len(candidates),
            image_shift_px=shift,phase_response=float(response) if np.isfinite(response) else 0.,
            feature_wall_s=time.perf_counter()-start)


class Track:
    """Image-space hypothesis; semantic identity is always explicitly qualified."""
    def __init__(self,config,mode='autonomous',designation=None):
        if mode not in ('autonomous','human_designated'):raise ValueError('Unknown mode')
        if mode=='human_designated' and designation is None:raise ValueError('One designation required')
        self.config=config;self.mode=mode;self.designation=designation;self.used_designation=False
        self.position=None;self.area=None;self.box=None;self.missed=0;self.state='SEARCH_UNRESOLVED'
        self.identity='UNESTABLISHED';self.pending=None;self.pending_count=0;self.ever=False
        self.resolved_count=0;self.resolved=False;self.last_time=None

    def pair(self,candidates):
        cfg=self.config;ordered=sorted(candidates,key=lambda c:c['area_px2'],reverse=True)
        if len(ordered)<2 or ordered[0]['area_px2']<cfg['acquire_min_primary_area_px2']:return None
        primary=ordered[0]
        choices=[c for c in ordered[1:] if c['area_px2']>=cfg['acquire_min_secondary_area_px2']
            and primary['area_px2']>=cfg['primary_secondary_area_ratio']*c['area_px2']
            and np.linalg.norm(center(c)-center(primary))>c['radius_px']+primary['radius_px']+3]
        if not choices:return None
        if len(choices)>1 and choices[0]['area_px2']<cfg['secondary_uniqueness_area_ratio']*choices[1]['area_px2']:return None
        return choices[0],center(choices[0])-center(primary)

    def acquire(self,candidates):
        pair=self.pair(candidates)
        if pair is None:self.pending=None;self.pending_count=0;return None
        candidate,vector=pair
        if self.pending is not None and np.linalg.norm(vector-self.pending)<=self.config['pair_vector_gate_px']:
            self.pending_count+=1
        else:self.pending_count=1
        self.pending=vector
        return candidate if self.pending_count>=self.config['acquire_confirm_frames'] else None

    def envelope(self,candidates,seed):
        cfg=self.config;box=seed['bbox_xywh']
        # Once resolved, group foreground fragments inside the propagated target
        # neighborhood. The representation no longer requires one component.
        prior=self.box if self.box is not None else box
        extent=max(*prior[2:],*box[2:],cfg['resolved_enter_extent_px'])
        half=.5*extent*cfg['resolved_roi_expansion']
        roi=[self.position[0]-half,self.position[1]-half,2*half,2*half]
        threshold=max(6.,seed['area_px2']*cfg['resolved_fragment_min_fraction'])
        pieces=[c for c in candidates if c['area_px2']>=threshold and in_box(center(c),roi)]
        if seed not in pieces:pieces.append(seed)
        return box_union([c['bbox_xywh'] for c in pieces]),len(pieces)

    def update(self,observation,timestamp_s,frame_index,is_partial=False):
        if self.last_time is not None and timestamp_s<=self.last_time:raise ValueError('Strict chronological input required')
        self.last_time=timestamp_s;start=time.perf_counter();cfg=self.config;candidates=observation['candidates']
        event=[];selected=None;box=None;pieces=0;representation=None
        if is_partial:
            self.state='PARTIAL_FRAME'
            substantial=[c['bbox_xywh'] for c in candidates if c['area_px2']>=cfg['acquire_min_secondary_area_px2']]
            box=box_union(substantial) if substantial else None
            return self.output(None,box,len(substantial),'TRANSMITTED_VISIBLE_REGION_ONLY',['PARTIAL_FRAME'],start)
        if self.position is not None:self.position+=np.asarray(observation['image_shift_px'])
        if self.mode=='human_designated' and not self.used_designation:
            if frame_index==self.designation['frame_index']:
                target_box=self.designation['visible_box_xywh'];uv=np.asarray(self.designation['center_uv'])
                choices=[c for c in candidates if in_box(center(c),target_box)]
                self.used_designation=True
                if choices:
                    selected=min(choices,key=lambda c:np.linalg.norm(center(c)-uv));event.append('HUMAN_DESIGNATION')
                    self.identity='HUMAN_DESIGNATED_LINK'
                else:self.state='LOST';event.append('HUMAN_DESIGNATION_FAILED_NO_COMPONENT')
            else:self.state='AWAITING_HUMAN_DESIGNATION'
        elif self.resolved and self.position is not None and self.missed<=cfg['reset_after_missed_frames']:
            # A resolved body can fragment into many components, each far
            # smaller than the previous component. Accept collective foreground
            # support, not a mandatory compact seed or a single-component gate.
            extent=max(self.box[2:]);half=.5*extent*cfg['resolved_roi_expansion']
            roi=[self.position[0]-half,self.position[1]-half,2*half,2*half]
            fragments=[c for c in candidates if in_box(center(c),roi)
                and c['area_px2']>=max(6.,self.area*cfg['resolved_fragment_min_fraction'])]
            area=sum(c['area_px2'] for c in fragments)
            if fragments and 1/cfg['max_area_step_ratio']<=area/self.area<=cfg['max_area_step_ratio']:
                box=box_union([c['bbox_xywh'] for c in fragments]);self.box=box;self.area=area
                self.position=np.asarray(box[:2])+np.asarray(box[2:])/2
                if self.missed:event.append('REACQUIRED')
                self.missed=0;self.state='RESOLVED_BODY'
                return self.output(None,box,len(fragments),'VISIBLE_FOREGROUND_ENVELOPE',event,start)
        elif self.position is None or self.missed>cfg['reset_after_missed_frames']:
            selected=self.acquire(candidates)
            self.state='REACQUIRING' if self.ever else 'SEARCH_UNRESOLVED'
            if selected is not None:
                if self.ever:
                    self.resolved=False;self.resolved_count=0
                self.identity='SECONDARY_IMAGE_HYPOTHESIS_IDENTITY_UNVERIFIED'
                event.append('REACQUIRED' if self.ever else 'AUTONOMOUS_ACQUIRED')
                if self.ever:event.append('STALE_GATE_RESET_IDENTITY_LINK_UNCERTAIN')
        else:
            gate=min(cfg['maximum_gate_px'],max(cfg['position_gate_px'],cfg['radius_gate_multiplier']*np.sqrt(self.area/np.pi))+self.missed*cfg['miss_gate_growth_px'])
            choices=[c for c in candidates if 1/cfg['max_area_step_ratio']<=c['area_px2']/self.area<=cfg['max_area_step_ratio']
                and np.linalg.norm(center(c)-self.position)<=gate]
            if choices:
                selected=min(choices,key=lambda c:np.linalg.norm(center(c)-self.position)/gate+.15*abs(np.log(c['area_px2']/self.area)))
                if self.missed:event.append('REACQUIRED')
        if selected is not None:
            self.position=center(selected).copy();self.area=selected['area_px2'];self.ever=True;self.missed=0
            self.resolved_count=self.resolved_count+1 if max(selected['bbox_xywh'][2:])>=cfg['resolved_enter_extent_px'] else 0
            if not self.resolved and self.resolved_count>=cfg['resolved_confirm_frames']:
                self.resolved=True;event.append('ENTERED_RESOLVED_BODY')
            if self.resolved:
                box,pieces=self.envelope(candidates,selected);representation='VISIBLE_FOREGROUND_ENVELOPE';self.state='RESOLVED_BODY'
            else:
                box=list(selected['bbox_xywh']);pieces=1;representation='COMPACT_COMPONENT';self.state='TRACKING'
            if 'HUMAN_DESIGNATION' in event:self.state='TARGET_DESIGNATED'
            elif 'REACQUIRED' in event and not self.resolved:self.state='REACQUIRED'
            self.box=box
        elif self.ever:
            self.missed+=1
            if self.missed==1:event.append('LOST')
            self.state='REACQUIRING' if self.missed>cfg['reset_after_missed_frames'] else 'LOST'
        return self.output(selected,box,pieces,representation,event,start)

    def output(self,selected,box,pieces,representation,event,start):
        return dict(mode=self.mode,state=self.state,identity_status=self.identity,
            selected_feature_index=None if selected is None else selected['feature_index'],
            target_center_uv=None if selected is None else center(selected).tolist(),
            visible_region_xywh=box,region_component_count=pieces,representation=representation,
            selection_available=box is not None and self.state!='PARTIAL_FRAME',
            visible_region_center_uv=None if box is None else [box[0]+box[2]/2,box[1]+box[3]/2],
            missed_frames=self.missed,events=event,resolved_body=self.resolved,
            association_wall_s=time.perf_counter()-start)
