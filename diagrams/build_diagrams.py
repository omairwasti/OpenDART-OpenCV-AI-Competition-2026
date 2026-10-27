from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch

def box(ax,x,y,text,width=2.4,height=.68,color='#edf5f7',size=10):
    ax.add_patch(FancyBboxPatch((x-width/2,y-height/2),width,height,
        boxstyle='round,pad=0.08,rounding_size=0.10',facecolor=color,edgecolor='#244b62',lw=1.2))
    ax.text(x,y,text,ha='center',va='center',fontsize=size,color='#142f43')

def arrow(ax,a,b,color='#246d7d',style='-'):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=14,
        color=color,lw=1.5,linestyle=style,connectionstyle='arc3'))

def diagrams(destination):
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none'})
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    fig,ax=plt.subplots(figsize=(13,6.5));ax.set(xlim=(-.1,13.1),ylim=(-1.1,6.5));ax.axis('off')
    ax.text(.15,6.12,'OpenDART | causal synthetic feedback and AWS delivery',fontsize=18,weight='bold',color='#142f43')
    box(ax,6.5,5.35,'Ground truth: renderer geometry and evaluation only; no per-frame truth fallback',11.9,.5,'#f6f0e4',10)
    top=[(1.5,'Synthetic camera\npixels'),(4.7,'OpenCV 5\nsegmentation'),(7.9,'Candidate\nassociation'),(11.1,'PlanarEKF\nestimated state')]
    lower=[(1.5,'New camera image'),(4.7,'Spacecraft dynamics'),(7.9,'Bounded force'),(11.1,'Estimated-state\nZEM guidance')]
    for x,t in top:box(ax,x,4.15,t)
    for x,t in lower:box(ax,x,2.85,t)
    for a,b in zip(top,top[1:]):arrow(ax,(a[0]+1.3,4.15),(b[0]-1.3,4.15))
    arrow(ax,(11.1,3.73),(11.1,3.27))
    for a,b in zip(lower[1:],lower[:-1]):arrow(ax,(a[0]-1.3,2.85),(b[0]+1.3,2.85))
    arrow(ax,(1.5,3.27),(1.5,3.73))
    ax.text(6.5,1.95,'Applied commands change geometry and therefore the next rendered observation',ha='center',fontsize=11,color='#246d7d')
    for x,t in [(1.5,'Judge browser'),(4.7,'Nginx'),(7.9,'Gunicorn / Flask'),(11.1,'OpenDART /\nOpenCV 5')]:box(ax,x,.9,t,color='#e7edf4')
    for x in (1.5,4.7,7.9):arrow(ax,(x+1.3,.9),(x+1.9,.9))
    box(ax,7.9,-.3,'Private S3: deployment releases / selected evidence',7.4,.42,'#f6f0e4',9)
    arrow(ax,(11.1,-.02),(11.1,.48),style='--')
    ax.text(.2,-.95,'Existing single EC2 instance in ap-southeast-2; recorded DRACO perception is a separate, non-causal evidence path.',fontsize=9,color='#455867')
    fig.tight_layout(pad=.4)
    for ext in ('png','svg','pdf'):fig.savefig(destination/f'overall_system.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(12.5,6));ax.set(xlim=(0,12.5),ylim=(-.4,6));ax.axis('off')
    ax.text(.2,5.6,'Agentic Vision | observe, decide, act, observe again',fontsize=18,weight='bold',color='#142f43')
    for x,y,t in [(1.7,4.5,'OBSERVE\nRendered camera image'),(6.25,4.5,'PERCEIVE\nOpenCV + association'),(10.8,4.5,'ESTIMATE\nPlanarEKF'),(10.8,2.5,'DECIDE\nZEM / hold / coast'),(6.25,2.5,'ACT\nBounded applied force'),(1.7,2.5,'WORLD CHANGES\nPropagate dynamics')]:box(ax,x,y,t,2.65,.85)
    arrow(ax,(3.15,4.5),(4.8,4.5));arrow(ax,(7.7,4.5),(9.35,4.5));arrow(ax,(10.8,3.98),(10.8,3.02))
    arrow(ax,(9.35,2.5),(7.7,2.5));arrow(ax,(4.8,2.5),(3.15,2.5));arrow(ax,(1.7,3.02),(1.7,3.98))
    box(ax,6.25,.65,'No candidate / blackout -> predict only -> stale or uncertain NAV_HOLD -> zero force\nNew usable image -> association and update -> guidance can resume; terminal coast remains explicit',11.3,.95,'#fff2df',10)
    ax.text(6.25,3.5,'reacquisition uses image evidence',ha='center',fontsize=9,color='#246d7d',bbox={'facecolor':'white','edgecolor':'none','pad':2})
    ax.text(.3,-.2,'Observability: update reasons, accepted/missing/rejected counts, NAV_HOLD, applied force, covariance and outcome telemetry.',fontsize=9,color='#455867')
    fig.tight_layout(pad=.4)
    for ext in ('png','svg','pdf'):fig.savefig(destination/f'agentic_vision.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path('diagrams'));a=p.parse_args();diagrams(a.out)
