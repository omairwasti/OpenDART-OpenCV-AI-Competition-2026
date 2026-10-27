"""One bounded public inference worker; no arbitrary inputs or network access."""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--image-id',required=True)
    parser.add_argument('--request-id',required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if os.name=='posix':
        import signal
        signal.alarm(15)
    for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
    from .draco_web_runtime import analyze
    result=analyze(args.assets,args.image_id,args.request_id)
    args.out.write_text(json.dumps(result,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
