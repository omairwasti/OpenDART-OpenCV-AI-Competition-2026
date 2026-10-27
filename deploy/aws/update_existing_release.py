"""Upload an allowlisted release and activate it on the single existing EC2.

No create/update/delete CloudFormation, EC2 or networking operation is issued.
The existing CloudFormation instance and private release bucket are reused.
"""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import subprocess
import tarfile

REGION='ap-southeast-2'


def aws(*args):
    result=subprocess.run(['aws',*args,'--region',REGION,'--output','json'],capture_output=True,text=True)
    if result.returncode:raise RuntimeError('AWS operation failed: '+' '.join(args[:2])+'; '+re.sub(r'\b\d{12}\b','[account]',result.stderr)[:500])
    return json.loads(result.stdout) if result.stdout.strip() else None


def update(release_path,evidence):
    release_path=Path(release_path);evidence=Path(evidence)
    if (evidence/'application_update_command.json').exists():raise FileExistsError('Update command already submitted; inspect its status')
    release=json.loads(release_path.read_text());archive=Path(release['archive'])
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=release['sha256'] or not re.fullmatch('[a-f0-9]{64}',release['sha256']):raise ValueError('Invalid release')
    if release['key']!='releases/'+release['sha256']+'.tar.gz':raise ValueError('Unexpected release key')
    with tarfile.open(archive) as tar:
        names=tar.getnames()
        if any(PurePosixPath(n).is_absolute() or '..' in PurePosixPath(n).parts for n in names):raise ValueError('Unsafe archive')
        if len([n for n in names if n.startswith('draco_assets/images/') and n.endswith('.png')])!=12:raise ValueError('Expected exactly twelve curated originals')
        if any(n.endswith(('.fits','.xml')) or 'annotations' in n for n in names):raise ValueError('Disallowed dataset/annotation payload')
    stack=aws('cloudformation','describe-stacks','--stack-name','opendart-task3a')['Stacks'][0]
    if not any(t['Key']=='Project' and t['Value']=='OpenDART' for t in stack['Tags']):raise ValueError('Stack ownership mismatch')
    outputs={r['OutputKey']:r['OutputValue'] for r in stack['Outputs']}
    instances=[i for r in aws('ec2','describe-instances','--filters','Name=instance-state-name,Values=pending,running,stopping,stopped')['Reservations'] for i in r['Instances']]
    if len(instances)!=1 or instances[0]['InstanceId']!=outputs['InstanceId'] or instances[0]['InstanceType']!='c7i-flex.large' or instances[0]['State']['Name']!='running':raise ValueError('Existing single-instance inventory mismatch')
    bucket=outputs['BucketName']
    for file,key in ((archive,release['key']),(release_path.with_name('release_manifest.json'),'releases/'+release['sha256']+'.manifest.json')):
        result=subprocess.run(['aws','s3','cp',str(file),'s3://'+bucket+'/'+key,'--region',REGION,'--sse','AES256','--only-show-errors'],capture_output=True,text=True)
        if result.returncode:raise RuntimeError('Private release upload failed')
    sha=release['sha256'];destination='/opt/opendart/releases/'+sha
    script='\n'.join(['#!/bin/bash','set -euo pipefail',
        'mkdir -p '+shlex.quote(destination),
        'aws s3 cp '+shlex.quote('s3://'+bucket+'/'+release['key'])+' '+shlex.quote(destination+'/release.tar.gz')+' --region '+REGION+' --only-show-errors',
        'echo '+shlex.quote(sha+'  '+destination+'/release.tar.gz')+' | sha256sum -c -',
        'tar --no-same-owner -xzf '+shlex.quote(destination+'/release.tar.gz')+' -C '+shlex.quote(destination),
        'bash '+shlex.quote(destination+'/deploy/aws/activate_existing_release.sh')+' '+shlex.quote(destination)+' '+sha])
    parameters=evidence/'application_update_parameters.json';parameters.write_text(json.dumps(dict(commands=[script],executionTimeout=['1800']),indent=2))
    result=aws('ssm','send-command','--instance-ids',outputs['InstanceId'],'--document-name','AWS-RunShellScript','--parameters','file://'+str(parameters.resolve()))
    record=dict(command_id=result['Command']['CommandId'],instance_id=outputs['InstanceId'],release_sha256=sha,endpoint=outputs['Endpoint'],
        instance_replacement=False,new_resources=False,activation='Private S3 release + SSM application-only activation; CloudFormation resources unchanged')
    (evidence/'application_update_command.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(dict(command_id=record['command_id'],release_sha256=sha,instance_replacement=False,new_resources=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--release',type=Path,required=True);parser.add_argument('--evidence',type=Path,required=True)
    args=parser.parse_args();update(args.release,args.evidence)
