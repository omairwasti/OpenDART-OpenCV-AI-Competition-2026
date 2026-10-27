param(
    [Parameter(Mandatory=$true)][ValidateSet('Prepare','Compute')][string]$Phase,
    [Parameter(Mandatory=$true)][string]$ReleaseMetadata,
    [string]$StackName = 'opendart-task3a',
    [string]$Region = 'ap-southeast-2',
    [string]$SubnetZone = 'ap-southeast-2a',
    [Parameter(Mandatory=$true)][string]$EvidenceDirectory
)
$ErrorActionPreference = 'Stop'
if ($Region -ne 'ap-southeast-2') { throw 'Only the approved Sydney region is allowed' }
function Invoke-AwsJson([string[]]$Arguments) {
    $regionArguments = if ($Arguments -contains '--region') { @() } else { @('--region',$Region) }
    $raw = & aws @Arguments @regionArguments --output json 2>$null
    if ($LASTEXITCODE -ne 0) { throw ('AWS command failed: '+($Arguments[0..1] -join ' ')) }
    if ($raw) { return ($raw | ConvertFrom-Json) }
}
$release = Get-Content -LiteralPath $ReleaseMetadata -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $release.archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $release.sha256) { throw 'Release checksum mismatch' }
New-Item -ItemType Directory -Force -Path $EvidenceDirectory | Out-Null
$statePath = Join-Path $EvidenceDirectory 'deployment-state.json'
if ($Phase -eq 'Prepare') {
    $existing = Invoke-AwsJson @('ec2','describe-instances','--filters','Name=instance-state-name,Values=pending,running,stopping,stopped')
    if (@($existing.Reservations.Instances | Where-Object {$_}).Count -ne 0) { throw 'An EC2 instance already exists; inspect before provisioning' }
    $vpc = (Invoke-AwsJson @('ec2','describe-vpcs','--filters','Name=is-default,Values=true')).Vpcs | Select-Object -First 1
    $subnet = (Invoke-AwsJson @('ec2','describe-subnets','--filters',"Name=vpc-id,Values=$($vpc.VpcId)","Name=availability-zone,Values=$SubnetZone",'Name=default-for-az,Values=true')).Subnets | Select-Object -First 1
    if (-not $vpc -or -not $subnet -or -not $subnet.MapPublicIpOnLaunch) { throw 'Expected default public network unavailable' }
    $parameter = Invoke-AwsJson @('ssm','get-parameter','--name','/aws/service/canonical/ubuntu/server/noble/stable/current/amd64/hvm/ebs-gp3/ami-id')
    $ami = (Invoke-AwsJson @('ec2','describe-images','--image-ids',$parameter.Parameter.Value)).Images | Select-Object -First 1
    if ($ami.OwnerId -ne '099720109477' -or $ami.Architecture -ne 'x86_64' -or $ami.State -ne 'available' -or $ami.Name -notlike 'ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*') { throw 'AMI publisher, architecture or state verification failed' }
    $offered = Invoke-AwsJson @('ec2','describe-instance-type-offerings','--location-type','availability-zone','--filters','Name=instance-type,Values=c7i-flex.large',"Name=location,Values=$SubnetZone")
    if (-not $offered.InstanceTypeOfferings) { throw 'Approved instance is not offered in selected zone' }
    $pricing = Invoke-AwsJson @('pricing','get-products','--region','us-east-1','--service-code','AmazonEC2','--filters','Type=TERM_MATCH,Field=regionCode,Value=ap-southeast-2','Type=TERM_MATCH,Field=instanceType,Value=c7i-flex.large','Type=TERM_MATCH,Field=operatingSystem,Value=Linux','Type=TERM_MATCH,Field=tenancy,Value=Shared','Type=TERM_MATCH,Field=preInstalledSw,Value=NA','Type=TERM_MATCH,Field=capacitystatus,Value=Used','--max-results','10')
    $rates = foreach ($priceEntry in $pricing.PriceList) { $p=$priceEntry|ConvertFrom-Json; foreach($term in $p.terms.OnDemand.PSObject.Properties.Value){foreach($dimension in $term.priceDimensions.PSObject.Properties.Value){[double]$dimension.pricePerUnit.USD}} }
    if (-not $rates -or @($rates | Where-Object {$_ -gt 0.12}).Count) { throw 'Instance price unavailable or materially above approved estimate' }
    $state = [ordered]@{StackName=$StackName;Region=$Region;VpcId=$vpc.VpcId;SubnetId=$subnet.SubnetId;AmiId=$ami.ImageId;AmiName=$ami.Name;InstanceType='c7i-flex.large';ReleaseSha256=$release.sha256;SourceSha256=$release.source_sha256;Ec2UsdHourly=$rates[0];CreatedUtc=[DateTime]::UtcNow.ToString('o')}
    $state | ConvertTo-Json | Set-Content -LiteralPath $statePath -Encoding utf8
    Invoke-AwsJson @('cloudformation','validate-template','--template-body',('file://'+(Join-Path $PSScriptRoot 'opendart.yaml'))) | Out-Null
    $null = Invoke-AwsJson @('cloudformation','create-stack','--stack-name',$StackName,'--template-body',('file://'+(Join-Path $PSScriptRoot 'opendart.yaml')),'--capabilities','CAPABILITY_IAM','--parameters',"ParameterKey=VpcId,ParameterValue=$($vpc.VpcId)","ParameterKey=SubnetId,ParameterValue=$($subnet.SubnetId)","ParameterKey=AmiId,ParameterValue=$($ami.ImageId)",'ParameterKey=EnableCompute,ParameterValue=false','--tags','Key=Project,Value=OpenDART')
    Write-Output 'Prepare stack requested. Poll stack status before Compute phase.'
} else {
    $state = Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json
    if ($state.StackName -ne $StackName -or $state.ReleaseSha256 -ne $release.sha256) { throw 'Deployment state/release mismatch' }
    $stack = (Invoke-AwsJson @('cloudformation','describe-stacks','--stack-name',$StackName)).Stacks | Select-Object -First 1
    if ($stack.StackStatus -ne 'CREATE_COMPLETE') { throw 'Prepare stack must be CREATE_COMPLETE' }
    $bucket = ($stack.Outputs | Where-Object OutputKey -eq BucketName).OutputValue
    & aws s3 cp $release.archive "s3://$bucket/$($release.key)" --region $Region --sse AES256 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Release upload failed' }
    $manifest = Join-Path (Split-Path $ReleaseMetadata) 'release_manifest.json'
    & aws s3 cp $manifest "s3://$bucket/releases/$($release.sha256).manifest.json" --region $Region --sse AES256 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Manifest upload failed' }
    $null = Invoke-AwsJson @('cloudformation','update-stack','--stack-name',$StackName,'--use-previous-template','--capabilities','CAPABILITY_IAM','--parameters','ParameterKey=VpcId,UsePreviousValue=true','ParameterKey=SubnetId,UsePreviousValue=true','ParameterKey=AmiId,UsePreviousValue=true','ParameterKey=EnableCompute,ParameterValue=true',"ParameterKey=ReleaseKey,ParameterValue=$($release.key)","ParameterKey=ReleaseSha256,ParameterValue=$($release.sha256)")
    Write-Output 'Immutable release uploaded; Compute stack update requested.'
}
