param(
    [string]$StackName = 'opendart-task3a',
    [string]$Region = 'ap-southeast-2',
    [string]$ExportDirectory,
    [string[]]$JobIds = @()
)
$ErrorActionPreference = 'Stop'
if ($Region -ne 'ap-southeast-2' -or $StackName -notmatch '^opendart-task3a(?:-[a-z0-9-]+)?$') { throw 'Unexpected deployment scope' }
function Read-Aws([string[]]$Arguments) {
    $raw = & aws @Arguments --region $Region --output json 2>$null
    if ($LASTEXITCODE -ne 0) { throw ('AWS read failed: '+($Arguments[0..1] -join ' ')) }
    if ($raw) { $raw | ConvertFrom-Json }
}
$stack = (Read-Aws @('cloudformation','describe-stacks','--stack-name',$StackName)).Stacks | Select-Object -First 1
if (-not ($stack.Tags | Where-Object {$_.Key -eq 'Project' -and $_.Value -eq 'OpenDART'})) { throw 'Stack ownership tag mismatch' }
$resources = (Read-Aws @('cloudformation','describe-stack-resources','--stack-name',$StackName)).StackResources
$approved = @('AWS::S3::Bucket','AWS::EC2::SecurityGroup','AWS::IAM::Role','AWS::IAM::InstanceProfile','AWS::EC2::Instance')
if ($resources | Where-Object {$_.ResourceType -notin $approved}) { throw 'Unexpected stack resources; inspect before cleanup' }
$bucket = ($resources | Where-Object LogicalResourceId -eq Artifacts).PhysicalResourceId
$instanceId = ($resources | Where-Object LogicalResourceId -eq Compute).PhysicalResourceId
$groupId = ($resources | Where-Object LogicalResourceId -eq WebSecurityGroup).PhysicalResourceId
$roleName = ($resources | Where-Object LogicalResourceId -eq InstanceRole).PhysicalResourceId
$profileName = ($resources | Where-Object LogicalResourceId -eq InstanceProfile).PhysicalResourceId
if (-not $bucket -or $bucket -ne ($stack.Outputs | Where-Object OutputKey -eq BucketName).OutputValue) { throw 'Bucket ownership mismatch' }
$volumes = @(); $interfaces = @()
if ($instanceId) {
    $instance = (Read-Aws @('ec2','describe-instances','--instance-ids',$instanceId)).Reservations.Instances | Select-Object -First 1
    $volumes = @($instance.BlockDeviceMappings.Ebs.VolumeId)
    $interfaces = @($instance.NetworkInterfaces.NetworkInterfaceId)
}
if ($ExportDirectory) {
    New-Item -ItemType Directory -Path $ExportDirectory -ErrorAction Stop | Out-Null
    $endpoint = ($stack.Outputs | Where-Object OutputKey -eq Endpoint).OutputValue
    foreach ($jobId in $JobIds) {
        if ($jobId -notmatch '^[a-f0-9]{32}$') { throw 'Invalid export job ID' }
        $status = Invoke-RestMethod "$endpoint/api/jobs/$jobId"
        $status | ConvertTo-Json -Depth 50 | Set-Content -LiteralPath (Join-Path $ExportDirectory "$jobId.json") -Encoding utf8
        if ($status.state -eq 'COMPLETE') {
            Invoke-WebRequest "$endpoint/jobs/$jobId/artifacts/telemetry.csv" -OutFile (Join-Path $ExportDirectory "$jobId.csv")
            foreach ($frame in $status.result.frames) {
                if ($frame.file -notmatch '^camera_(?:[0-9]{3}|final)\.png$') { throw 'Unexpected artifact name' }
                Invoke-WebRequest "$endpoint/jobs/$jobId/artifacts/$($frame.file)" -OutFile (Join-Path $ExportDirectory ($jobId+'_'+$frame.file))
            }
        }
    }
    [pscustomobject]@{StackName=$StackName;Region=$Region;Endpoint=$endpoint;ExportedUtc=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ExportDirectory 'deployment.json') -Encoding utf8
}
# This is the exact bucket obtained from the tagged stack, never a user-supplied bucket.
& aws s3 rm "s3://$bucket/" --recursive --only-show-errors --region $Region
if ($LASTEXITCODE -ne 0) { throw 'Could not empty the owned deployment bucket' }
& aws cloudformation delete-stack --stack-name $StackName --region $Region
if ($LASTEXITCODE -ne 0) { throw 'Stack deletion failed' }
$deleted = $false
for ($attempt=0; $attempt -lt 180; $attempt++) {
    $probe = & aws cloudformation describe-stacks --stack-name $StackName --region $Region --query 'Stacks[0].StackStatus' --output text 2>&1
    if ($LASTEXITCODE -ne 0) {
        if (($probe -join '') -match 'does not exist') { $deleted=$true; break }
        throw 'Stack deletion status could not be verified'
    }
    if ($probe -match 'DELETE_FAILED') { throw 'Stack deletion failed; inspect retained resources' }
    Start-Sleep -Seconds 5
}
if (-not $deleted) { throw 'Stack deletion still pending' }
if ($instanceId) {
    $remaining = (Read-Aws @('ec2','describe-instances','--instance-ids',$instanceId)).Reservations.Instances | Where-Object {$_.State.Name -ne 'terminated'}
    if ($remaining) { throw 'EC2 was not terminated' }
}
foreach ($volumeId in $volumes) { if ((Read-Aws @('ec2','describe-volumes','--filters',"Name=volume-id,Values=$volumeId")).Volumes) { throw 'Root volume remains' } }
foreach ($interfaceId in $interfaces) { if ((Read-Aws @('ec2','describe-network-interfaces','--filters',"Name=network-interface-id,Values=$interfaceId")).NetworkInterfaces) { throw 'Network interface/public IP association remains' } }
if ((Read-Aws @('ec2','describe-security-groups','--filters',"Name=group-id,Values=$groupId")).SecurityGroups) { throw 'Security group remains' }
if ((Read-Aws @('s3api','list-buckets')).Buckets | Where-Object Name -eq $bucket) { throw 'Deployment bucket remains' }
foreach ($check in @(@('get-role','--role-name',$roleName),@('get-instance-profile','--instance-profile-name',$profileName))) {
    if (-not $check[2]) { continue }
    $iamArgs = @('iam') + $check + @('--region',$Region,'--output','json')
    $probe = & aws @iamArgs 2>&1
    if ($LASTEXITCODE -eq 0 -or ($probe -join '') -notmatch 'NoSuchEntity') { throw 'IAM resource removal not verified' }
}
Write-Output 'Verified stack, EC2, root EBS, network/public IP association, security group, role/profile and bucket removal.'
