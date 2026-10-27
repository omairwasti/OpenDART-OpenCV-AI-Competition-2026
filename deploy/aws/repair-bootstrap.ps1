param(
    [Parameter(Mandatory=$true)][ValidateSet('Plan','Apply','Bootstrap')][string]$Phase,
    [Parameter(Mandatory=$true)][string]$ReleaseMetadata,
    [Parameter(Mandatory=$true)][string]$EvidenceDirectory,
    [string]$StackName = 'opendart-task3a',
    [string]$ChangeSetName = 'bootstrap-official-cli',
    [string]$Region = 'ap-southeast-2'
)
$ErrorActionPreference = 'Stop'
if ($Region -ne 'ap-southeast-2') { throw 'Unapproved region' }
function Read-Aws([string[]]$Arguments) {
    $raw = & aws @Arguments --region $Region --output json 2>$null
    if ($LASTEXITCODE -ne 0) { throw ('AWS command failed: '+($Arguments[0..1] -join ' ')) }
    if ($raw) { $raw | ConvertFrom-Json }
}
$release = Get-Content -LiteralPath $ReleaseMetadata -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $release.archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $release.sha256) { throw 'Release checksum mismatch' }
$stack = (Read-Aws @('cloudformation','describe-stacks','--stack-name',$StackName)).Stacks | Select-Object -First 1
if (-not ($stack.Tags | Where-Object {$_.Key -eq 'Project' -and $_.Value -eq 'OpenDART'})) { throw 'Stack ownership mismatch' }
if ($Phase -eq 'Plan') {
    $bucket = ($stack.Outputs | Where-Object OutputKey -eq BucketName).OutputValue
    & aws s3 cp $release.archive "s3://$bucket/$($release.key)" --region $Region --sse AES256 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Release upload failed' }
    & aws s3 cp (Join-Path (Split-Path $ReleaseMetadata) 'release_manifest.json') "s3://$bucket/releases/$($release.sha256).manifest.json" --region $Region --sse AES256 --only-show-errors
    if ($LASTEXITCODE -ne 0) { throw 'Manifest upload failed' }
    $null = Read-Aws @('cloudformation','create-change-set','--stack-name',$StackName,'--change-set-name',$ChangeSetName,'--change-set-type','UPDATE','--template-body',('file://'+(Join-Path $PSScriptRoot 'opendart.yaml')),'--capabilities','CAPABILITY_IAM','--parameters','ParameterKey=VpcId,UsePreviousValue=true','ParameterKey=SubnetId,UsePreviousValue=true','ParameterKey=AmiId,UsePreviousValue=true','ParameterKey=EnableCompute,ParameterValue=true',"ParameterKey=ReleaseKey,ParameterValue=$($release.key)","ParameterKey=ReleaseSha256,ParameterValue=$($release.sha256)")
    Write-Output 'Repair change set created; inspect readiness before Apply.'
} elseif ($Phase -eq 'Apply') {
    $change = Read-Aws @('cloudformation','describe-change-set','--stack-name',$StackName,'--change-set-name',$ChangeSetName)
    if ($change.Status -ne 'CREATE_COMPLETE') { throw 'Repair change set is not ready' }
    if ($change.Changes.Count -ne 1) { throw 'Repair must only modify Compute; STOP' }
    $resource = $change.Changes[0].ResourceChange
    if ($resource.LogicalResourceId -ne 'Compute' -or $resource.Action -ne 'Modify' -or $resource.Replacement -eq 'True') { throw 'Unexpected resource or replacement; STOP' }
    if (@($resource.Details | Where-Object {$_.Target.Name -ne 'UserData'}).Count) { throw 'Repair may only alter user data; STOP' }
    $instanceId = ($stack.Outputs | Where-Object OutputKey -eq InstanceId).OutputValue
    $instance = (Read-Aws @('ec2','describe-instances','--instance-ids',$instanceId)).Reservations.Instances | Select-Object -First 1
    # AWS documents Conditional UserData updates as restarts for EBS roots,
    # replacements for instance-store roots. Verify the actual instance.
    if ($instance.RootDeviceType -ne 'ebs' -or $instance.InstanceType -ne 'c7i-flex.large') { throw 'Expected approved EBS-backed instance unavailable; STOP' }
    $null = Read-Aws @('cloudformation','execute-change-set','--stack-name',$StackName,'--change-set-name',$ChangeSetName)
    Write-Output 'Verified existing-instance-only modification; update requested.'
} else {
    if ($stack.StackStatus -ne 'UPDATE_COMPLETE') { throw 'Wait for the existing-instance update' }
    $instanceId = ($stack.Outputs | Where-Object OutputKey -eq InstanceId).OutputValue
    $attribute = Read-Aws @('ec2','describe-instance-attribute','--instance-id',$instanceId,'--attribute','userData')
    $scriptText = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String($attribute.UserData.Value))
    if (-not $scriptText.Contains($release.sha256)) { throw 'Instance user data does not select this release' }
    $commands = "#!/bin/bash`nset -euo pipefail`nexec > /var/log/opendart-bootstrap-repair.log 2>&1`n" + $scriptText
    $parameters = Join-Path $EvidenceDirectory 'repair-bootstrap-parameters.json'
    @{commands=@($commands);executionTimeout=@('1800')} | ConvertTo-Json | Set-Content -LiteralPath $parameters -Encoding utf8
    $command = Read-Aws @('ssm','send-command','--instance-ids',$instanceId,'--document-name','AWS-RunShellScript','--parameters',('file://'+$parameters))
    [pscustomobject]@{CommandId=$command.Command.CommandId;InstanceId=$instanceId;ReleaseSha256=$release.sha256} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $EvidenceDirectory 'repair-command.json') -Encoding utf8
    Write-Output ('Bootstrap resumed through SSM; command '+$command.Command.CommandId)
}
