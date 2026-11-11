# RailGuard: Codex <-> Google Antigravity CLI discussion and implementation helper.
# Run from the RailGuard Git repository in PowerShell:
#   .\agent_council.ps1 -Task "P0: audit repository and data availability" -Rounds 2
#   .\agent_council.ps1 -Task "P1: implement image preprocessing" -Rounds 2 -Implement
# No automatic git commit/push. Antigravity is given a copy of the repository.

[CmdletBinding()]
param(
    [ValidateRange(1, 3)]
    [int]$Rounds = 2,

    [string]$Task = 'P0: audit RailGuard and prepare the first milestone from WORKFLOW.md',

    [switch]$Implement
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Write-Utf8([string]$Path, [string]$Value) {
    $parent = Split-Path -Path $Path -Parent
    if ($parent -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }
    [System.IO.File]::WriteAllText($Path, $Value, [System.Text.UTF8Encoding]::new($false))
}

function Read-Text([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) { throw "Expected output file not found: $Path" }
    return [System.IO.File]::ReadAllText($Path, [System.Text.Encoding]::UTF8)
}

function Invoke-Codex([string]$Prompt, [string]$OutputPath, [bool]$AllowWrite = $false) {
    $sandbox = if ($AllowWrite) { 'workspace-write' } else { 'read-only' }
    Write-Host "Codex ($sandbox): $OutputPath" -ForegroundColor Cyan
    & codex exec --sandbox $sandbox --output-last-message $OutputPath $Prompt | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Codex exited with status $LASTEXITCODE" }
    $answer = Read-Text $OutputPath
    if ([string]::IsNullOrWhiteSpace($answer)) { throw 'Codex returned an empty response' }
    return $answer
}

function Update-ReviewSnapshot([string]$Repository, [string]$Snapshot) {
    if (Test-Path -LiteralPath $Snapshot) {
        Remove-Item -LiteralPath $Snapshot -Recurse -Force
    }
    New-Item -Path $Snapshot -ItemType Directory -Force | Out-Null
    # Review in a separate snapshot: Antigravity cannot directly edit the working repository.
    & robocopy.exe $Repository $Snapshot /E /R:1 /W:1 /NFL /NDL /NJH /NJS /NP `
        /XD '.git' '.venv' 'venv' 'data' 'outputs' 'runs' 'agent_runs' '__pycache__' '.pytest_cache' 'node_modules' `
        /XF '*.pt' '*.pth' '*.onnx' '*.ckpt' '*.safetensors' '*.zip' '*.bin' '.env'
    if ($LASTEXITCODE -ge 8) { throw "Snapshot failed (robocopy exit code $LASTEXITCODE)" }
}

function Invoke-Antigravity([string]$Prompt, [string]$OutputPath, [string]$ReviewFolder) {
    Write-Host "Antigravity (review snapshot): $OutputPath" -ForegroundColor Magenta
    Push-Location $ReviewFolder
    try {
        $output = & agy -p $Prompt --output-format text --print-timeout 10m
        $exitCode = $LASTEXITCODE
    } finally {
        Pop-Location
    }
    if ($exitCode -ne 0) { throw "Antigravity exited with status $exitCode" }
    $answer = ($output -join [Environment]::NewLine).Trim()
    if ([string]::IsNullOrWhiteSpace($answer)) { throw 'Antigravity returned an empty response' }
    Write-Utf8 $OutputPath $answer
    return $answer
}

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) { throw 'Missing codex command in PATH' }
if (-not (Get-Command agy -ErrorAction SilentlyContinue)) { throw 'Missing agy command in PATH' }
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw 'Missing git command in PATH' }
if (-not (Get-Command robocopy.exe -ErrorAction SilentlyContinue)) { throw 'robocopy.exe is required on Windows' }

$repository = (& git rev-parse --show-toplevel 2>$null)
if ($LASTEXITCODE -ne 0 -or -not $repository) { throw 'Run this script from inside the RailGuard Git repository' }
$repository = $repository.Trim()
if ([string]::IsNullOrWhiteSpace($Task)) { throw 'Task must not be empty' }
if (-not (Test-Path -LiteralPath (Join-Path $repository 'WORKFLOW.md'))) {
    throw 'WORKFLOW.md not found. First run: git pull --ff-only origin main'
}

Push-Location $repository
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$runDir = Join-Path $repository ("docs/agent_runs/$stamp")
New-Item -Path $runDir -ItemType Directory -Force | Out-Null
$snapshotRoot = Join-Path $env:TEMP ("railguard_agy_" + $stamp + '_' + $PID)
$snapshot = Join-Path $snapshotRoot 'review_copy'
New-Item -Path $snapshotRoot -ItemType Directory -Force | Out-Null
$summaryPath = Join-Path $runDir 'README.md'
$taskPath = Join-Path $runDir 'TASK.md'
Write-Utf8 $taskPath ("# Task`n`n" + $Task + "`n`nRounds: $Rounds`nImplementation enabled: $Implement`n")
$summary = @(
    '# RailGuard agent discussion',
    '',
    "Task: $Task",
    "Rounds: $Rounds",
    "Implementation: $Implement",
    '',
    'Codex is the sole source-code writer. Antigravity reviews an isolated copy.',
    'All command output and review notes are stored alongside this file.',
    'No automatic Git commit or push.'
) -join "`n"
Write-Utf8 $summaryPath $summary

try {
    $planPrompt = @"
You are the lead implementation agent for RailGuard (CS406 image processing).
Read WORKFLOW.md, docs/RESEARCH_PROTOCOL.md, README.md and relevant source code.
Current task: $Task
DESIGN PHASE ONLY: Do not edit any source files or run destructive commands.
Write a concrete, small-scope technical plan: current code audit, exact files/functions,
algorithms to compare (not blindly chain), dataset requirements, leakage risks,
unit + real-image tests, measurable acceptance criteria, and unresolved blockers.
Preserve existing M1-M4 and E1-E5 research protocol. No fabricated results.
Do not expand this task into the entire project.
"@
    $plan = Invoke-Codex $planPrompt (Join-Path $runDir 'codex_plan_0.md')

    for ($i = 1; $i -le $Rounds; $i++) {
        Update-ReviewSnapshot $repository $snapshot
        $reviewInput = @"
# Independent design review: round $i/$Rounds
Task: $Task

Read WORKFLOW.md and relevant source code in this isolated review snapshot.
Analyze methodology, illumination, Canny/Hough/RANSAC, L vs E geometry, YOLO
person detection and pose failures, test coverage, and dataset leakage as relevant.
Classify feedback as BLOCKING or NICE-TO-HAVE. Do NOT edit files.
If real imagery is missing, explicitly say what cannot be validated.

## Codex proposal
$plan
"@
        Write-Utf8 (Join-Path $snapshot 'REVIEW_INPUT.md') $reviewInput
        $reviewPrompt = 'Read REVIEW_INPUT.md, WORKFLOW.md, and relevant source code. Provide a rigorous independent peer review in Markdown. Do not modify any files.'
        $review = Invoke-Antigravity $reviewPrompt (Join-Path $runDir ("antigravity_review_$i.md")) $snapshot
        $previousPlan = Join-Path $runDir ("codex_plan_$($i-1).md")
        $reviewFile = Join-Path $runDir ("antigravity_review_$i.md")
        $revisionPrompt = @"
You are RailGuard lead agent. Read WORKFLOW.md, $previousPlan and $reviewFile.
Task: $Task. This is a READ-ONLY review phase: do not edit source files.
Explicitly accept/reject each BLOCKING issue with concrete reasoning.
Revise plan for just this one milestone, with exact tests and data blockers.
No invented experimental measurements; keep unmeasured results as TBD.
"@
        $plan = Invoke-Codex $revisionPrompt (Join-Path $runDir ("codex_plan_$i.md"))
    }

    $approvedPlan = Join-Path $runDir 'FINAL_PLAN.md'
    Write-Utf8 $approvedPlan $plan
    Write-Host "Plan available: $approvedPlan" -ForegroundColor Green

    if ($Implement) {
        $implPrompt = @"
You are the ONLY code-writing agent for RailGuard. Read WORKFLOW.md and
$approvedPlan . Execute ONLY this single milestone: $Task
Follow the final peer-reviewed plan. You may edit files in the workspace.
Add tests, run the relevant tests, and document evidence and blockers.
Do not commit, push, delete datasets, run heavy model training, or silently download
large weights. Preserve existing CLI/demo compatibility and prior tests.
Do not fabricate measurements. If data or permissions are missing, implement
what is testable and clearly list blockers.
Summarize changed files, tests, outcomes, and remaining risks.
"@
        $implementation = Invoke-Codex $implPrompt (Join-Path $runDir 'codex_implementation.md') $true
        $changes = (& git status --short | Out-String)
        Write-Utf8 (Join-Path $runDir 'git_status_after_implementation.txt') $changes
        # Important: snapshot the working tree AFTER edits, including uncommitted files.
        Update-ReviewSnapshot $repository $snapshot
        $reviewInput = @"
# Post-implementation code review
Task: $Task

Read this working-tree snapshot, WORKFLOW.md, current source and tests.
Identify blocking bugs, untested claims and remaining research limitations.
Only review. Do not edit files, install dependencies or run training.

## Final agreed plan
$plan

## Codex implementation summary
$implementation

## Changed working-tree files
$changes
"@
        Write-Utf8 (Join-Path $snapshot 'REVIEW_INPUT.md') $reviewInput
        $postReviewPrompt = 'Read REVIEW_INPUT.md and the actual source/tests in this snapshot. Review implementation independently; distinguish BLOCKING from NONBLOCKING problems. Do not modify files.'
        $postReview = Invoke-Antigravity $postReviewPrompt (Join-Path $runDir 'antigravity_code_review.md') $snapshot
        $postReviewFile = Join-Path $runDir 'antigravity_code_review.md'
        $fixPrompt = @"
You are the only code-writing agent in RailGuard for task: $Task.
Read WORKFLOW.md and $postReviewFile, then inspect your current source.
Fix valid BLOCKING problems and reject inaccurate findings with evidence.
Run affected tests and record actual results. Do not commit, push, delete data,
run heavy training, or claim tests which were not executed.
"@
        $null = Invoke-Codex $fixPrompt (Join-Path $runDir 'codex_followup.md') $true
    }
    $finalStatus = (& git status --short | Out-String)
    Write-Utf8 (Join-Path $runDir 'final_git_status.txt') $finalStatus
    $discussion = Join-Path $repository 'docs/AGENT_DISCUSSION.md'
    $previous = if (Test-Path -LiteralPath $discussion) { Read-Text $discussion } else { '# Agent discussion index' + "`n" }
    $indexEntry = "`n## $stamp`n- Task: $Task`n- Rounds: $Rounds`n- Implemented: $Implement`n- Log directory: docs/agent_runs/$stamp`n- Plan: docs/agent_runs/$stamp/FINAL_PLAN.md`n"
    Write-Utf8 $discussion ($previous.TrimEnd() + "`n" + $indexEntry)
    Write-Host "DONE. Logs: $runDir" -ForegroundColor Green
    Write-Host "Check FINAL_PLAN.md and the reviews before any commit/push." -ForegroundColor Yellow
} finally {
    Pop-Location
    if (Test-Path -LiteralPath $snapshotRoot) {
        Remove-Item -LiteralPath $snapshotRoot -Recurse -Force
    }
}