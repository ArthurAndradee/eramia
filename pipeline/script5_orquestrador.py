#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#SBATCH --job-name=hcpa_orchestrator
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=shared
#SBATCH --time=23:50:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
"""
Script 5 — Orquestrador Universal (daemon único de despacho)

Ponto de entrada ÚNICO para rodar toda a matriz experimental:

    sbatch run_all_experiments.sh
    # ou diretamente (este arquivo já tem as diretivas #SBATCH acima):
    sbatch pipeline/script5_orquestrador.py

Este processo NÃO roda os experimentos em si — ele é um despachante de
longa duração que:
  1. Lê `experiments/filter_matrix_template.json` e descobre quais filtros
     ainda faltam (via success markers já existentes).
  2. Para cada filtro pendente, submete Script 1 (criação de dataset) SEM
     `--nodelist`, deixando o próprio scheduler do SLURM escolher o nó
     (`grace1`/`grace2`) menos ocupado no momento — essa é a "distribuição
     inteligente" pedida: o SLURM já sabe fila, GPUs livres e carga atual
     melhor do que qualquer polling que este script poderia reimplementar.
  3. Descobre em qual nó aquele Script 1 caiu através de um marcador que o
     próprio job grava em `experiments/node_assignments/{filtro}.node` assim
     que começa a rodar (visível via NFS a partir de qualquer nó/processo).
  4. Fixa (`--nodelist`) todas as etapas seguintes daquele filtro nesse MESMO
     nó (obrigatório: `SSD_BASE` é disco local, não compartilhado entre nós).
  5. Registra cada transição de estágio em `logs/master.log` (única fonte
     confiável de sucesso/falha por estágio, já que o `sacct` deste cluster
     está indisponível — `slurmdbd` fora do ar).
  6. Nunca duplica submissão: antes de submeter qualquer etapa, verifica se
     já não há um job com aquele nome na fila (`squeue`) e se o estágio já
     não terminou com sucesso (via `logs/master.log`/success markers).
  7. Falhas de uma repetição/filtro são isoladas (`failed_jobs.json`) e não
     interrompem os demais.
  8. Detecta nó em estado down/drain por tempo demais e remigra o(s)
     filtro(s) associado(s) para o outro nó automaticamente, sem editar
     código (baseado em `sinfo`).
  9. Escreve relatório de progresso periódico e relatório final quando tudo
     estiver concluído ou definitivamente falho.
 10. Perto do limite de tempo da própria alocação (partição `shared`, 24h),
     se ainda houver trabalho pendente, se auto-resubmete e sai — para quem
     submeteu, continua sendo "uma vez só".

Reaproveita integralmente script1_cria_dataset.py, create-tfrecord.py,
script2_treina.py, script3_avalia.py e script4_deleta_dataset.py — este
arquivo apenas os invoca via SLURM, não duplica nenhuma lógica deles.
"""

import os
import sys
import re
import time
import argparse
import subprocess
import json
import logging
import shutil
import tempfile
import fcntl
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Set

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils_common import setup_logging, get_paths

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.dirname(PIPELINE_DIR)
SIF_PATH = os.path.join(REPO_ROOT_DIR, "singularity.sif")

NUM_REPETITIONS = 10
# Which `NUM_REPETITIONS`-sized window of repetition indices this run
# targets -- 0 (default) is the main campaign's own 0-9. Set to 10 by the
# XAI methodology mini-campaign (--rep-offset 10, see main()) so it uses
# fresh repetitions 10-19 for its 10 filters instead of colliding with
# those same filters' ALREADY-COMPLETED 0-9 from the main campaign (same
# filter name + repetition = same success marker/checkpoint identity on
# disk -- reusing 0-9 would make every stage silently no-op as "already
# done" instead of actually training under the new XAI methods). Read via
# target_rep_range() everywhere a rep range or "how many done" count is
# needed, so a single flag threads through the whole dispatcher without
# touching every call site individually.
REP_OFFSET = 0
# Script 1 (dataset creation) is deliberately submitted WITHOUT --nodelist so
# SLURM can pick whichever of grace1/grace2 is free — but that means, in the
# worst case, several filters could each get pinned to the same node before
# any of them finishes its 10 repetitions + cleanup, each holding its own
# ~6GB (dataset + TFRecords) on that node's local SSD until it's done. With
# only 2 physical nodes this is normally self-limiting, but nothing
# structurally prevents it. Capping how many filters may be "in flight"
# (dataset created, cleanup not yet done) at once bounds worst-case SSD
# usage to MAX_INFLIGHT_FILTERS x ~6GB regardless of scheduling pathology
# (see documentation/METHODOLOGICAL_NOTES.md capacity audit).
MAX_INFLIGHT_FILTERS = 6
# Sliding-window cap on SLURM queue depth for the campaign itself (Script
# 1/TFRecord/Script 2+3/Script 4 jobs) — deliberately separate from, and much
# stricter than, MAX_INFLIGHT_FILTERS above (which bounds per-node SSD usage,
# not queue depth). Requested explicitly: never hold more than this many
# campaign jobs PENDING+RUNNING at once, so the campaign never floods the
# cluster's shared job queue for other users. This only throttles WHEN a job
# is submitted — it does not skip, reorder, or duplicate any of the 5760
# (filter, repetition) work items; every submission still goes through the
# exact same generate_script_*() templates and idempotency checks as before.
MAX_CONCURRENT_CAMPAIGN_JOBS = 2
# Job-name prefixes used by every stage this orchestrator itself submits
# (see generate_script_1/_tfrecord/_2_3/_4 below) — deliberately excludes
# "hcpa_orchestrator" (the lightweight dispatcher itself, `shared` partition,
# not a campaign work item) so the dispatcher's own job never counts against
# its cap.
CAMPAIGN_JOB_PREFIXES = ("hcpa_dataset_", "hcpa_tfrecord_", "hcpa_train_", "hcpa_cleanup_")
# The 2 GPU nodes this campaign runs on (see `--partition grace`, `sinfo -p
# grace`). Used only for the node-balance preference below — spreading the
# MAX_CONCURRENT_CAMPAIGN_JOBS=2 jobs across both nodes instead of letting
# them stack on one while the other sits idle.
GRACE_NODES = {"grace1", "grace2"}
# Max total launch attempts for a single train+eval repetition before it is
# recorded as a permanent failure. The GPU/driver train crash documented in
# documentation/METHODOLOGICAL_NOTES.md is INTERMITTENT (a given rep usually
# succeeds on a fresh attempt), so a rep is auto-resubmitted until it either
# succeeds or exhausts this budget — this recovers transient crashes without
# distinguishing them a priori from a genuinely deterministic bug, which
# simply fails every attempt and then lands in the permanent-failure list
# anyway (same end state as the old no-retry behaviour, just N attempts
# later — so deterministic failures are still bounded, never looped). Only
# train/eval repetitions are covered; dataset/tfrecord failures keep their
# manual handling. The counter is derived from the number of 'train STARTED'
# events in master.log (see count_stage_attempts), which is inherently
# idempotent and survives dispatcher restarts — no separate counter state.
# At the observed ~15% per-attempt crash rate: 1 attempt -> ~15% of reps
# lost; 3 attempts -> ~0.3%.
MAX_TRAIN_ATTEMPTS = 3
POLL_INTERVAL_DEFAULT = 60          # seconds between daemon loop iterations
NODE_DOWN_TIMEOUT_DEFAULT = 1800    # seconds a node may be down/drain before we migrate its filters
SELF_RESUBMIT_MARGIN = 600          # resubmit itself if less than this many seconds remain
# A campaign job PENDING longer than this is assumed stuck waiting on a
# specific grace node that's unavailable (almost always: fully occupied by
# an unrelated job from another cluster user — jobs here are hard-pinned
# via --nodelist for retry/locality reasons, see other_grace_node()), not
# because our own MAX_CONCURRENT_CAMPAIGN_JOBS queue is genuinely busy.
# Confirmed in production: 2 jobs pinned to grace1 while it was fully held
# by another user's ~12h job stalled ALL submissions for ~2h — including to
# grace2, which was completely idle, and including brand-new Script 1 jobs
# that don't even pin a node. See cancel_stuck_pending_campaign_jobs().
STUCK_PENDING_THRESHOLD_MIN = 15

NODE_MARKER_DIR = Path(REPO_ROOT_DIR) / "experiments" / "node_assignments"
# Private, user-owned scratch for the sbatch wrapper scripts submit_job()
# generates — deliberately NOT the default system /tmp (shared, world-
# visible across every user on the login/dispatcher node). tempfile still
# guarantees a unique, race-free filename (mkstemp under the hood) and
# 0600 permissions; this just picks the containing directory (mode 700,
# same as $HOME) instead of relying on shared system temp space at all.
SBATCH_TMP_DIR = Path(REPO_ROOT_DIR) / "logs" / ".sbatch_tmp"
MASTER_LOG_PATH = Path(REPO_ROOT_DIR) / "logs" / "master.log"
PROGRESS_REPORT_PATH = Path(REPO_ROOT_DIR) / "logs" / "progress_report.txt"
FINAL_REPORT_PATH = Path(REPO_ROOT_DIR) / "logs" / "final_report.txt"
FAILED_JOBS_PATH = Path(REPO_ROOT_DIR) / "failed_jobs.json"
ALERT_LOG_PATH = Path(REPO_ROOT_DIR) / "logs" / "ALERTAS_PENDENTES.txt"
NOTIFY_EMAIL = "botooooxgamer123@gmail.com"
NODE_DOWN_SINCE_PATH = Path(REPO_ROOT_DIR) / "experiments" / "node_down_since.json"

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Filter matrix
# --------------------------------------------------------------------------

class FilterMatrix:
    """Loads the JSON list of filters to process."""

    def __init__(self, matrix_file: Optional[Path] = None, logger=None):
        self.logger = logger
        self.matrix = []
        if matrix_file and matrix_file.exists():
            self.load_from_file(matrix_file)
        else:
            raise FileNotFoundError(
                f"Filter matrix file not found: {matrix_file}. "
                f"Provide --matrix-file or ensure experiments/filter_matrix_template.json exists."
            )

    def load_from_file(self, matrix_file: Path):
        with open(matrix_file, 'r') as f:
            self.matrix = json.load(f)
        if self.logger:
            self.logger.info(f"Loaded filter matrix from {matrix_file}: {len(self.matrix)} filters")

    def get_filters(self) -> List[Dict]:
        return self.matrix


# --------------------------------------------------------------------------
# SLURM job generation/submission — reuses the exact same per-script CLIs as
# the standalone slurm/job_template_*.sh; only wraps them with bookkeeping
# (node marker, master.log events) needed for autonomous orchestration.
# --------------------------------------------------------------------------

class SLURMJobSubmitter:
    def __init__(self, partition: str = "grace", logger=None,
                 xai_methods: Optional[str] = None, xai_subsample: Optional[int] = None,
                 keep_checkpoint: bool = False):
        self.partition = partition
        self.logger = logger
        # XAI methodology mini-campaign options (see script3_avalia.py's own
        # --xai-methods/--xai-subsample/--keep-checkpoint) — None/False here
        # means generate_script_2_3() omits these flags entirely, so
        # script3_avalia.py falls back to its original defaults (Grad-CAM
        # only, no subsampling, delete checkpoint after eval). The main
        # 576-filter campaign never sets these — only the 10-filter mini
        # campaign does, via main()'s CLI args below.
        self.xai_methods = xai_methods
        self.xai_subsample = xai_subsample
        self.keep_checkpoint = keep_checkpoint

    def submit_job(self, script_content: str, job_name: str, use_gpu: bool = False,
                   dependency: Optional[str] = None, time_limit: str = "24:00:00",
                   nodelist: Optional[str] = None) -> Optional[str]:
        script_path = None
        try:
            SBATCH_TMP_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False,
                                              dir=str(SBATCH_TMP_DIR)) as f:
                f.write(script_content)
                script_path = f.name
            os.chmod(script_path, 0o700)

            cmd = ["sbatch", "--partition", self.partition, "--exclusive", "--job-name", job_name]
            if nodelist:
                cmd.extend(["--nodelist", nodelist])
            if use_gpu:
                cmd.extend(["--gres", "gpu:1"])
            if time_limit:
                cmd.extend(["--time", time_limit])
            if dependency:
                cmd.extend(["--dependency", dependency])
            cmd.append(script_path)

            if self.logger:
                self.logger.info(f"Submitting: {' '.join(cmd)}")

            # cwd=REPO_ROOT_DIR: `#SBATCH --output=logs/...` resolves relative
            # to wherever sbatch is invoked from, not to this process's own cwd.
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, cwd=REPO_ROOT_DIR)
            if result.returncode != 0:
                if self.logger:
                    self.logger.error(f"sbatch failed for {job_name}: {result.stderr.strip()}")
                return None
            output = result.stdout.strip()
            if "Submitted batch job" in output:
                return output.split()[-1]
            if self.logger:
                self.logger.error(f"Could not parse job ID from: {output}")
            return None
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to submit {job_name}: {e}")
            return None
        finally:
            if script_path and os.path.exists(script_path):
                os.unlink(script_path)

    # -- bash snippet shared by every generated script -------------------

    @staticmethod
    def _preamble(filter_name: str, base_ssd: str, base_home: str) -> str:
        return f"""cd "{PIPELINE_DIR}"

export SSD_BASE="{base_ssd}"
export HOME_BASE="{base_home}"

# SSD_BASE is node-local scratch, NOT auto-bound by this cluster's
# singularity.conf (confirmed empirically) — must bind explicitly, and must
# exist on the host before --bind will accept it.
mkdir -p "$SSD_BASE"
mkdir -p "{NODE_MARKER_DIR}"
mkdir -p "{MASTER_LOG_PATH.parent}"

MASTER_LOG="{MASTER_LOG_PATH}"
MASTER_LOG_LOCK="{MASTER_LOG_PATH}.lock"
FILTER_NAME="{filter_name}"
log_event() {{
    # log_event <stage> <status> [rep]
    # flock-guarded: master.log is appended to concurrently by jobs on both
    # grace1 AND grace2 (and by the Python dispatcher itself via
    # log_master_event()) over NFS — a single short printf is very unlikely
    # to interleave, but flock makes the append atomic with certainty rather
    # than relying on that assumption.
    (
        flock -x 200
        printf '[%s] filter=%s node=%s gpu=%s stage=%s rep=%s status=%s\\n' \\
            "$(date '+%Y-%m-%d %H:%M:%S')" "$FILTER_NAME" "$(hostname)" "${{GPU_IDX:-NA}}" "$1" "${{3:-NA}}" "$2" >> "$MASTER_LOG"
    ) 200>"$MASTER_LOG_LOCK"
}}
"""

    def generate_script_1(self, filter_name: str, base_ssd: str, base_home: str) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name=hcpa_dataset_{filter_name}
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --partition=grace

set -uo pipefail
{self._preamble(filter_name, base_ssd, base_home)}
# Record which node this filter landed on — the ONLY reliable way the
# dispatcher (which may be running on a completely different node) learns
# this, since SLURM's own job history (sacct) is unavailable on this cluster.
echo "$(hostname)" > "{NODE_MARKER_DIR}/{filter_name}.node"
log_event dataset STARTED

if singularity exec --bind "{REPO_ROOT_DIR}:{REPO_ROOT_DIR}" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" "{SIF_PATH}" \\
        python3 script1_cria_dataset.py --filtro "{filter_name}" --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"
then
    log_event dataset SUCCESS
    echo "✓ Script 1 completed successfully"
else
    log_event dataset FAILED
    echo "✗ Script 1 failed"
    exit 1
fi
"""

    def generate_script_tfrecord(self, filter_name: str, base_ssd: str, base_home: str) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name=hcpa_tfrecord_{filter_name}
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --partition=grace

set -uo pipefail
{self._preamble(filter_name, base_ssd, base_home)}
log_event tfrecord STARTED

if singularity exec --bind "{REPO_ROOT_DIR}:{REPO_ROOT_DIR}" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" "{SIF_PATH}" \\
        python3 create-tfrecord.py --filtro "{filter_name}" --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"
then
    log_event tfrecord SUCCESS
    echo "✓ TFRecord generation completed successfully"
else
    log_event tfrecord FAILED
    echo "✗ TFRecord generation failed"
    exit 1
fi
"""

    def generate_script_2_3(self, filter_name: str, repetition: int, base_ssd: str, base_home: str,
                           time_limit: str = "10:00:00") -> str:
        extra_eval_flags = ""
        if self.xai_methods:
            extra_eval_flags += f' --xai-methods "{self.xai_methods}"'
        if self.xai_subsample:
            extra_eval_flags += f" --xai-subsample {self.xai_subsample}"
        if self.keep_checkpoint:
            extra_eval_flags += " --keep-checkpoint"
        checkpoint_log_status = "KEPT" if self.keep_checkpoint else "REMOVED"
        return f"""#!/bin/bash
#SBATCH --job-name=hcpa_train_{filter_name}_{repetition}
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time={time_limit}
#SBATCH --partition=grace
#SBATCH --gres=gpu:1

set -uo pipefail
{self._preamble(filter_name, base_ssd, base_home)}
export GPU_IDX=0
SINGULARITY_ARGS=(exec --nv --bind "{REPO_ROOT_DIR}:{REPO_ROOT_DIR}" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" "{SIF_PATH}")

log_event train STARTED {repetition}
if singularity "${{SINGULARITY_ARGS[@]}}" python3 script2_treina.py \\
        --filtro "{filter_name}" --repeticao {repetition} --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"
then
    log_event train SUCCESS {repetition}
    echo "✓ Script 2 completed successfully"
else
    log_event train FAILED {repetition}
    echo "✗ Script 2 failed"
    exit 1
fi

log_event eval STARTED {repetition}
if singularity "${{SINGULARITY_ARGS[@]}}" python3 script3_avalia.py \\
        --filtro "{filter_name}" --repeticao {repetition} --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"{extra_eval_flags}
then
    log_event eval SUCCESS {repetition}
    log_event checkpoint {checkpoint_log_status} {repetition}
    echo "✓ Script 3 completed successfully"
else
    log_event eval FAILED {repetition}
    echo "✗ Script 3 failed"
    exit 1
fi
"""

    def generate_script_4(self, filter_name: str, base_ssd: str, base_home: str) -> str:
        return f"""#!/bin/bash
#SBATCH --job-name=hcpa_cleanup_{filter_name}
#SBATCH --output=logs/%x_%j.log
#SBATCH --error=logs/%x_%j.err
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --partition=grace

set -uo pipefail
{self._preamble(filter_name, base_ssd, base_home)}
SINGULARITY_ARGS=(exec --bind "{REPO_ROOT_DIR}:{REPO_ROOT_DIR}" --bind "$HOME:$HOME" --bind "$SSD_BASE:$SSD_BASE" "{SIF_PATH}")

log_event cleanup STARTED

# Dry-run first (safety check — script4 itself also refuses to delete unless
# all {NUM_REPETITIONS} success markers exist, this is belt-and-suspenders).
if ! singularity "${{SINGULARITY_ARGS[@]}}" python3 script4_deleta_dataset.py \\
        --filtro "{filter_name}" --base-ssd "$SSD_BASE" --base-home "$HOME_BASE" --dry-run
then
    log_event cleanup FAILED
    echo "✗ Script 4 dry-run failed"
    exit 1
fi

if singularity "${{SINGULARITY_ARGS[@]}}" python3 script4_deleta_dataset.py \\
        --filtro "{filter_name}" --base-ssd "$SSD_BASE" --base-home "$HOME_BASE"
then
    log_event cleanup SUCCESS
    echo "✓ Script 4 (cleanup) completed successfully"
else
    log_event cleanup FAILED
    echo "✗ Script 4 (cleanup) failed"
    exit 1
fi
"""


# --------------------------------------------------------------------------
# Cluster/queue introspection (read-only; never used to pick a node — that
# choice is delegated to SLURM itself by omitting --nodelist on Script 1)
# --------------------------------------------------------------------------

def squeue_job_names(user: str) -> Set[str]:
    """Set of job names currently PENDING or RUNNING for `user`."""
    try:
        result = subprocess.run(["squeue", "-u", user, "-h", "-o", "%j"],
                                 capture_output=True, text=True, timeout=15)
        return {line.strip() for line in result.stdout.splitlines() if line.strip()}
    except Exception as e:
        logger.warning(f"squeue query failed ({e}) — assuming nothing is queued (may cause a redundant resubmission attempt, which is safe/idempotent)")
        return set()


def count_active_campaign_jobs(queued_names: Set[str]) -> int:
    """How many PENDING+RUNNING jobs in `queued_names` belong to this
    campaign (Script 1/TFRecord/Script 2+3/Script 4) — used to enforce
    MAX_CONCURRENT_CAMPAIGN_JOBS. Excludes the dispatcher's own
    "hcpa_orchestrator" job by construction (its name doesn't match any
    CAMPAIGN_JOB_PREFIXES entry)."""
    return sum(1 for name in queued_names if name.startswith(CAMPAIGN_JOB_PREFIXES))


def cancel_stuck_pending_campaign_jobs(user: str) -> int:
    """Cancel this campaign's own jobs that have been PENDING for more than
    STUCK_PENDING_THRESHOLD_MIN minutes (see that constant's comment for
    why this matters — almost always a job hard-pinned via --nodelist to a
    grace node another cluster user is occupying for a long time).

    REPLACES an earlier version of this fix that only *excluded* such jobs
    from the MAX_CONCURRENT_CAMPAIGN_JOBS budget count instead of cancelling
    them. That had no ceiling: every cycle, whichever jobs newly crossed the
    15-minute mark freed up budget for 1-2 MORE submissions, but nothing
    ever reduced the total again while the external occupation persisted —
    confirmed in production: 33 jobs piled up over ~22h of grace1 being held
    by another user, silently recreating the exact mass-submission problem
    this dispatcher was built to prevent. Cancelling instead of discounting
    keeps the hard invariant this campaign was designed around (never more
    than MAX_CONCURRENT_CAMPAIGN_JOBS jobs in flight, PENDING+RUNNING, at
    any time) unconditionally true, no matter how long an external
    occupation lasts.

    Only ever touches this campaign's own jobs: `squeue -u {user}` is
    scoped to our own user by SLURM itself, and job-name prefix-filtered on
    top of that — never another user's job, and `scancel` would refuse a
    foreign job anyway even if one somehow matched. Safe for other cluster
    users: a job stuck PENDING has never started, holds no node/GPU/memory
    allocation, and cancelling it only removes bookkeeping from the
    scheduler's own pending queue (a (very minor) net positive for
    everyone else, never a negative). Safe for this campaign: nothing was
    running, no checkpoint/data exists yet, and dedup is keyed by job NAME
    (not job ID) — the next cycle just re-evaluates that filter/rep as if
    it had never been submitted and resubmits it through the normal path
    (including the existing retry node-diversion preference, if it's a
    retry). Returns the number cancelled, for logging only."""
    try:
        result = subprocess.run(
            ["squeue", "-u", user, "-h", "-t", "PENDING", "-o", "%i %j %V"],
            capture_output=True, text=True, timeout=15,
        )
    except Exception as e:
        logger.warning(f"squeue (stuck-pending check) failed ({e}) — assuming none stuck")
        return 0
    now = datetime.now()
    to_cancel = []
    for line in result.stdout.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) != 3:
            continue
        jobid, name, submit_str = parts
        if not name.startswith(CAMPAIGN_JOB_PREFIXES):
            continue
        try:
            submit_time = datetime.strptime(submit_str, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            continue
        pending_minutes = (now - submit_time).total_seconds() / 60
        if pending_minutes > STUCK_PENDING_THRESHOLD_MIN:
            to_cancel.append((jobid, name, round(pending_minutes)))
    for jobid, name, pending_minutes in to_cancel:
        try:
            subprocess.run(["scancel", jobid], capture_output=True, text=True, timeout=15)
            logger.warning(
                f"Cancelled stuck-PENDING job {name} (job={jobid}, pending {pending_minutes}min) "
                f"— almost certainly waiting on a node another cluster user is occupying; "
                f"will be re-evaluated and resubmitted through the normal path next cycle."
            )
        except Exception as e:
            logger.warning(f"scancel failed for stuck job {jobid} ({name}): {e}")
    return len(to_cancel)


def filter_has_active_job(filter_name: str, queued_names: Set[str]) -> bool:
    """True if `filter_name` currently has any PENDING/RUNNING campaign job
    (dataset/tfrecord/cleanup, or a train+eval job for any repetition).
    Used to seed `nodes_busy` at the start of each dispatcher cycle from
    jobs that were already queued before this cycle started.

    The train-job check requires the remainder after the prefix to be all
    digits (the repetition number) rather than a plain `startswith` — filter
    names in this matrix are sometimes prefixes of OTHER filter names (e.g.
    "AHE40.0" vs "AHE40.0_Unsharp1.5"), so a bare prefix match on
    "hcpa_train_AHE40.0_" would also incorrectly match
    "hcpa_train_AHE40.0_Unsharp1.5_0"."""
    if f"hcpa_dataset_{filter_name}" in queued_names:
        return True
    if f"hcpa_tfrecord_{filter_name}" in queued_names:
        return True
    if f"hcpa_cleanup_{filter_name}" in queued_names:
        return True
    train_prefix = f"hcpa_train_{filter_name}_"
    return any(name.startswith(train_prefix) and name[len(train_prefix):].isdigit()
               for name in queued_names)


def node_balance_blocks(node: Optional[str], nodes_busy: Set[str], enforce_balance: bool) -> bool:
    """True if this submission should be deferred to prefer node balance:
    `node` already has a campaign job this cycle AND at least one other
    grace node is still completely free. Never blocks when `node` is None
    (Script 1, before SLURM has picked a node — nothing to balance yet)."""
    if not enforce_balance or node is None:
        return False
    return node in nodes_busy and bool(GRACE_NODES - nodes_busy)


def other_grace_node(node: Optional[str]) -> Optional[str]:
    """The other grace node, or None if `node` isn't a recognized grace node.

    Used to route train-stage RETRIES (attempt >= 2) away from the node a
    rep just failed on. Diagnosed from master.log: failures cluster in
    temporary per-node bad streaks (on grace2, P(fail | previous job on the
    same node also failed) = 50.4% vs 20.3% after a success) — and because
    a filter is pinned to one node for its whole lifecycle
    (get_node_assignment), every retry was landing on that same node,
    turning a streak into a near-certain loss instead of an independent
    second chance. Safe to diversify: SSD_BASE resolves to the shared
    $HOME filesystem in practice (confirmed separately), not real
    node-local scratch, so a retry on the other node pays no locality
    cost. Dataset/tfrecord/cleanup stay pinned to the original node —
    only train-stage retries use this."""
    others = GRACE_NODES - {node}
    return next(iter(others), None) if node in GRACE_NODES else None


def squeue_busy_nodes(partition: str) -> Set[str]:
    """Nodes in `partition` currently running a GPU (train_*) job."""
    try:
        result = subprocess.run(["squeue", "-p", partition, "-h", "-t", "RUNNING", "-o", "%N %j"],
                                 capture_output=True, text=True, timeout=15)
        busy = set()
        for line in result.stdout.splitlines():
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2 and "train" in parts[1]:
                busy.add(parts[0])
        return busy
    except Exception as e:
        logger.warning(f"squeue busy-node query failed: {e}")
        return set()


def sinfo_node_states(partition: str) -> Dict[str, str]:
    try:
        result = subprocess.run(["sinfo", "-p", partition, "-h", "-N", "-o", "%N %t"],
                                 capture_output=True, text=True, timeout=15)
        states = {}
        for line in result.stdout.splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                states[parts[0]] = parts[1]
        return states
    except Exception as e:
        logger.warning(f"sinfo query failed: {e}")
        return {}


# --------------------------------------------------------------------------
# State: node assignments (experiments/node_assignments/{filter}.node) and
# master.log (the single source of truth for per-stage success/failure,
# since this cluster's `sacct` is unavailable).
# --------------------------------------------------------------------------

def get_node_assignment(filter_name: str) -> Optional[str]:
    p = NODE_MARKER_DIR / f"{filter_name}.node"
    if p.exists():
        content = p.read_text().strip()
        return content or None
    return None


def clear_node_assignment(filter_name: str):
    p = NODE_MARKER_DIR / f"{filter_name}.node"
    if p.exists():
        p.unlink()


_MASTER_LOG_LINE_RE = re.compile(
    r"^\[(?P<ts>[^\]]+)\] filter=(?P<filter>\S+) node=(?P<node>\S+) gpu=(?P<gpu>\S+) "
    r"stage=(?P<stage>\S+) rep=(?P<rep>\S+) status=(?P<status>\S+)"
)


def _read_master_log_lines() -> List[dict]:
    if not MASTER_LOG_PATH.exists():
        return []
    events = []
    with open(MASTER_LOG_PATH, "r", errors="replace") as f:
        for line in f:
            m = _MASTER_LOG_LINE_RE.match(line)
            if m:
                events.append(m.groupdict())
    return events


def last_stage_status(events: List[dict], filter_name: str, stage: str, rep: Optional[int] = None) -> Optional[str]:
    """Most recent status (STARTED/SUCCESS/FAILED) logged for filter+stage[+rep]."""
    rep_str = str(rep) if rep is not None else "NA"
    status = None
    for e in events:
        if e["filter"] == filter_name and e["stage"] == stage and e["rep"] == rep_str:
            status = e["status"]
    return status


def count_stage_attempts(events: List[dict], filter_name: str, stage: str, rep: Optional[int] = None) -> int:
    """Number of times a (filter, stage[, rep]) job was actually launched =
    count of 'STARTED' events for it in master.log. Each generated job logs
    exactly one STARTED for its stage when it begins running, so this is a
    reliable, idempotent (append-only log => stable count) and restart-safe
    attempt counter — used by the train/eval auto-retry (MAX_TRAIN_ATTEMPTS)
    without needing any separate persisted counter."""
    rep_str = str(rep) if rep is not None else "NA"
    return sum(1 for e in events
               if e["filter"] == filter_name and e["stage"] == stage
               and e["rep"] == rep_str and e["status"] == "STARTED")


def _train_failure_reason(train_status: Optional[str], eval_status: Optional[str]) -> str:
    """Human-readable classification of why a train+eval rep attempt failed,
    for the retry audit log (the RESULT of each retry is captured separately
    by the resubmitted job's own STARTED/SUCCESS/FAILED events in master.log
    and, on success, by its resultados/ CSV)."""
    if train_status == "FAILED":
        return "Script 2 (training) exited with failure (likely intermittent GPU/driver crash)"
    if eval_status == "FAILED":
        return "Script 3 (evaluation) exited with failure"
    if eval_status == "STARTED":
        return "hard crash during evaluation (job left queue with no completion signal)"
    if train_status == "STARTED":
        return "hard crash during training (job left queue with no completion signal)"
    return "unknown failure (no train/eval completion signal in master.log)"


def log_master_event(filter_name: str, node: str, stage: str, status: str, rep: Optional[int] = None):
    """Used by the dispatcher itself (not the generated job scripts) to log
    orchestration-level events (submissions, migrations, self-resubmit).

    flock-guarded against the SAME lock file the bash log_event() helper
    (see SLURMJobSubmitter._preamble) uses, since job scripts running on
    grace1/grace2 append to this same master.log concurrently with the
    dispatcher process itself.
    """
    MASTER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock_path = MASTER_LOG_PATH.with_suffix(MASTER_LOG_PATH.suffix + ".lock")
    rep_str = str(rep) if rep is not None else "NA"
    line = (f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] filter={filter_name} node={node} "
            f"gpu=NA stage={stage} rep={rep_str} status={status}\n")
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            with open(MASTER_LOG_PATH, "a") as f:
                f.write(line)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


# --------------------------------------------------------------------------
# Success markers (the pre-existing, authoritative "is this rep done" signal)
# --------------------------------------------------------------------------

def count_inflight_filters(success_markers_dir: Path, failed: dict, events: List[dict],
                            scope_filters: Optional[Set[str]] = None) -> int:
    """
    Number of filters currently pinned to a node (dataset already created)
    but not yet fully resolved (completed+cleaned-up, or permanently
    failed) — i.e. filters whose ~6GB dataset+TFRecords are presently
    occupying node-local SSD. Used to cap MAX_INFLIGHT_FILTERS so worst-case
    SSD usage stays bounded regardless of how SLURM happens to schedule the
    unpinned Script 1 jobs (see MAX_INFLIGHT_FILTERS comment above).

    `scope_filters`, when given, restricts the count to filter names in
    that set — i.e. the CURRENT run's own matrix. Without this, a small
    side matrix (e.g. the 10-filter XAI mini-campaign) gets blocked by
    leftover .node markers from a completely different, already-finished
    campaign run: confirmed live, 61 main-campaign filters sitting at
    partial completion (never marked failed, cleanup never ran, so their
    datasets are still genuinely on disk) blew MAX_INFLIGHT_FILTERS=6
    before the mini-campaign's own 10 filters -- which have nothing to do
    with that leftover state -- ever got a chance to submit anything. For
    the main campaign, scope_filters covers its own full matrix, so this
    is a no-op (unchanged behavior)."""
    count = 0
    for marker_file in NODE_MARKER_DIR.glob("*.node"):
        filter_name = marker_file.stem
        if scope_filters is not None and filter_name not in scope_filters:
            continue
        if not _filter_disk_resolved(filter_name, success_markers_dir, failed, events):
            count += 1
    return count


def target_rep_range() -> range:
    """The NUM_REPETITIONS-sized window of repetition indices this run
    targets -- see REP_OFFSET's comment."""
    return range(REP_OFFSET, REP_OFFSET + NUM_REPETITIONS)


def success_marker_reps(success_markers_dir: Path, filter_name: str) -> Set[int]:
    """Repetitions with an existing success marker, restricted to
    target_rep_range() -- so with REP_OFFSET=0 (main campaign, unchanged
    behavior) this is exactly the markers that exist; with REP_OFFSET=10
    (XAI mini-campaign) any pre-existing 0-9 markers from the main
    campaign are invisible here, so this filter's progress is judged only
    by its NEW 10-19 markers, not the old ones."""
    return _reps_done_in_range(success_markers_dir, filter_name, target_rep_range())


def _reps_done_in_range(success_markers_dir: Path, filter_name: str, rep_range) -> Set[int]:
    """Success-marker repetitions for `filter_name` restricted to an
    arbitrary range, independent of the current REP_OFFSET -- used by
    filter_is_resolved() to also recognize completion in the ORIGINAL
    0-9 window for filters outside the current run's own matrix (see its
    docstring).

    Matches markers by an EXACT `{filter_name}_{rep}.ok` prefix, not a bare
    glob(f"{filter_name}_*.ok") -- that glob is not prefix-anchored, so for
    e.g. filter_name="BenGraham_MaxGreen2.0" it also matches
    "BenGraham_MaxGreen2.0_Canny_10.ok", a DIFFERENT filter that merely
    shares a name prefix (extremely common here -- this project's filter
    names are composable chains, e.g. "X", "X_Canny", "X_AHE40.0" all
    coexist). That silently merged a sibling filter's markers into this
    one's rep count and made this function's caller believe filters were
    complete when they were not. Confirmed live: BenGraham_MaxGreen2.0
    genuinely had only 4/10 mini-campaign repetitions on disk (reps
    10/16/17/19; the CSVs and checkpoints for 11-15/18 never existed) but
    write_final_report() reported it as "10/10 concluído" -- the missing
    6 were BenGraham_MaxGreen2.0_Canny's own markers, wrongly counted
    here. Requiring the remainder after the exact "{filter_name}_" prefix
    to be ALL DIGITS (a real repetition number, not "Canny_10") fixes
    this without changing behavior for any filter name that isn't a
    prefix of another."""
    reps = set()
    target = set(rep_range)
    prefix = f"{filter_name}_"
    for p in success_markers_dir.glob(f"{filter_name}_*.ok"):
        rest = p.stem[len(prefix):] if p.stem.startswith(prefix) else None
        if not rest or not rest.isdigit():
            continue
        rep = int(rest)
        if rep in target:
            reps.add(rep)
    return reps


# --------------------------------------------------------------------------
# Failure bookkeeping
# --------------------------------------------------------------------------

def _empty_failed_jobs() -> dict:
    return {"filters": {}, "repetitions": {}, "cleanup": {}}


def load_failed_jobs() -> dict:
    if FAILED_JOBS_PATH.exists():
        try:
            data = json.loads(FAILED_JOBS_PATH.read_text())
            data.setdefault("filters", {})
            data.setdefault("repetitions", {})
            data.setdefault("cleanup", {})
            return data
        except Exception:
            return _empty_failed_jobs()
    return _empty_failed_jobs()


def save_failed_jobs(data: dict):
    FAILED_JOBS_PATH.write_text(json.dumps(data, indent=2, sort_keys=True))


def notify_operator(subject: str, body: str):
    """Best-effort operator notification for something the daemon's own
    automated recovery (retry, node migration, self-resubmit) could NOT
    fix -- requested explicitly (run unattended, but notify on anything
    that can't be auto-repaired). Two layers, deliberately redundant:

    1. ALERT_LOG_PATH (guaranteed): a plain append-only local file. This
       is the reliable channel -- always works, no external dependency.
    2. `mailx` to NOTIFY_EMAIL (best-effort): HPC mail relays frequently
       accept a message locally (exit 0) without it ever actually being
       delivered externally (no SPF/DKIM, silently dropped upstream) --
       confirmed only that the local relay accepts submissions, NOT that
       delivery to Gmail succeeds. Never trust this as the only channel;
       never let its failure block or crash the caller.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"[{timestamp}] {subject}\n{body}\n{'-' * 60}\n"
    try:
        ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ALERT_LOG_PATH, "a") as f:
            f.write(entry)
    except Exception as e:
        logger.error(f"Could not write to ALERT_LOG_PATH ({e}) — alert only in daemon log: {subject}")
    try:
        subprocess.run(
            ["mailx", "-s", f"[HCPA] {subject}", NOTIFY_EMAIL],
            input=body, capture_output=True, text=True, timeout=20,
        )
    except Exception as e:
        logger.warning(f"mailx notification failed ({e}) — alert still recorded in {ALERT_LOG_PATH}")


def record_failure(failed: dict, filter_name: str, stage: str, rep: Optional[int], reason: str):
    """Record a permanent failure and stop the daemon from retrying it
    every cycle. `stage='cleanup'` is bucketed separately from
    dataset/tfrecord failures: a cleanup failure means the science for this
    filter is already done (all repetitions evaluated) — only the disk
    cleanup didn't happen — so it must NOT make the filter look unfinished
    or FAILED overall, just stop the retry spam and flag it for manual
    cleanup of datasets_filtrados/TFRecords."""
    entry = {
        "filter": filter_name,
        "stage": stage,
        "repetition": rep,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
        "log_hint": f"logs/hcpa_{'train' if stage in ('train', 'eval') else ('dataset' if stage == 'dataset' else ('tfrecord' if stage == 'tfrecord' else 'cleanup'))}_"
                    f"{filter_name}{f'_{rep}' if rep is not None else ''}_*.log",
    }
    if stage == "cleanup":
        bucket, key = failed["cleanup"], filter_name
    elif rep is None:
        bucket, key = failed["filters"], filter_name
    else:
        bucket, key = failed["repetitions"], f"{filter_name}_{rep}"
    is_new = key not in bucket
    bucket[key] = entry
    save_failed_jobs(failed)
    logger.error(f"✗ FAILED: filter={filter_name} stage={stage} rep={rep} reason={reason}")
    if is_new:
        # Only on the FIRST time this exact filter/rep/stage gives up
        # permanently (not on every dispatcher cycle after) -- this is the
        # single choke point every kind of unrecoverable failure funnels
        # through (dataset, tfrecord, train/eval repetition, cleanup), so
        # one notification call here covers all of them.
        notify_operator(
            f"falha permanente: {filter_name} ({stage}{f', rep {rep}' if rep is not None else ''})",
            f"Filtro: {filter_name}\nEstagio: {stage}\nRepeticao: {rep}\nMotivo: {reason}\n"
            f"O daemon ja esgotou as tentativas automaticas para isso -- nao vai tentar de novo sozinho.\n"
            f"Ver failed_jobs.json e {ALERT_LOG_PATH} para o historico completo.",
        )


# --------------------------------------------------------------------------
# Per-filter reconciliation/dispatch — the heart of the daemon. Called every
# loop iteration for every filter; always safe to call repeatedly (never
# double-submits: checks squeue + master.log before acting).
# --------------------------------------------------------------------------

def process_filter(filter_name: str, submitter: SLURMJobSubmitter, base_ssd: str, base_home: str,
                    success_markers_dir: Path, queued_names: Set[str], failed: dict,
                    budget: List[int], nodes_busy: Set[str], events: List[dict],
                    enforce_balance: bool = True, matrix_filters: Optional[Set[str]] = None) -> str:
    """Advance one filter's pipeline by whatever step is currently actionable.
    Returns a short status string for progress reporting.

    `budget` is a 1-element mutable list shared across every filter processed
    in the current dispatcher cycle: budget[0] starts each cycle at
    MAX_CONCURRENT_CAMPAIGN_JOBS minus jobs already queued, and is
    decremented by 1 every time this function actually calls
    submitter.submit_job() successfully. Once it hits 0, every remaining
    submission this cycle is deferred (treated exactly like an sbatch
    failure — "will retry next cycle") rather than attempted, so the queue
    depth for this campaign never exceeds MAX_CONCURRENT_CAMPAIGN_JOBS.

    `nodes_busy` is shared the same way, tracking which of grace1/grace2
    already have a campaign job (queued before this cycle, or submitted
    earlier in this same cycle). When `enforce_balance` is True (pass 1 of
    the dispatcher's per-cycle loop — see run_daemon()), a submission whose
    filter is pinned to a node already in `nodes_busy`, while the OTHER
    tracked node is still free, is deferred — giving a filter pinned to the
    idle node first refusal at the remaining budget instead of stacking two
    jobs on the same node. `enforce_balance=False` (pass 2, only for filters
    deferred this way in pass 1) drops that preference so any leftover
    budget still gets used if no filter targets the idle node. This changes
    ONLY when/whether a job is submitted — every stage, ordering,
    idempotency check and job template below is unchanged.

    `events` is master.log parsed ONCE per dispatcher cycle by the caller
    (run_daemon) and passed down here — re-parsing the whole log per filter,
    every cycle, was the dominant cost of a cycle once the campaign had a
    few hundred completed experiments (master.log and the node-marker/
    success-marker dirs only grow), and was the direct cause of the
    growing gaps between jobs finishing and the next one being submitted.
    Purely a perf fix: no stage, ordering, or idempotency logic changed."""

    reps_done = success_marker_reps(success_markers_dir, filter_name)

    # Only truly COMPLETED once cleanup (Script 4) has ALSO succeeded — all
    # repetitions having a success marker is necessary but not sufficient;
    # returning "COMPLETED" before that would short-circuit this function
    # and Script 4 would never get submitted (confirmed via a real end-to-
    # end validation run: cleanup was silently skipped until this check was
    # tightened).
    if len(reps_done) >= NUM_REPETITIONS:
        cleanup_status = last_stage_status(events, filter_name, "cleanup")
        if cleanup_status == "SUCCESS":
            return "COMPLETED"
        if filter_name in failed["cleanup"]:
            return "COMPLETED (cleanup failed — dataset/TFRecords left on SSD, see failed_jobs.json)"
        # fall through to the cleanup-submission block near the end of this
        # function (dataset/tfrecord/train/eval are all already done, so the
        # checks below are no-ops until that block is reached)

    if filter_name in failed["filters"]:
        return "FAILED (dataset/tfrecord — see failed_jobs.json)"

    node = get_node_assignment(filter_name)

    # --- Stage: dataset (Script 1) ---------------------------------------
    if node is None:
        inflight = count_inflight_filters(success_markers_dir, failed, events, scope_filters=matrix_filters)
        if inflight >= MAX_INFLIGHT_FILTERS:
            return (f"dataset: deferred ({inflight}/{MAX_INFLIGHT_FILTERS} filters already "
                     f"in-flight — bounds worst-case SSD usage, see MAX_INFLIGHT_FILTERS)")
        job_name = f"hcpa_dataset_{filter_name}"
        if job_name in queued_names:
            return "dataset: queued/running (node not yet known)"
        if budget[0] <= 0:
            return f"dataset: deferred (queue at capacity, max {MAX_CONCURRENT_CAMPAIGN_JOBS} campaign jobs)"
        job_id = submitter.submit_job(
            submitter.generate_script_1(filter_name, base_ssd, base_home),
            job_name, use_gpu=False, time_limit="02:00:00", nodelist=None,
        )
        if job_id:
            budget[0] -= 1
            log_master_event(filter_name, "pending", "dataset", f"SUBMITTED(job={job_id})")
            return "dataset: submitted (no node pin — SLURM will choose)"
        return "dataset: submission failed, will retry next cycle"

    dataset_status = last_stage_status(events, filter_name, "dataset")
    if dataset_status != "SUCCESS":
        job_name = f"hcpa_dataset_{filter_name}"
        if job_name in queued_names:
            return f"dataset: running on {node}"
        if dataset_status == "FAILED":
            record_failure(failed, filter_name, "dataset", None, "Script 1 failed — see log_hint")
            return "FAILED (dataset)"
        # Node marker exists (job started) but no STARTED/SUCCESS/FAILED
        # resolution and the job has left the queue: a hard crash (signal-
        # level abort, uncatchable in Python — the same class of bug found
        # and fixed for training in a previous audit) is the most likely
        # explanation. Treat as failed rather than silently stalling forever.
        record_failure(failed, filter_name, "dataset", None,
                        "job left the queue with no SUCCESS/FAILED event logged (possible hard crash)")
        return "FAILED (dataset, no completion signal)"

    # --- Stage: tfrecord --------------------------------------------------
    tfrecord_status = last_stage_status(events, filter_name, "tfrecord")
    if tfrecord_status != "SUCCESS":
        job_name = f"hcpa_tfrecord_{filter_name}"
        if job_name in queued_names:
            return f"tfrecord: running on {node}"
        if tfrecord_status == "FAILED":
            record_failure(failed, filter_name, "tfrecord", None, "TFRecord generation failed — see log_hint")
            return "FAILED (tfrecord)"
        if tfrecord_status == "STARTED":
            # started per log but job no longer queued and not yet resolved -> crashed
            record_failure(failed, filter_name, "tfrecord", None,
                            "job left the queue with no SUCCESS/FAILED event logged (possible hard crash)")
            return "FAILED (tfrecord, no completion signal)"
        if budget[0] <= 0:
            return f"tfrecord: deferred (queue at capacity, max {MAX_CONCURRENT_CAMPAIGN_JOBS} campaign jobs)"
        if node_balance_blocks(node, nodes_busy, enforce_balance):
            return f"tfrecord: deferred (node balance — {node} busy, other node free)"
        job_id = submitter.submit_job(
            submitter.generate_script_tfrecord(filter_name, base_ssd, base_home),
            job_name, use_gpu=False, time_limit="01:00:00", nodelist=node,
        )
        if job_id:
            budget[0] -= 1
            nodes_busy.add(node)
            return f"tfrecord: submitted on {node}"
        return "tfrecord: submission failed (sbatch error — see dispatcher log), will retry next cycle"

    # --- Stage: train+eval per missing repetition -------------------------
    reps_missing = [r for r in target_rep_range() if r not in reps_done]
    submitted_any = False
    submitted_status = None
    for rep in reps_missing:
        rep_key = f"{filter_name}_{rep}"
        if rep_key in failed["repetitions"]:
            continue  # permanently given up (retries exhausted or manual set-aside)
        job_name = f"hcpa_train_{filter_name}_{rep}"
        if job_name in queued_names:
            continue  # currently queued/running — leave it alone

        # Bounded auto-retry for train/eval reps (requested: recover the
        # intermittent GPU/driver crash instead of leaving ~800 holes). A rep
        # still in reps_missing, not in the queue, but with >=1 prior launch
        # (train STARTED event) means that launch failed — a hard crash, a
        # Script 2/3 error, or a timeout. Resubmit it until it succeeds or
        # MAX_TRAIN_ATTEMPTS is reached; the attempt count comes straight
        # from master.log (count_stage_attempts) so it's idempotent and
        # restart-safe. A genuinely deterministic failure just fails every
        # attempt and is then recorded permanent below — bounded, never a
        # loop.  NOTE: dataset/tfrecord failures are intentionally NOT retried
        # here (only train/eval reps) — they keep their existing manual
        # handling.
        train_status = last_stage_status(events, filter_name, "train", rep)
        eval_status = last_stage_status(events, filter_name, "eval", rep)
        attempts = count_stage_attempts(events, filter_name, "train", rep)
        retry_reason = None
        submit_node = node
        if attempts >= 1:
            retry_reason = _train_failure_reason(train_status, eval_status)
            if attempts >= MAX_TRAIN_ATTEMPTS:
                # Exhausted the budget — record permanent NOW (independent of
                # queue/node budget, so a genuinely-broken rep is retired
                # promptly and never loops).
                record_failure(failed, filter_name, "train", rep,
                                f"{retry_reason} — gave up after {attempts}/{MAX_TRAIN_ATTEMPTS} attempts")
                continue
            # else: this is a retry — but only LOG it as "resubmitting" at the
            # actual submission point below (not here), so a rep that defers
            # to a later cycle for queue/node budget doesn't emit a
            # misleading "resubmitting" line every cycle while it waits.
            # Route retries to the OTHER grace node when possible (see
            # other_grace_node()'s docstring): master.log analysis showed
            # failures cluster in temporary per-node bad streaks, and
            # pinning every retry to the same node that just failed was
            # turning those streaks into near-certain permanent losses
            # instead of independent second chances. Only the train
            # submission below is affected — dataset/tfrecord/cleanup keep
            # using `node` (the original pin) everywhere else in this
            # function, unchanged.
            submit_node = other_grace_node(node) or node
        if budget[0] <= 0:
            # Queue at capacity — stop submitting entirely (not just this
            # rep) rather than looping through the remaining missing reps;
            # they're retried next cycle like any other deferred submission.
            break
        if node_balance_blocks(submit_node, nodes_busy, enforce_balance):
            # NOTE: no longer safe to `break` here unconditionally — with
            # retry diversification above, different reps in this loop can
            # now target different nodes (a fresh rep stays pinned to
            # `node`, a retry targets the other one), so a block on this
            # rep's node doesn't mean every remaining rep is equally
            # blocked. `continue` to let the loop still try the next rep.
            continue
        job_id = submitter.submit_job(
            submitter.generate_script_2_3(filter_name, rep, base_ssd, base_home, time_limit="10:00:00"),
            job_name, use_gpu=True, nodelist=submit_node,
        )
        if job_id:
            submitted_any = True
            budget[0] -= 1
            nodes_busy.add(submit_node)
            if attempts >= 1:
                # Full retry audit line, emitted exactly once per actual
                # resubmission: filter, rep, attempt number, reason for the
                # prior failure, and the new job id. The RESULT of this retry
                # then appears as that job's own SUCCESS/FAILED in master.log
                # (and, on success, its resultados/ CSV).
                diversion = f" (diverted from {node})" if submit_node != node else ""
                logger.warning(
                    f"RETRY submitted: filter={filter_name} rep={rep} job={job_id} on {submit_node}{diversion} "
                    f"— attempt {attempts + 1}/{MAX_TRAIN_ATTEMPTS} (prior failure: {retry_reason})"
                )
                submitted_status = (f"train/eval: RESUBMITTED rep {rep} "
                                    f"(retry {attempts + 1}/{MAX_TRAIN_ATTEMPTS}) on {submit_node}{diversion}")
            else:
                submitted_status = f"train/eval: submitted next missing repetition on {submit_node}"
            break  # one job per filter per cycle — see process_filter()'s docstring
        # else: sbatch failed (already logged inside submit_job) — leave
        # queued_names/master.log untouched so this exact rep is retried
        # next cycle, instead of silently reporting "submitted" for
        # something that never actually got queued.

    if submitted_any:
        return submitted_status or f"train/eval: submitted next missing repetition on {node}"

    # --- Stage: cleanup (only once genuinely 10/10; unreachable if any rep
    #     permanently failed — script4 itself also refuses in that case) ---
    reps_done = success_marker_reps(success_markers_dir, filter_name)  # re-check after this cycle's submissions
    if len(reps_done) >= NUM_REPETITIONS:
        job_name = f"hcpa_cleanup_{filter_name}"
        cleanup_status = last_stage_status(events, filter_name, "cleanup")
        if cleanup_status == "SUCCESS":
            return "COMPLETED"
        if job_name in queued_names:
            return f"cleanup: running on {node}"
        if cleanup_status == "FAILED":
            # Circuit-breaker: without this, a genuinely failing cleanup
            # (confirmed via real validation: script4 legitimately refuses
            # when fewer than NUM_REPETITIONS markers exist, e.g. during a
            # reduced-repetition test) gets resubmitted every single poll
            # cycle forever. The science is already done at this point (all
            # repetitions evaluated) — only the disk cleanup didn't happen —
            # so this is recorded separately from a real filter failure.
            record_failure(failed, filter_name, "cleanup", None,
                            "Script 4 (cleanup) failed — datasets_filtrados/TFRecords left on SSD; "
                            "science for this filter is unaffected. See log_hint.")
            return "COMPLETED (cleanup failed — see failed_jobs.json)"
        if budget[0] <= 0:
            return f"cleanup: deferred (queue at capacity, max {MAX_CONCURRENT_CAMPAIGN_JOBS} campaign jobs)"
        if node_balance_blocks(node, nodes_busy, enforce_balance):
            return f"cleanup: deferred (node balance — {node} busy, other node free)"
        job_id = submitter.submit_job(
            submitter.generate_script_4(filter_name, base_ssd, base_home),
            job_name, use_gpu=False, time_limit="00:30:00", nodelist=node,
        )
        if job_id:
            budget[0] -= 1
            nodes_busy.add(node)
            return f"cleanup: submitted on {node}"
        return "cleanup: submission failed (sbatch error — see dispatcher log), will retry next cycle"

    return f"waiting: {len(reps_done)}/{NUM_REPETITIONS} repetitions done, rest queued/running on {node}"


def filter_is_resolved(filter_name: str, success_markers_dir: Path, failed: dict, events: List[dict]) -> bool:
    """True if nothing more can or will happen automatically for this filter
    (fully completed AND cleaned up, or every still-missing repetition has
    been marked failed, or the dataset/tfrecord step itself failed).

    Requiring cleanup to have actually SUCCEEDED (not just all repetitions
    having a marker) is essential — otherwise the daemon considers a filter
    "resolved" the instant its last repetition finishes and Script 4 never
    gets a chance to run (confirmed via a real end-to-end validation run).

    `events` is master.log parsed once per cycle by the caller (see
    process_filter's docstring) — not re-read here.

    Strictly scoped to target_rep_range() (the CURRENT run's own window),
    deliberately -- this must say "not resolved" for a mini-campaign
    filter that already completed reps 0-9 in the main campaign, since
    THIS run's job (reps 10-19) is genuinely not done yet. Do not widen
    this to also accept the original 0-9 window "for convenience": that
    was tried and immediately made every one of the mini-campaign's 10
    filters (all of which, by construction, were pulled from the ALREADY
    COMPLETED main campaign) look instantly resolved, so run_daemon
    exited after 1/100 conditions with "All filters resolved" (confirmed
    live, job 815643). The global-disk-usage scan in
    count_inflight_filters() has a different, looser notion of "resolved"
    (a filter is off node-local SSD if it's done in EITHER window) — that
    lives in _filter_disk_resolved() below, kept deliberately separate."""
    if filter_name in failed["filters"]:
        return True
    return _resolved_in_range(filter_name, success_markers_dir, failed, events, target_rep_range())


def _resolved_in_range(filter_name: str, success_markers_dir: Path, failed: dict, events: List[dict],
                        rep_range) -> bool:
    """Core of filter_is_resolved(), parameterized by an explicit
    repetition window instead of always using target_rep_range() (i.e.
    REP_OFFSET-independent) -- lets a caller ask "is this filter resolved
    with respect to THIS SPECIFIC window" for a window other than the
    dispatcher's current one. Needed because the common real completion
    state for a main-campaign filter is NOT all-10-reps-plus-cleanup: it's
    9/10 reps done with the 10th permanently failed, which script4
    (cleanup) refuses to touch by design (see process_filter's cleanup
    stage comment) -- that filter is legitimately "resolved" (nothing
    more will ever happen for it) but its dataset/TFRecords are never
    cleaned off disk. Recognizing that requires checking the "every
    missing rep is permanently failed" fallback against the SAME window
    the reps actually live in, not whatever REP_OFFSET the current run
    happens to be using."""
    reps_done = _reps_done_in_range(success_markers_dir, filter_name, rep_range)
    if len(reps_done) >= NUM_REPETITIONS:
        if filter_name in failed["cleanup"]:
            return True  # science done; cleanup permanently failed — terminal either way
        return last_stage_status(events, filter_name, "cleanup") == "SUCCESS"
    missing = [r for r in rep_range if r not in reps_done]
    return all(f"{filter_name}_{r}" in failed["repetitions"] for r in missing)


def _filter_disk_resolved(filter_name: str, success_markers_dir: Path, failed: dict, events: List[dict]) -> bool:
    """Like filter_is_resolved(), but for count_inflight_filters()'s
    purpose only: "is this filter's dataset/TFRecords off node-local SSD
    (or never going to be touched again automatically), regardless of
    which repetition window that happened in?" A filter counts as
    resolved here if EITHER the original reps 0-9 window OR the current
    target_rep_range() window resolves it (each checked via
    _resolved_in_range, so the "missing reps permanently failed"
    fallback works correctly in the window the failures actually
    occurred in — see that function's docstring) -- unlike
    filter_is_resolved(), which must stay strict to target_rep_range()
    alone (see its docstring for why widening THAT one broke the
    mini-campaign's own completion check). This function only answers
    the disk-usage-bound question; it must never be used to decide
    whether a run's own work is finished."""
    if filter_name in failed["filters"]:
        return True
    return (_resolved_in_range(filter_name, success_markers_dir, failed, events, range(0, NUM_REPETITIONS))
            or _resolved_in_range(filter_name, success_markers_dir, failed, events, target_rep_range()))


# --------------------------------------------------------------------------
# Node-down detection & automatic migration (no code edits required)
# --------------------------------------------------------------------------

def check_node_health_and_migrate(partition: str, down_timeout: int, user: str):
    states = sinfo_node_states(partition)
    if not states:
        return
    down_since = {}
    if NODE_DOWN_SINCE_PATH.exists():
        try:
            down_since = json.loads(NODE_DOWN_SINCE_PATH.read_text())
        except Exception:
            down_since = {}

    now = time.time()
    for node, state in states.items():
        is_down = any(tag in state.lower() for tag in ("down", "drain", "fail", "maint"))
        if is_down:
            first_seen = down_since.get(node, now)
            down_since[node] = first_seen
            elapsed = now - first_seen
            if elapsed > down_timeout:
                logger.warning(f"Node {node} has been '{state}' for {elapsed:.0f}s (> {down_timeout}s) — "
                                f"migrating any filter still assigned to it")
                for marker_file in sorted(NODE_MARKER_DIR.glob("*.node")):
                    try:
                        assigned_node = marker_file.read_text().strip()
                    except Exception:
                        continue
                    if assigned_node != node:
                        continue
                    filter_name = marker_file.stem
                    for stage_prefix in ("hcpa_dataset_", "hcpa_tfrecord_", "hcpa_cleanup_"):
                        subprocess.run(["scancel", "-u", user, "-n", f"{stage_prefix}{filter_name}"],
                                        capture_output=True, timeout=15)
                    for rep in target_rep_range():
                        subprocess.run(["scancel", "-u", user, "-n", f"hcpa_train_{filter_name}_{rep}"],
                                        capture_output=True, timeout=15)
                    marker_file.unlink(missing_ok=True)
                    log_master_event(filter_name, node, "migration", f"NODE_{node}_DOWN_REASSIGNING")
                    logger.warning(f"  -> reassigned filter '{filter_name}' (will resubmit Script 1 "
                                    f"unpinned on the next cycle)")
                down_since.pop(node, None)
        else:
            down_since.pop(node, None)

    NODE_DOWN_SINCE_PATH.write_text(json.dumps(down_since, indent=2))


# --------------------------------------------------------------------------
# Progress reporting
# --------------------------------------------------------------------------

def estimate_avg_rep_seconds(events: List[dict]) -> Optional[float]:
    """Average wall-clock duration of completed train+eval stages, derived
    purely from master.log timestamps (the only cross-node-visible timing
    source available, since per-run timing metadata lives on node-local SSD)."""
    starts: Dict[Tuple[str, str, str], datetime] = {}
    durations = []
    for e in events:
        key = (e["filter"], e["stage"], e["rep"])
        ts = datetime.strptime(e["ts"], "%Y-%m-%d %H:%M:%S")
        if e["status"] == "STARTED":
            starts[key] = ts
        elif e["status"] == "SUCCESS" and key in starts:
            durations.append((ts - starts[key]).total_seconds())
    if not durations:
        return None
    return sum(durations) / len(durations)


def _format_bytes(num_bytes: float) -> str:
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def home_disk_status(home_base: Path) -> Optional[str]:
    """
    Cheap ($HOME free/used/total via a single statvfs call, not a directory
    walk — trash_datasets/ alone can reach the TB range over a full 576x10
    campaign, so summing directory sizes every poll cycle would be its own
    performance problem) visibility check for the one filesystem the
    dispatcher itself can actually see (SSD_BASE is node-local scratch on
    grace1/grace2, invisible to the dispatcher which runs on `shared`).
    """
    try:
        usage = shutil.disk_usage(str(home_base))
        pct_used = 100.0 * usage.used / usage.total
        return (f"{_format_bytes(usage.used)} used / {_format_bytes(usage.total)} total "
                f"({pct_used:.1f}%), {_format_bytes(usage.free)} free")
    except Exception as e:
        return f"(could not stat {home_base}: {e})"


def write_progress_report(filter_matrix: FilterMatrix, success_markers_dir: Path, failed: dict,
                           partition: str, statuses: Dict[str, str], home_base: Optional[Path] = None):
    filters = [f["name"] for f in filter_matrix.get_filters()]
    filters_set = set(filters)
    total_reps = len(filters) * NUM_REPETITIONS
    done_reps = sum(len(success_marker_reps(success_markers_dir, f)) for f in filters)
    # failed["repetitions"]/failed["filters"] are the SAME global
    # failed_jobs.json used by every run (main campaign included) -- for a
    # small custom --matrix-file (e.g. the XAI mini-campaign), most of its
    # entries belong to filters that aren't even in THIS run's matrix, so
    # both counts are scoped down to `filters_set` here. Without this, a
    # mini-campaign inherits the main campaign's full historical failure
    # count and this report's "remaining" clamps to 0 even with real work
    # still pending (cosmetic only -- process_filter()'s own per-filter
    # logic never used this global count, so nothing was actually stuck).
    failed_reps = sum(1 for k in failed["repetitions"] if k.rsplit("_", 1)[0] in filters_set)
    failed_reps += sum(
        NUM_REPETITIONS - len(success_marker_reps(success_markers_dir, f))
        for f in failed["filters"] if f in filters_set
    )
    remaining_reps = max(total_reps - done_reps - failed_reps, 0)

    events = _read_master_log_lines()
    avg = estimate_avg_rep_seconds(events)
    busy_nodes = squeue_busy_nodes(partition)
    node_states = sinfo_node_states(partition)

    lines = []
    lines.append(f"Relatório de progresso — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 60)
    lines.append(f"Completed: {done_reps} / {total_reps}")
    lines.append(f"Remaining: {remaining_reps}")
    lines.append(f"Failed repetitions: {failed_reps}")
    if avg and remaining_reps > 0:
        # up to 2 concurrent repetitions in flight (1 GPU per node, 2 nodes)
        concurrency = max(1, len([n for n in node_states if node_states.get(n, "").lower() not in ("down", "drain")]))
        eta_seconds = (remaining_reps * avg) / concurrency
        eta = timedelta(seconds=int(eta_seconds))
        lines.append(f"Estimated remaining time: {eta}")
    else:
        lines.append("Estimated remaining time: (not enough data yet)")
    lines.append("")
    if home_base is not None:
        lines.append(f"$HOME disk usage ({home_base}): {home_disk_status(home_base)}")
        lines.append("  (SSD_BASE is node-local on grace1/grace2 — not visible from here; "
                      "see documentation/METHODOLOGICAL_NOTES.md for the pre-campaign capacity estimate)")
        lines.append("")
    lines.append("Current node usage:")
    for node in sorted(node_states.keys()):
        state = node_states[node]
        gpu = "GPU busy" if node in busy_nodes else "GPU free"
        lines.append(f"  {node}: {state} ({gpu})")
    lines.append("")
    lines.append("Per-filter status:")
    for f in filters:
        lines.append(f"  {f}: {statuses.get(f, '?')}")
    lines.append("")

    PROGRESS_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Progress: {done_reps}/{total_reps} done, {remaining_reps} remaining, {failed_reps} failed")


def write_final_report(filter_matrix: FilterMatrix, success_markers_dir: Path, failed: dict,
                       resultados_dir: Path, start_time: float):
    filters = [f["name"] for f in filter_matrix.get_filters()]
    completed_filters = [f for f in filters if len(success_marker_reps(success_markers_dir, f)) >= NUM_REPETITIONS]
    failed_filters = list(failed["filters"].keys())
    partial_filters = [f for f in filters if f not in completed_filters and f not in failed_filters]

    events = _read_master_log_lines()
    total_gpu_seconds = 0.0
    starts: Dict[Tuple[str, str, str], datetime] = {}
    for e in events:
        if e["stage"] not in ("train", "eval"):
            continue
        key = (e["filter"], e["stage"], e["rep"])
        ts = datetime.strptime(e["ts"], "%Y-%m-%d %H:%M:%S")
        if e["status"] == "STARTED":
            starts[key] = ts
        elif e["status"] == "SUCCESS" and key in starts:
            total_gpu_seconds += (ts - starts[key]).total_seconds()

    elapsed = timedelta(seconds=int(time.time() - start_time))
    lines = [
        "RELATÓRIO FINAL DA CAMPANHA EXPERIMENTAL",
        "=" * 60,
        f"Gerado em: {datetime.now().isoformat()}",
        f"Tempo total de execução do orquestrador: {elapsed}",
        f"GPU-horas consumidas (soma train+eval por repetição): {total_gpu_seconds / 3600:.2f}h",
        "",
        f"Filtros concluídos (10/10 repetições): {len(completed_filters)}",
        *[f"  - {f}" for f in completed_filters],
        "",
        f"Filtros com falha (nível dataset/tfrecord): {len(failed_filters)}",
        *[f"  - {f}" for f in failed_filters],
        "",
        f"Filtros parciais/incompletos (alguma repetição falhou): {len(partial_filters)}",
        *[f"  - {f}" for f in partial_filters],
        "",
        f"Repetições individuais marcadas como falha: {len(failed['repetitions'])}",
        "",
        "Localização dos artefatos:",
        f"  CSVs de métricas:      {resultados_dir}/*.csv",
        f"  Success markers:       {resultados_dir}/_success_markers/",
        f"  Logs (por job):        {REPO_ROOT_DIR}/logs/*.log e *.err",
        f"  Log mestre:            {MASTER_LOG_PATH}",
        f"  Falhas estruturadas:   {FAILED_JOBS_PATH}",
        "",
    ]
    FINAL_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    FINAL_REPORT_PATH.write_text("\n".join(lines))
    logger.info(f"Final report written to {FINAL_REPORT_PATH}")


# --------------------------------------------------------------------------
# Self-resubmission near the partition's time limit
# --------------------------------------------------------------------------

def seconds_until_slurm_deadline() -> Optional[float]:
    end_time = os.environ.get("SLURM_JOB_END_TIME") if os.environ.get("SLURM_JOB_ID") else None
    if not end_time:
        return None
    try:
        return float(end_time) - time.time()
    except ValueError:
        return None


def self_resubmit_and_exit(wrapper_script: Path, logger):
    logger.warning("Approaching this job's time limit with work still pending — "
                    "resubmitting a fresh dispatcher and exiting cleanly.")
    try:
        subprocess.run(["sbatch", str(wrapper_script)], cwd=REPO_ROOT_DIR, timeout=30)
    except Exception as e:
        logger.error(f"Self-resubmission failed: {e} — the campaign will stall until "
                      f"someone manually resubmits {wrapper_script}")
    sys.exit(0)


# --------------------------------------------------------------------------
# Main daemon loop
# --------------------------------------------------------------------------

def run_daemon(filter_matrix: FilterMatrix, submitter: SLURMJobSubmitter, base_ssd: str, base_home: str,
               paths: dict, poll_interval: int, node_down_timeout: int, wrapper_script: Path):
    user = os.environ.get("USER", "")
    success_markers_dir = paths["success_markers"]
    resultados_dir = paths["resultados"]
    start_time = time.time()

    logger.info("=" * 80)
    logger.info("HCPA orchestrator daemon starting")
    logger.info(f"  Filters in matrix: {len(filter_matrix.get_filters())}")
    logger.info(f"  SSD base: {base_ssd}")
    logger.info(f"  HOME base: {base_home}")
    logger.info(f"  Poll interval: {poll_interval}s")
    logger.info("=" * 80)

    # Guard against two daemon instances running concurrently (e.g. `sbatch
    # run_all_experiments.sh` fired twice by accident). Without this, two
    # independent squeue-then-submit cycles can interleave in the same race
    # window (both see a repetition as "not yet queued" and both submit it),
    # wasting GPU time on a duplicate run — the underlying scripts are
    # idempotent so this wouldn't corrupt results, but it's a real, avoidable
    # waste. Reusing SLURM's own job table as the lock (rather than a
    # separate lockfile) means it's automatically released if this job dies.
    my_job_id = os.environ.get("SLURM_JOB_ID")
    # SLURM_JOB_NAME, not a hardcoded "hcpa_orchestrator": this daemon also
    # runs under other names (e.g. "hcpa_xai_mini_orchestrator" for the XAI
    # mini-campaign, see run_xai_mini_campaign.sh's #SBATCH --job-name).
    # Hardcoding the main campaign's name meant this guard silently never
    # caught a duplicate mini-campaign dispatcher -- confirmed live: two
    # "hcpa_xai_mini_orchestrator" instances (815646, 815653) ran
    # concurrently for several minutes before this was caught by hand.
    my_job_name = os.environ.get("SLURM_JOB_NAME", "hcpa_orchestrator")
    if my_job_id:
        try:
            result = subprocess.run(
                ["squeue", "-u", user, "-n", my_job_name, "-h", "-o", "%i"],
                capture_output=True, text=True, timeout=15,
            )
            other_ids = [j for j in result.stdout.split() if j and j != my_job_id]
            if other_ids:
                logger.error(
                    f"Another {my_job_name} job is already running (job id(s): {other_ids}) — "
                    f"exiting to avoid duplicate submissions. Cancel it first if it's stale "
                    f"(scancel {' '.join(other_ids)})."
                )
                sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not check for a concurrent orchestrator instance ({e}) — proceeding anyway.")

    failed = load_failed_jobs()
    iteration = 0
    while True:
        iteration += 1
        # Parsed once per cycle and threaded through every call below
        # (process_filter, filter_is_resolved, count_inflight_filters)
        # instead of each of them re-reading/re-parsing master.log from
        # scratch per filter (576x/cycle) — that per-filter re-parse was the
        # dominant cost of a cycle once the campaign had a few hundred
        # completed experiments, since master.log only grows, and was the
        # direct cause of the growing gaps between a job finishing and the
        # next one being submitted. Purely a perf fix — no stage, ordering,
        # or idempotency logic changed.
        events = _read_master_log_lines()
        remaining = seconds_until_slurm_deadline()
        if remaining is not None and remaining < SELF_RESUBMIT_MARGIN:
            filters_left = any(
                not filter_is_resolved(f["name"], success_markers_dir, failed, events)
                for f in filter_matrix.get_filters()
            )
            if filters_left:
                self_resubmit_and_exit(wrapper_script, logger)
            else:
                logger.info("Time limit approaching but no work remains — exiting normally.")
                break

        # Everything below is best-effort per-iteration work: a single
        # uncaught exception anywhere here (a transient NFS hiccup, a stray
        # JSON write failure, a squeue/sinfo hiccup that escaped its own
        # try/except) must never kill the whole daemon — that would silently
        # stall a campaign meant to run unattended for days, with no
        # auto-recovery (unlike the 24h self-resubmit path, which only
        # triggers near the time limit, not on a crash). Log and retry next
        # cycle instead of propagating.
        try:
            check_node_health_and_migrate(submitter.partition, node_down_timeout, user)

            # Jobs stuck PENDING behind a node another cluster user is
            # occupying are actively cancelled here (not merely excluded
            # from the budget count below — see
            # cancel_stuck_pending_campaign_jobs()'s docstring for why that
            # earlier approach had no ceiling and let the queue grow
            # unbounded). Must run BEFORE squeue_job_names() so the budget
            # calculation right after reflects the queue as it actually is,
            # post-cancellation.
            cancel_stuck_pending_campaign_jobs(user)

            queued_names = squeue_job_names(user)
            # Sliding-window queue cap (requested explicitly): recomputed
            # fresh every cycle from the live queue, so it's correct even if
            # jobs finished/were cancelled outside this daemon (e.g. by hand)
            # between cycles. budget[0] is shared (and decremented) across
            # every filter processed below, so the TOTAL number of jobs this
            # cycle submits, added to what's already queued, never exceeds
            # MAX_CONCURRENT_CAMPAIGN_JOBS — see process_filter()'s docstring.
            budget = [max(0, MAX_CONCURRENT_CAMPAIGN_JOBS - count_active_campaign_jobs(queued_names))]
            # Seed nodes_busy from jobs already queued/running BEFORE this
            # cycle (not from anything submitted below — that's added as it
            # happens), so pass 1 already knows e.g. "grace1 has a job" from
            # last cycle and gives grace2-pinned filters first refusal at
            # the remaining budget instead of stacking a 2nd job on grace1.
            nodes_busy = {
                get_node_assignment(f["name"]) for f in filter_matrix.get_filters()
                if get_node_assignment(f["name"]) and filter_has_active_job(f["name"], queued_names)
            }
            statuses = {}
            balance_deferred = []
            matrix_filters = {f["name"] for f in filter_matrix.get_filters()}
            for filter_def in filter_matrix.get_filters():
                name = filter_def["name"]
                try:
                    status = process_filter(name, submitter, base_ssd, base_home,
                                             success_markers_dir, queued_names, failed,
                                             budget, nodes_busy, events, enforce_balance=True,
                                             matrix_filters=matrix_filters)
                except Exception as e:
                    logger.error(f"Unexpected error processing filter '{name}': {e} — will retry next cycle")
                    status = f"ERROR (dispatcher-side): {e}"
                statuses[name] = status
                if "node balance" in status:
                    balance_deferred.append(name)

            # Pass 2 (only runs if pass 1 left budget unused because no
            # filter targeted the idle node): retry exactly the filters
            # pass 1 deferred for balance, this time without the balance
            # preference, so leftover budget is never wasted/deadlocked
            # waiting for a node that genuinely has no pending work.
            if budget[0] > 0 and balance_deferred:
                for name in balance_deferred:
                    if budget[0] <= 0:
                        break
                    try:
                        statuses[name] = process_filter(name, submitter, base_ssd, base_home,
                                                          success_markers_dir, queued_names, failed,
                                                          budget, nodes_busy, events, enforce_balance=False,
                                                          matrix_filters=matrix_filters)
                    except Exception as e:
                        logger.error(f"Unexpected error processing filter '{name}' (balance-fallback pass): "
                                     f"{e} — will retry next cycle")
                        statuses[name] = f"ERROR (dispatcher-side): {e}"

            write_progress_report(filter_matrix, success_markers_dir, failed, submitter.partition, statuses,
                                   home_base=Path(base_home))

            all_resolved = all(
                filter_is_resolved(f["name"], success_markers_dir, failed, events)
                for f in filter_matrix.get_filters()
            )
            if all_resolved:
                logger.info("All filters resolved (completed or permanently failed) — writing final report.")
                write_final_report(filter_matrix, success_markers_dir, failed, resultados_dir, start_time)
                break
        except Exception as e:
            logger.error(f"Unexpected error in daemon loop iteration {iteration}: {e} — "
                         f"will retry next cycle rather than crash the daemon", exc_info=True)

        time.sleep(poll_interval)

    logger.info("Orchestrator daemon exiting.")


def main():
    parser = argparse.ArgumentParser(description="Script 5: single-entry-point orchestrator daemon")
    parser.add_argument("--matrix-file", type=str, default=None,
                        help="Path to filter matrix JSON (default: experiments/filter_matrix_template.json)")
    parser.add_argument("--base-ssd", type=str, default=None,
                        help="Base SSD path (defaults to $SSD_BASE env var or the repo directory)")
    parser.add_argument("--base-home", type=str, default=None, help="Base home path (defaults to $HOME)")
    parser.add_argument("--partition", type=str, default="grace",
                        help="SLURM partition for the actual pipeline jobs (script1-4)")
    parser.add_argument("--poll-interval", type=int, default=POLL_INTERVAL_DEFAULT,
                        help="Seconds between dispatcher loop iterations")
    parser.add_argument("--node-down-timeout", type=int, default=NODE_DOWN_TIMEOUT_DEFAULT,
                        help="Seconds a node may stay down/drain before its filters are migrated")
    parser.add_argument("--xai-methods", type=str, default=None,
                        help="XAI methodology mini-campaign only: comma-separated methods "
                             "passed through to script3_avalia.py's --xai-methods (e.g. "
                             "'gradcam,lime,occlusion'). Unset (default) omits "
                             "the flag entirely, so script3_avalia.py uses its own default "
                             "(gradcam only) -- the main campaign never sets this.")
    parser.add_argument("--xai-subsample", type=int, default=None,
                        help="XAI methodology mini-campaign only: passed through to "
                             "script3_avalia.py's --xai-subsample. Unset (default) omits the "
                             "flag, so no subsampling happens (main campaign behavior).")
    parser.add_argument("--keep-checkpoint", action="store_true",
                        help="XAI methodology mini-campaign only: passed through to "
                             "script3_avalia.py's --keep-checkpoint. Off by default -- the "
                             "main campaign always deletes checkpoints after evaluation.")
    parser.add_argument("--rep-offset", type=int, default=0,
                        help="XAI methodology mini-campaign only: shifts the repetition window "
                             "this run targets from [0, NUM_REPETITIONS) to "
                             "[offset, offset+NUM_REPETITIONS) -- see REP_OFFSET's module-level "
                             "comment for why (avoids colliding with the main campaign's own "
                             "already-completed 0-9 for the same filter names). Default 0 is the "
                             "main campaign's own range, unchanged.")
    parser.add_argument("--wrapper-script", type=str, default="run_all_experiments.sh",
                        help="sbatch script self_resubmit_and_exit() resubmits when this "
                             "dispatcher's own SLURM time limit approaches with work still "
                             "pending -- must match whatever wrapper was used to submit THIS "
                             "run (default is the main campaign's; the XAI mini-campaign passes "
                             "its own wrapper here so a resubmit doesn't accidentally restart "
                             "the main campaign's dispatcher instead).")
    args = parser.parse_args()

    paths = get_paths(args.base_ssd, args.base_home)
    base_ssd = str(paths["base_ssd"])
    base_home = str(paths["base_home"])

    global logger, REP_OFFSET
    logger = setup_logging("script5_orquestrador_daemon")
    REP_OFFSET = args.rep_offset

    matrix_file = Path(args.matrix_file) if args.matrix_file else paths["filter_matrix_file"]
    filter_matrix = FilterMatrix(matrix_file, logger)

    submitter = SLURMJobSubmitter(partition=args.partition, logger=logger,
                                   xai_methods=args.xai_methods, xai_subsample=args.xai_subsample,
                                   keep_checkpoint=args.keep_checkpoint)
    wrapper_script = Path(REPO_ROOT_DIR) / args.wrapper_script

    run_daemon(filter_matrix, submitter, base_ssd, base_home, paths,
               args.poll_interval, args.node_down_timeout, wrapper_script)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise  # normal daemon exits (argparse --help, sys.exit(1) guard, etc.) — not a crash
    except Exception as e:
        import traceback
        # Every per-cycle exception inside run_daemon's own loop is already
        # caught and logged there (best-effort, keeps the daemon alive) --
        # reaching HERE means something escaped that (a setup-time crash
        # before the loop started, or a bug in the loop's own try/except
        # scaffolding itself) and the daemon is about to die entirely with
        # no automatic recovery. This is exactly the "can't be fixed
        # automatically, notify the operator" case requested explicitly.
        try:
            notify_operator(
                "daemon caiu (falha nao recuperavel)",
                f"O despachante SLURM_JOB_ID={os.environ.get('SLURM_JOB_ID', '?')} "
                f"encerrou com uma excecao nao tratada e NAO vai se recuperar sozinho -- "
                f"nada mais sera submetido ate reiniciar manualmente.\n\n"
                f"Erro: {e}\n\n{traceback.format_exc()}",
            )
        except Exception:
            pass  # notification itself must never mask the real crash below
        raise
