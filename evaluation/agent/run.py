"""Explicit live Agent evaluation; execution and scoring are saved separately."""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import signal
from uuid import uuid4

from .dataset import PROJECT, load_cases, select_cases, validate_cases
sys.path.insert(0, str(PROJECT/'src'))
from .runtime import execute_case, decisions_values
from .observe import PaidCallGate, ObservationSink
from .score import score_case, summarize
from .rules import check_rules


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_rows(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l]


def append_row(path, row):
    with Path(path).open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()


def load_agent_run(run_dir):
    home = Path(run_dir)
    m = json.loads((home/'manifest.json').read_text(encoding='utf-8'))
    for name, digest in m['files'].items():
        if file_hash(home/name) != digest:
            raise ValueError('artifact hash mismatch: '+name)
    cases = json.loads((home/'cases.json').read_text(encoding='utf-8'))
    validate_cases(cases)
    executions, scores = read_rows(home/'executions.jsonl'), read_rows(home/'scores/v1/scores.jsonl')
    expected = set(m['expected_case_ids'])
    if {c['case_id'] for c in cases} != expected:
        raise ValueError('case coverage mismatch')
    for rows in (executions, scores):
        ids = [r['case_id'] for r in rows]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError('result coverage mismatch')
    e_by = {r['case_id']: r for r in executions}
    if any(s['trial_id'] != e_by[s['case_id']]['trial_id'] for s in scores):
        raise ValueError('score execution identity mismatch')
    c_by = {c['case_id']:c for c in cases}
    if any(score_case(c_by[s['case_id']],e_by[s['case_id']],s['semantic']) != s for s in scores):
        raise ValueError('score policy mismatch')
    if summarize(cases, executions, scores) != json.loads((home/'summary.json').read_text(encoding='utf-8')):
        raise ValueError('summary mismatch')
    return m, cases, executions, scores


class EvaluationEngine:
    def __init__(self, cases, output, values, live):
        self.cases = {c['case_id']: c for c in cases}
        self.output, self.values, self.live = output, values, live
        self.gate, self.lock = PaidCallGate(), threading.Lock()
        self.started, self.executions = set(), []

    def invoke(self, case_id):
        with self.lock:
            if case_id in self.started:
                raise ValueError('refusing automatic full-task retry')
            self.started.add(case_id)
        case = self.cases[case_id]
        trial_id = case_id+'-'+uuid4().hex[:10]
        if self.gate.stopped.is_set() or (self.gate.blocked and not case['fault']):
            row = {'case_id': case_id, 'trial_id':trial_id, 'group':case['group'], 'execution_status':'not_executed',
                   'turns':[], 'route':[], 'events':[], 'runs':[], 'raw_records':[], 'usage':{}, 'latency_ms':None,
                   'stop_reason':'batch_paused_auth_or_credit_error','error':'credential_paused'}
        else:
            row = asyncio.run(execute_case(case, trial_id=trial_id, run_dir=self.output, values=self.values,
                                           live=self.live, gate=self.gate))
        with self.lock:
            self.executions.append(row)
            append_row(self.output/'executions.jsonl', row)
            append_row(self.output/'traces.jsonl', {'case_id':case_id,'trial_id':trial_id,'events':row['events']})
            print(json.dumps({'case_id':case_id,'status':row['execution_status'],'elapsed_ms':row['latency_ms'],
                              'completed_records':len(self.executions)}, ensure_ascii=False), flush=True)
        return row

    def record_unexecuted(self, case_id, reason):
        case = self.cases[case_id]
        row = {'case_id':case_id,'trial_id':case_id+'-'+uuid4().hex[:10], 'group':case['group'],
               'execution_status':'not_executed','turns':[],'route':[],'events':[],'runs':[],
               'raw_records':[],'usage':{},'latency_ms':None,'stop_reason':reason,'error':reason}
        with self.lock:
            self.executions.append(row)
            append_row(self.output/'executions.jsonl',row)
            append_row(self.output/'traces.jsonl',{'case_id':case_id,'trial_id':row['trial_id'],'events':[]})


def prepare_snapshot(values, output):
    from submission_assistant.runtime.rag import ExistingKnowledgeBackend
    backend = ExistingKnowledgeBackend(PROJECT, values)
    # ASCII-only location is necessary for this project's Windows HNSW build.
    target = PROJECT.parents[1]/'.cache/agent_eval_indices'/uuid4().hex
    target.parent.mkdir(parents=True, exist_ok=True)
    files = {str(p.relative_to(backend.chroma_dir)):file_hash(p) for p in backend.chroma_dir.rglob('*') if p.is_file()}
    shutil.copytree(backend.chroma_dir, target)
    copied = {str(p.relative_to(target)):file_hash(p) for p in target.rglob('*') if p.is_file()}
    if files != copied:
        raise ValueError('index changed while snapshotting')
    return {**values, '_EVAL_INDEX_PATH':str(target)}, {'version':backend.version,'source':str(backend.chroma_dir),'snapshot':str(target), 'files':files}


async def score_saved(cases, executions, values, output, gate):
    import httpx
    from submission_assistant.intent.jev import JevSettings
    from .judge import judge_semantics
    sink = ObservationSink('judge')
    e_by = {r['case_id']:r for r in executions}
    rows = []
    async with httpx.AsyncClient(trust_env=True, follow_redirects=False, event_hooks=sink.http_hooks(gate)) as client:
        for case in cases:
            e = e_by[case['case_id']]
            semantic = {'checks':{},'errors':[],'attempts':0}
            if case['semantic_checks'] and not gate.blocked and e['execution_status']=='completed':
                # Deterministically failed tasks need no extra paid verdict.
                if not any(r['status']=='fail' for r in check_rules(case,e)):
                    semantic = await judge_semantics(case,e,settings=JevSettings.from_values(decisions_values(values)),client=client)
                else:
                    semantic['errors'].append('rule_failed_judging_skipped')
            elif case['semantic_checks']:
                semantic['errors'].append('not_judged_due_to_execution_or_credential_state')
            row = score_case(case,e,semantic)
            rows.append(row)
            append_row(output/'scores.jsonl',row)
            print(json.dumps({'scored':case['case_id'],'outcome':row['business_outcome']},ensure_ascii=False),flush=True)
    write_json(output/'usage.json',{'http_events':sink.events,'totals':sink.totals('http_result')})
    return rows


def run_batch(cases, *, output, values, concurrency, live):
    from .evalscope_adapter import run_evalscope
    from .report import render_markdown
    validate_cases(cases)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('choose a new run directory')
    if not live and any(not c['fault'] for c in cases):
        raise ValueError('business tasks require --live')
    output.mkdir(parents=True)
    write_json(output/'cases.json',cases)
    manifest = {'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'complete':False,
        'execution_complete':False, 'expected_case_ids':[c['case_id'] for c in cases], 'concurrency':concurrency,
        'mode':'live' if live else 'deterministic_fault_only', 'judge_model':'pplx-decider-flash-v1.2',
        'runtime_model':values.get('LLM_MODEL'), 'review_status':'uncalibrated_agent_draft',
        'cache_mode':'per_trial_local_only; production Redis disabled', 'background_profile_maintenance':False,
        'files':{'cases.json':file_hash(output/'cases.json')}}
    write_json(output/'manifest.json',manifest)
    if live:
        from submission_assistant.runtime.context import build_runtime_config
        manifest['runtime_config']=build_runtime_config(values,PROJECT).metadata
        values, manifest['knowledge'] = prepare_snapshot(values, output)
        write_json(output/'manifest.json',manifest)
    engine = EvaluationEngine(cases,output,values,live)
    framework_error = None
    previous_handler = None
    if threading.current_thread() is threading.main_thread():
        previous_handler = signal.getsignal(signal.SIGINT)
        def stop_dispatch(signum, frame):
            engine.gate.stopped.set()
            raise KeyboardInterrupt()
        signal.signal(signal.SIGINT,stop_dispatch)
    try:
        run_evalscope(output,mode='live',concurrency=concurrency,runner=engine)
    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as exc:
        framework_error = type(exc).__name__
        engine.gate.stopped.set()
    finally:
        if previous_handler is not None:
            signal.signal(signal.SIGINT,previous_handler)
    missing = set(engine.cases)-{r['case_id'] for r in engine.executions}
    for case_id in sorted(missing):
        engine.record_unexecuted(case_id,'framework_interrupted_'+str(framework_error))
    manifest['framework_error'] = framework_error
    manifest['execution_complete'] = all(r['execution_status']!='not_executed' for r in engine.executions)
    scores_dir = output/'scores/v1'
    scores_dir.mkdir(parents=True)
    try:
        scores = asyncio.run(score_saved(cases,engine.executions,values,scores_dir,engine.gate))
    except (KeyboardInterrupt, asyncio.CancelledError):
        engine.gate.stopped.set()
        framework_error = 'interrupted_during_scoring'
        scores = read_rows(scores_dir/'scores.jsonl') if (scores_dir/'scores.jsonl').exists() else []
        known={r['case_id'] for r in scores}
        e_by={e['case_id']:e for e in engine.executions}
        for c in cases:
            if c['case_id'] not in known:
                s=score_case(c,e_by[c['case_id']],{'checks':{},'errors':['scoring_interrupted'],'attempts':0})
                scores.append(s)
                append_row(scores_dir/'scores.jsonl',s)
    summary = summarize(cases,engine.executions,scores)
    write_json(output/'summary.json',summary)
    (output/'report.zh.md').write_text(render_markdown(summary,engine.executions,scores),encoding='utf-8')
    write_json(scores_dir/'manifest.json',{'complete':True,'rubric_version':'agent-rubric-v1-uncalibrated',
        'executions_sha256':file_hash(output/'executions.jsonl'),'scores_sha256':file_hash(scores_dir/'scores.jsonl')})
    run_evalscope(output,mode='replay',concurrency=1)
    html = list((output/'evalscope/replay').rglob('report.html'))
    if not html:
        raise ValueError('native HTML report missing')
    manifest['files'] = {name:file_hash(output/name) for name in
        ['cases.json','executions.jsonl','traces.jsonl','scores/v1/scores.jsonl','scores/v1/manifest.json','summary.json']}
    manifest['native_reports'] = [str(p.relative_to(output)) for p in html]
    if 'knowledge' in manifest:
        source = Path(manifest['knowledge']['source'])
        current = {str(p.relative_to(source)):file_hash(p) for p in source.rglob('*') if p.is_file()}
        if current != manifest['knowledge']['files']:
            raise ValueError('source index changed during evaluation; check external writers')
    eligible = manifest['execution_complete'] and not engine.gate.blocked and not framework_error and not engine.gate.stopped.is_set()
    manifest['batch_paused'] = engine.gate.blocked
    manifest['interrupted'] = engine.gate.stopped.is_set()
    manifest['framework_error'] = framework_error
    write_json(output/'manifest.json',manifest)
    load_agent_run(output)
    manifest['complete'] = bool(eligible)
    write_json(output/'manifest.json',manifest)
    return summary


def rescore_missing(run_dir, version, *, values):
    """New scoring version only. Never executes an Agent or overwrites v1."""
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in version):
        raise ValueError('invalid score version')
    home = Path(run_dir).resolve()
    _,cases,executions,old = load_agent_run(home)
    target = home/'scores'/version
    target.mkdir(parents=True,exist_ok=False)
    e_by = {e['case_id']:e for e in executions}
    old_by = {s['case_id']:s for s in old}
    missing = [c for c in cases if old_by[c['case_id']]['business_outcome']=='unavailable'
               and e_by[c['case_id']]['execution_status']=='completed']
    if missing:
        scored = asyncio.run(score_saved(missing,[e_by[c['case_id']] for c in missing],values,target,PaidCallGate()))
    else:
        scored=[]
    by = {**old_by, **{s['case_id']:s for s in scored}}
    # score_saved already appended rescored rows; replace this NEW version's file with full coverage.
    (target/'scores.jsonl').write_text(''.join(json.dumps(by[c['case_id']],ensure_ascii=False)+'\n' for c in cases),encoding='utf-8')
    write_json(target/'manifest.json',{'complete':True,'source_version':'v1','rescored_case_ids':[c['case_id'] for c in missing],
               'executions_sha256':file_hash(home/'executions.jsonl'),'scores_sha256':file_hash(target/'scores.jsonl')})
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['smoke','full','repeat'],default='smoke')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--score-only',type=Path)
    parser.add_argument('--score-version')
    parser.add_argument('--live',action='store_true')
    args = parser.parse_args()
    from dotenv import dotenv_values
    values = {**dotenv_values(PROJECT/'.env',encoding='utf-8-sig',interpolate=False),**os.environ}
    if args.score_only:
        if not args.score_version or not args.live:
            parser.error('--score-only requires --score-version and explicit --live for possible paid judging')
        rescore_missing(args.score_only,args.score_version,values=values)
        return
    if not args.output:
        parser.error('--output required for a new execution')
    cases = select_cases(load_cases(),stage=args.stage)
    if args.stage == 'repeat':
        if args.output.exists():
            raise FileExistsError('repeat output must be new')
        for i in range(3):
            run_batch(cases,output=args.output/f'trial-{i+1}',values=values,concurrency=1,live=args.live)
    else:
        run_batch(cases,output=args.output,values=values,concurrency=1 if args.stage=='smoke' else 2,live=args.live)


if __name__ == '__main__':
    main()
