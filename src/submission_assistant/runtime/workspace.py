"""Virtual file scopes, not OS mounts. SQL/index/config files never exposed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from uuid import uuid4


def scope_key(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()[:24]


class WorkspaceManager:
    def __init__(self, project: Path, request, run_id):
        self.project = project.resolve()
        parts = [scope_key(x) for x in (request.user_id,request.conv_id,request.turn_id,run_id)]
        self.runtime = self.project.joinpath('runtime',*parts)
        self.blobs = self.project.joinpath('data','memory','blobs',*parts)
        self.shared = self.project.joinpath('data','workspaces',*parts[:2],'shared')
        self._roots = {
            '/system/skills':self.project/'resources'/'skills',
            '/system/templates':self.project/'resources'/'templates',
            '/system/manuals':self.project/'resources'/'manuals',
            '/private/scratch':self.runtime/'scratch',
            '/private/tool_results':self.blobs,
            **{f'/shared/conversation/{name}':self.shared/name for name in ('inputs','evidence','agent_results','outputs')},
        }
        for path in [self.runtime/'scratch',self.runtime/'cache',self.blobs,*[self.shared/n for n in ('inputs','evidence','agent_results','outputs')]]:
            self._check_components(path)
            path.mkdir(parents=True,exist_ok=True)
        self._created_runtime = self.runtime.resolve()

    def _check_components(self, path):
        # Check before resolve so symlink/junction escapes are not hidden by canonicalization.
        if not path.is_relative_to(self.project):
            raise PermissionError('resource unavailable')
        for p in [path,*path.parents]:
            if p == self.project.parent:
                break
            if p.is_symlink() or (p.exists() and getattr(p.stat(),'st_file_attributes',0)&0x400):
                raise PermissionError('resource unavailable')

    def resolve(self, ref):
        if not isinstance(ref,str) or not ref.startswith('/') or '\\' in ref or ':' in ref or '\x00' in ref:
            raise PermissionError('resource unavailable')
        parts = PurePosixPath(ref).parts
        if '..' in parts or any(p.startswith('.') for p in parts[1:]):
            raise PermissionError('resource unavailable')
        prefix = next((key for key in sorted(self._roots,key=len,reverse=True) if ref == key or ref.startswith(key+'/')),None)
        if prefix is None:
            raise PermissionError('resource unavailable')
        root = self._roots[prefix]
        target = root / ref[len(prefix):].lstrip('/')
        self._check_components(target)
        if not target.resolve().is_relative_to(root.resolve()):
            raise PermissionError('resource unavailable')
        if target.suffix.lower() in ('.sqlite3','.db','.env','.pyc','.exe'):
            raise PermissionError('resource unavailable')
        return target

    def virtual(self, path):
        for prefix,root in self._roots.items():
            if path.is_relative_to(root):
                ref = prefix+'/'+path.relative_to(root).as_posix()
                self.resolve(ref)
                return ref
        raise PermissionError('resource unavailable')

    def save_tool_result(self, call_id, result):
        # Names aren't model paths. Durable raw files start outside runtime.
        filename = scope_key(call_id)+'-'+uuid4().hex+'.json'
        path = self.blobs/filename
        self._check_components(path)
        with path.open('x',encoding='utf-8') as handle:
            json.dump(result,handle,ensure_ascii=False,allow_nan=False)
        return '/private/tool_results/'+filename

    def publish_evidence(self,evidence):
        text=evidence.get('text','')
        if not isinstance(text,str): raise ValueError('evidence text must be string')
        filename=hashlib.sha256(text.encode()).hexdigest()+'.md'
        path=self.shared/'evidence'/filename
        self._check_components(path)
        if not path.exists():
            try:
                with path.open('x',encoding='utf-8') as handle: handle.write(text)
            except FileExistsError: pass
        return '/shared/conversation/evidence/'+filename

    def cleanup(self):
        self._check_components(self.runtime)
        target = self.runtime.resolve()
        runtime_root = (self.project/'runtime').resolve()
        if target!=self._created_runtime or target==runtime_root or not target.is_relative_to(runtime_root):
            raise PermissionError('unsafe cleanup target')
        # Only this run's temporary directory. Caller must seal/cancel workers first.
        if target.exists():
            shutil.rmtree(target)
