import asyncio
import importlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec('submission_assistant.runtime.tools'), 'tool runtime missing')
        self.api = importlib.import_module('submission_assistant.runtime.tools')
        from submission_assistant.runtime.contracts import Request
        from submission_assistant.runtime.memory import MemoryManager
        from submission_assistant.runtime.workspace import WorkspaceManager
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.req = Request(user_id='u',conv_id='c',turn_id='t',message='费用')
        self.memory = MemoryManager(self.root/'memory.sqlite3')
        self.memory.begin_turn(self.req)
        self.memory.create_run(self.req,'r','fees',[])
        self.workspace = WorkspaceManager(self.root,self.req,'r')
        self.tools = self.api.ToolRegistry(self.workspace,self.memory,self.req,'r',max_chars=80)
        (self.root/'resources'/'skills').mkdir(parents=True)
        (self.root/'resources'/'skills'/'a.md').write_text('第一行\n'+('x'*250)+'\n第三行',encoding='utf-8')

    def tearDown(self):
        if hasattr(self,'memory'): self.memory.close()
        if hasattr(self,'tmp'): self.tmp.cleanup()

    def call(self, name, args, allowed=None):
        return asyncio.run(self.tools.execute(name,json.dumps(args),allowed or ['glob','grep','read','query_memory']))

    def test_read_long_single_line_is_losslessly_continued_and_cursor_is_bound(self):
        args = {'ref':'/system/skills/a.md'}
        result = self.call('read',args)
        text = result.data['text']
        self.assertEqual(result.status,'partial')
        cursor = result.next_cursor
        while cursor:
            result = self.call('read',{**args,'cursor':cursor})
            text += result.data['text']
            cursor = result.next_cursor
        self.assertEqual(text,'第一行\n'+('x'*250)+'\n第三行')
        first = self.call('read',args)
        result = self.call('read',{'ref':'/system/skills/other.md','cursor':first.next_cursor})
        self.assertEqual(result.error.code,'STALE_CURSOR')
        (self.root/'resources'/'skills'/'a.md').write_text('changed',encoding='utf-8')
        self.assertEqual(self.call('read',{**args,'cursor':first.next_cursor}).error.code,'STALE_CURSOR')

    def test_path_traversal_sql_and_unregistered_tools_never_execute(self):
        for ref in ('/system/../.env','C:/secret','/shared/user/memory/x','/system/memory.sqlite3'):
            with self.subTest(ref=ref):
                self.assertEqual(self.call('read',{'ref':ref}).status,'error')
        self.assertEqual(self.call('query_memory',{'sql':'select * from users'}).error.code,'INVALID_ARGUMENT')
        self.assertEqual(self.call('read',{'ref':'/system/skills/a.md','max_lines':'5'}).error.code,'INVALID_ARGUMENT')
        self.assertEqual(self.call('search_knowledge',{'query':'fees'}).error.code,'TOOL_NOT_ALLOWED')
        self.assertEqual(self.call('delete_files',{}).error.code,'TOOL_NOT_ALLOWED')

    def test_glob_and_literal_grep_find_only_authorized_resources(self):
        found = self.call('glob',{'root':'/system/skills','pattern':'*.md'})
        self.assertEqual(found.data['entries'][0]['path'],'/system/skills/a.md')
        matched = self.call('grep',{'root':'/system/skills','pattern':'第一行','context_lines':0})
        self.assertEqual(matched.data['matches'][0]['line_number'],1)
        self.assertEqual(self.call('grep',{'root':'/system/skills','pattern':'不存在'}).status,'empty')

    def test_query_memory_does_not_expose_other_run_and_literal_sql_is_safe(self):
        self.memory.append(self.req,'r','assistant',{'content':'本run'},1)
        self.memory.create_run(self.req,'other','fees',[])
        self.memory.append(self.req,'other','assistant',{'content':'其他私有'},1)
        result = self.call('query_memory',{'scope':'instant'})
        self.assertEqual(result.data['records'][0]['content'],'本run')
        result = self.call('query_memory',{'keyword':"' OR 1=1 --"})
        self.assertEqual(result.status,'empty')

    def test_raw_tool_artifacts_survive_runtime_cleanup(self):
        ref = self.workspace.save_tool_result('call',{'data':'原文'})
        asyncio.run(self.memory.seal_run(self.req,'r','completed'))
        self.workspace.cleanup()
        self.assertFalse(self.workspace.runtime.exists())
        self.assertIn('原文',self.workspace.resolve(ref).read_text(encoding='utf-8'))

    def test_reparse_escape_is_rejected(self):
        from unittest.mock import patch
        # Windows symlink creation needs privileges; inject only the OS link detector.
        target = self.root/'resources'/'skills'/'a.md'
        with patch.object(Path,'is_symlink',lambda p: p == target):
            result = self.call('read',{'ref':'/system/skills/a.md'})
        self.assertEqual(result.error.code,'RESOURCE_UNAVAILABLE')
