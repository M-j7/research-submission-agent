import sys
import unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class SkillTests(unittest.TestCase):
    def test_catalog_discloses_metadata_not_full_policy_body(self):
        from submission_assistant.runtime.skills import SkillCatalog
        from submission_assistant.runtime.prompt import PromptManager
        entries=SkillCatalog(ROOT/'resources'/'skills').entries()
        self.assertEqual(len(entries),6,'six agreed draft Skills not deployed')
        self.assertEqual({e['name'] for e in entries},{'policy-evidence-check','manuscript-format-check','submission-material-check','publication-fee-check','editorial-workflow-guide','publication-policy-interpretation'})
        public={'profile':{},'summary':'','recent':[],'older':[]}
        instant={'message':'问题','intent_ids':[],'messages':[],'status':{}}
        snapshot=PromptManager(input_budget=30000).build(public,instant,system_prompt='角色',tools=[],skills=entries)
        self.assertIn('/system/skills/publication-fee-check/SKILL.md',snapshot.messages[0]['content'])
        self.assertNotIn('### 1. 费用项目、金额与条件一起核对',snapshot.messages[0]['content'])
        self.assertTrue(all(set(e)=={'name','description','path'} for e in entries))
