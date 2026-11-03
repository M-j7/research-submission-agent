"""Metadata only discovery. Bodies are disclosed using the ordinary read tool."""
from pathlib import Path
import re


class SkillCatalog:
    def __init__(self, root: Path):
        self.root=root

    def entries(self):
        result=[]
        for path in sorted(self.root.glob('*/SKILL.md')):
            if path.is_symlink() or path.parent.is_symlink():
                continue
            text=path.read_text(encoding='utf-8-sig')
            header=re.match(r'^---\s*\n(.*?)\n---\s*\n',text,re.S)
            if not header:
                raise ValueError('invalid Skill metadata')
            fields=dict(line.split(':',1) for line in header.group(1).splitlines() if ':' in line)
            fields={k.strip():v.strip() for k,v in fields.items()}
            if fields.get('name')!=path.parent.name or not fields.get('description'):
                raise ValueError('Skill name/path mismatch or empty description')
            result.append({'name':fields['name'],'description':fields['description'],
                           'path':f'/system/skills/{path.parent.name}/SKILL.md'})
        return result
