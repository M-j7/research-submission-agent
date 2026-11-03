"""Paired descriptive effects; no independent trials or causal significance claimed."""
from itertools import combinations
from statistics import median

from .ablation_design import ARMS

FACTORS = ('boundary', 'examples', 'history')
BITS_TO_ARM = {tuple(getattr(a, f) for f in FACTORS): a.name for a in ARMS}


def arm_summary(rows):
    result = {}
    for a in ARMS:
        selected = [r for r in rows if r['arm'] == a.name]
        valid = [r for r in selected if r['available']]
        tp = sum(len(r['tp']) for r in valid)
        fp = sum(len(r['fp']) for r in valid)
        fn = sum(len(r['fn']) for r in valid)
        n = len(valid)
        result[a.name] = {'samples': len(selected), 'available': n, 'unavailable': len(selected)-n,
            'task_exact': sum(r['task_exact'] for r in valid),
            'intent_exact': sum(r['intent_exact'] for r in valid),
            'visible_exact': sum(r['visible_exact'] for r in valid),
            'task_rate': sum(r['task_exact'] for r in valid)/n if n else None,
            'intent_rate': sum(r['intent_exact'] for r in valid)/n if n else None,
            'visible_rate': sum(r['visible_exact'] for r in valid)/n if n else None,
            'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else None,
            'recall': tp/(tp+fn) if tp+fn else None,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            'logical_input_tokens': sum(r['input_tokens'] for r in valid if r.get('input_tokens') is not None),
            'usage_known_rows': sum(r.get('input_tokens') is not None for r in valid),
            'latency_p50_ms': median(r['latency_ms'] for r in valid) if n else None}
    return result


def factor_effects(rows, *, metric='task_exact'):
    by = {(r['case_id'], r['arm']): r for r in rows}
    case_ids = sorted({r['case_id'] for r in rows})
    def contrast(terms):
        values = []
        for case_id in case_ids:
            for arms in terms:
                selected = [(coef, by.get((case_id, name))) for coef, name in arms]
                if all(r and r['available'] for _, r in selected):
                    values.append(sum(coef * float(r[metric]) for coef, r in selected))
        return {'delta_pp': 100*sum(values)/len(values) if values else None,
                'paired_contrasts': len(values), 'metric': metric}
    main, interactions = {}, {}
    for i, factor in enumerate(FACTORS):
        terms = []
        for bits, off_name in BITS_TO_ARM.items():
            if bits[i]:
                continue
            on = list(bits); on[i] = True
            terms.append(((1, BITS_TO_ARM[tuple(on)]), (-1, off_name)))
        main[factor] = contrast(terms)
    for i, j in combinations(range(3), 2):
        terms = []
        for bits in BITS_TO_ARM:
            if bits[i] or bits[j]:
                continue
            cells = []
            for vi, vj, coefficient in ((True, True, 1), (True, False, -1), (False, True, -1), (False, False, 1)):
                changed = list(bits); changed[i], changed[j] = vi, vj
                cells.append((coefficient, BITS_TO_ARM[tuple(changed)]))
            terms.append(cells)
        interactions[FACTORS[i]+':'+FACTORS[j]] = contrast(terms)
    return {'main': main, 'interactions': interactions, 'metric': metric,
            'caveat': '配对描述性差值，非独立重复试验，不能解释为总体因果贡献或统计显著性。'}
