"""Calibrate on a development set; score the unchanged policy on an independent test set."""
import argparse, hashlib, json, math, sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from urdu_core import Fixer, SplitCut, apply_edits, lookup, read_json, sentences, sha, span_key, write_json

def fingerprint(s):
    return hashlib.sha256(s.encode()).hexdigest()
def token_lines(s):
    return [' '.join((t.key for t in sent)) for sent in sentences(s)]

def inspect(fx,path,split,corpus=None):
    docs = []
    identities = set()
    groups = set()
    texts = set()
    kinds = set()
    overlap = set()
    db = None
    if corpus:
        corpus = Path(corpus)
        if sha(corpus/"resources.json")!=fx.asset:
            raise ValueError('Training provenance and model vocabulary differ')
        hashes = read_json(corpus / 'manifest.json')['files']
        if "provenance.sqlite" in hashes and sha(corpus/"provenance.sqlite")!=hashes["provenance.sqlite"]:
            raise ValueError('Corpus provenance checksum mismatch')
        db = sqlite3.connect(f"{(corpus / 'provenance.sqlite').resolve().as_uri()}?mode=ro", uri=True)
    with open(path,encoding="utf-8-sig") as f:
        for n,line in enumerate(f,1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("split")!=split or row.get("kind") not in {"human","synthetic","example"}:
                raise ValueError(f'{path}:{n}: invalid split or kind')
            if not all(isinstance(row.get(k),str) and row[k].strip() for k in ("id","group","text","domain")) or not isinstance(row.get("edits"),list):
                raise ValueError(f'{path}:{n}: expected id, group, text, domain and edits')
            ident = row['id']
            text = row['text']
            if ident in identities:
                raise ValueError(f'Duplicate evaluation id {ident}')
            identities.add(ident)
            groups.add(fingerprint(row['group']))
            kinds.add(row['kind'])
            tokens = {(t.start, t.end): t for s in sentences(text) for t in s}
            gold = {}
            edits = []
            for e in row["edits"]:
                key = (e['start'], e['end'])
                t = tokens.get(key)
                v = e['replacement']
                if (not t or not span_key(lookup(v)) or not span_key(lookup(t.raw)) or t.raw==v or key in gold):
                    raise ValueError(f'{ident}: expected a distinct Urdu lexical substitution at {key}')
                gold[key] = v
                edits.append(dict(e, original=t.raw))
            corrected = apply_edits(text, edits)
            for version in (text,corrected):
                lines = token_lines(version)
                texts.update((fingerprint(s) for s in lines))
                if db and any(db.execute("SELECT 1 FROM lines WHERE sig=? AND part='train'",(hashlib.sha256(s.encode()).digest(),)).fetchone() for s in lines):
                    overlap.add(ident)
            if db and db.execute("SELECT 1 FROM docs WHERE grp=? AND part='train' LIMIT 1",(row["group"],)).fetchone():
                overlap.add(ident)
            eligible_words = sum((1 for t in tokens.values() if span_key(lookup(t.raw))))
            coverage = 0
            for key,v in gold.items():
                t = tokens[key]
                observed = fx.ids.get(t.key, 0)
                target = fx.ids.get(lookup(v), -1)
                coverage += int(t.editable and target > 0 and (target in fx.graph[observed, 1:]))
            predictions = []
            for e in fx.decisions(text):
                if e["margin"]<=0:
                    continue
                k = (e['start'], e['end'])
                status = 'fixed' if e['replacement'] == gold.get(k) else 'wrong' if k in gold else 'harm'
                predictions.append((e['margin'], status, 'curated' if e.get('curated') else 'edit'))
            docs.append((row['domain'], eligible_words, len(gold), coverage, predictions))
    if db:
        db.close()
    if not docs:
        raise ValueError('Empty evaluation set.')
    return docs,{"ids":sorted(fingerprint(x) for x in identities),"groups":sorted(groups),"texts":sorted(texts),"kinds":sorted(kinds),
                 "train_overlap_rows":len(overlap),"training_provenance_checked":bool(corpus)}

def counts(rows):
    totals = Counter()
    events = []
    for _,tokens,errors,cover,decisions in rows:
        totals.update({'documents': 1, 'words': tokens, 'errors': errors, 'covered': cover})
        if not errors:
            totals.update({'clean_documents': 1, 'clean_control_words': tokens})
        for d in decisions:  # (margin, status[, kind]); two-field rows count as generic edits
            events.append((d[0], d[1], not errors, d[2] if len(d) > 2 else "edit"))
    return (totals, events)

def wilson(k,n):
    if not n:
        return [None, None]
    z = 1.95996398454
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0, c - h), min(1, c + h)]

def rates(t,c,intervals=False):
    edits = c['fixed'] + c['wrong'] + c['harm']
    clean = t['clean_control_words']
    v={"documents":t["documents"],"gold_errors":t["errors"],"clean_control_words":clean,"suggestions":edits,"fixed":c["fixed"],
       "incorrect_replacements":c["wrong"],"edits_to_correct_words":c["harm"],"clean_control_false_edits":c["control_harm"],
       "precision":c["fixed"]/edits if edits else None,"recall":c["fixed"]/t["errors"] if t["errors"] else None,
       "candidate_recall":t["covered"]/t["errors"] if t["errors"] else None,
       "false_edits_per_1000_clean":1000*c["control_harm"]/clean if clean else None}
    if intervals:
        v['precision_wilson95'] = wilson(c['fixed'], edits)
        v["clean_false_rate_wilson95_per_1000"]=[None if x is None else x*1000 for x in wilson(c["control_harm"],clean)]
    return v

def acc(c,event):
    status, clean = event[1], event[2]
    c[status] += 1
    if clean and status=="harm":
        c['control_harm'] += 1

def applies(event,cut):
    """Whether an event (margin, status, clean, kind) passes the policy cutoff."""
    if isinstance(cut, SplitCut):
        cut = cut.curated if event[3]=="curated" else cut.edit
    return cut is not None and event[0]>=cut

def cut_json(cut):
    return dict(cut._asdict()) if isinstance(cut, SplitCut) else cut

def measure(rows,cut):
    t, events = counts(rows)
    c = Counter()
    kinds = {"curated": Counter(), "edit": Counter()}
    for e in events:
        if applies(e, cut):
            acc(c, e)
            acc(kinds[e[3]], e)
    v = rates(t, c, True)
    v["by_candidate_kind"] = {k: {"suggestions": n["fixed"]+n["wrong"]+n["harm"], "fixed": n["fixed"],
                                  "incorrect_replacements": n["wrong"], "edits_to_correct_words": n["harm"],
                                  "clean_control_false_edits": n["control_harm"]} for k, n in kinds.items()}
    return v

def pick_split(rows,precision,max_false,min_edits,grid=600):
    """Choose separate curated/edit cutoffs jointly on dev: most fixes subject to the same targets.

    Each class's candidate cutoffs are quantiles of its own positive margins, plus 'never'.
    """
    import numpy as np
    t, events = counts(rows)
    if not t["errors"] or not t["clean_control_words"]:
        raise ValueError('Calibration needs both errors and independent clean controls.')
    tables = {}
    for kind in ("curated", "edit"):
        ev = [e for e in events if e[3]==kind]
        m = np.array([e[0] for e in ev], dtype=np.float64)
        cols = {s: np.array([e[1]==s for e in ev]) for s in ("fixed","wrong","harm")}
        cols["control_harm"] = np.array([e[1]=="harm" and e[2] for e in ev])
        cuts = np.unique(np.quantile(m, np.linspace(0, 1, grid))) if len(m) else np.array([])
        cuts = np.concatenate((cuts, [np.inf]))  # inf = never suggest this class
        order = np.argsort(-m)
        sm = m[order]
        n_at = np.searchsorted(-sm, -cuts, side="right")  # events with margin >= cut
        tables[kind] = (cuts, {k: np.concatenate(([0], np.cumsum(v[order])))[n_at] for k, v in cols.items()})
    (ca, A), (cb, B) = tables["curated"], tables["edit"]
    S = {k: A[k][:, None] + B[k][None, :] for k in A}
    edits = S["fixed"] + S["wrong"] + S["harm"]
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(edits > 0, S["fixed"] / np.maximum(edits, 1), 0)
    ok = (edits >= min_edits) & (prec >= precision) & (1000 * S["control_harm"] / t["clean_control_words"] <= max_false)
    if not ok.any():
        return None
    score = np.where(ok, S["fixed"] + prec * 0.5, -1)  # most fixes, then precision as tie-break
    i, j = np.unravel_index(np.argmax(score), score.shape)
    val = lambda x: None if not np.isfinite(x) else float(x)
    return SplitCut(val(ca[i]), val(cb[j]))

def pick(rows,precision,max_false,min_edits):
    import itertools
    t, events = counts(rows)
    if not t["errors"] or not t["clean_control_words"]:
        raise ValueError('Calibration needs both errors and independent clean controls.')
    c = Counter()
    best = None
    bestkey = (-1, -1.0)
    for cut,items in itertools.groupby(sorted(events,reverse=True),key=lambda x:x[0]):
        for e in items:
            acc(c, e)
        v = rates(t, c)
        if v["suggestions"]>=min_edits and v["precision"]>=precision and v["false_edits_per_1000_clean"]<=max_false:
            key = (v['fixed'], v['precision'])
            if key>bestkey:
                best, bestkey = (cut, key)
    return best

def verify_policy(fx,pol):
    if (pol["model_sha256"]!=fx.fingerprint or pol["resources_sha256"]!=fx.asset):
        raise ValueError('Policy belongs to different model or candidate list.')
    split = pol.get('thresholds')
    cut = SplitCut(split.get('curated'), split.get('edit')) if split else pol['threshold']
    for x in (cut if isinstance(cut, SplitCut) else [cut]):
        if x is not None and (not math.isfinite(x) or x<=0):
            raise ValueError('Invalid policy threshold.')
    return cut

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['calibrate', 'test'])
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--resources', type=Path, required=True)
    p.add_argument('--rows', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--data', type=Path)
    p.add_argument('--precision', type=float, default=0.95)
    p.add_argument('--false-per-1000', type=float, default=0.5)
    p.add_argument('--min-edits', type=int, default=5)
    p.add_argument('--split-thresholds', action='store_true',
                   help='calibrate separate cutoffs for curated confusion pairs and generic one-edit candidates')
    a = p.parse_args()
    if not 0<a.precision<=1 or a.false_per_1000<0 or a.min_edits<1:
        p.error('Invalid calibration targets.')
    fx = Fixer(a.model, a.resources)
    split = 'dev' if a.action == 'calibrate' else 'test'
    docs, meta = inspect(fx, a.rows, split, a.data)
    if a.action=="calibrate":
        single = pick(docs, a.precision, a.false_per_1000, a.min_edits)
        # "threshold" stays a single number so older runtimes can still read the policy.
        pol={"model_sha256":fx.fingerprint,"resources_sha256":fx.asset,"dev_sha256":sha(a.rows),"threshold":single,
             "minimum_precision":a.precision,"max_false_per_1000":a.false_per_1000,"dev":meta}
        cut = single
        if a.split_thresholds:
            split = pick_split(docs, a.precision, a.false_per_1000, a.min_edits)
            if split is not None:
                pol["thresholds"] = cut_json(split)
                cut = split
        write_json(a.policy, pol)
    else:
        pol = read_json(a.policy)
        cut = verify_policy(fx, pol)
        if sha(a.rows)==pol["dev_sha256"] or any(set(meta[k]) & set(pol["dev"][k]) for k in ("ids","groups","texts")):
            raise ValueError("Development and test overlap. Use independent documents, groups and sentence texts.")
    summary = measure(docs, cut)
    gates = None
    if a.action=="test":
        lo = summary['precision_wilson95'][0]
        hi = summary['clean_false_rate_wilson95_per_1000'][1]
        gates={"human_dev_and_test":meta["kinds"]==pol["dev"]["kinds"]==["human"],"training_overlap_checked":bool(a.data) and pol["dev"]["training_provenance_checked"],
               "no_detected_train_overlap":meta["train_overlap_rows"]==0 and pol["dev"]["train_overlap_rows"]==0,
               "at_least_10000_clean_words":summary["clean_control_words"]>=10000,"at_least_100_errors":summary["gold_errors"]>=100,
               "precision_lower_bound":lo is not None and lo>=pol["minimum_precision"],
               "clean_false_rate_upper_bound":hi is not None and hi<=pol["max_false_per_1000"],
               "at_least_20pct_recall":(summary["recall"] or 0)>=.20}
    by_domain = defaultdict(list)
    for row in docs:
        by_domain[row[0]].append(row)
    report={"split":split,"rows_sha256":sha(a.rows),"model_sha256":fx.fingerprint,"threshold":cut_json(cut),"kinds":meta["kinds"],
            "train_overlap_rows":meta["train_overlap_rows"],"metrics":summary,"by_domain":{d:measure(rr,cut) for d,rr in by_domain.items()},
            "always_keep_baseline":measure(docs,None),"release_checks":gates,"meets_proposed_release_checks":all(gates.values()) if gates is not None else None}
    write_json(a.report, report)
    print(json.dumps({k:report[k] for k in ["split","threshold","metrics","meets_proposed_release_checks"]},indent=2))
if __name__=="__main__":
    main()
