"""_5_yesno_parse.py — strict yes/no answer extraction + a re-scorer.

Motivation: the yes/no harnesses used `re.search(r"\\b(yes|no)\\b", text)`, which
takes the first yes/no ANYWHERE in the string. That is loose in two ways:
  * a preamble ("I'm not sure, but no") is read as the answer;
  * a degenerate generation ("NoNo") has no word boundary, so it parses as Unk.
CMM's official scorer has the opposite failure: `answer in pred[:5]` matches a
SUBSTRING, so "nonsense" would count as "no".

`parse_yes_no` follows AV-SpeakerBench's extraction shape (punctuation -> space,
split, look at tokens) with a documented fallback chain:
  1. punctuation stripped, split on whitespace, FIRST token that is exactly
     yes/no  -> that answer          ("No. What else?" -> No)
  2. else, the string STARTS with yes/no (degenerate "NoNo", "Yes,")
     -> that answer
  3. else "Unk".
Step 1 before step 2 so a real sentence always beats a prefix heuristic.

Run directly to re-score every stored yes/no CSV with this parser and report any
row whose label would change:
    python method/sink_analysis/qwen2_5_omni/_5_yesno_parse.py
"""
import glob
import os
import re

_PUNCT = re.compile(r"[.,:!'\";/\?`~@#\$%\^&\*\(\)\[\]\{\}\\|<>\n]")


def parse_yes_no(text: str) -> str:
    """-> 'Yes' | 'No' | 'Unk'. See module docstring for the fallback chain."""
    if text is None:
        return "Unk"
    s = str(text).strip()
    if not s:
        return "Unk"
    for tok in _PUNCT.sub(" ", s).split():
        t = tok.lower()
        if t in ("yes", "no"):
            return t.capitalize()
    m = re.match(r"(yes|no)", s, re.IGNORECASE)
    return m.group(1).capitalize() if m else "Unk"


if __name__ == "__main__":
    import pandas as pd

    REPO = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    # (glob, text column, stored-prediction column or None)
    SETS = [
        (f"{REPO}/results/qwen2_5_omni/stage5_intervention/*FULL*.csv",
         "generated", "predicted"),
        (f"{REPO}/results/qwen3_omni/avhbench_baselines/gen_*full*.csv",
         "raw", "pred"),
        (f"{REPO}/results/qwen2_5_omni/cmm/cmm_*.csv", "pred", None),
        (f"{REPO}/results/qwen3_omni/cmm/cmm_*.csv", "pred", None),
    ]
    tot = chg = unk = 0
    for pat, tcol, pcol in SETS:
        for f in sorted(glob.glob(pat)):
            try:
                d = pd.read_csv(f)
            except Exception:
                continue
            if tcol not in d or not len(d):
                continue
            new = d[tcol].map(parse_yes_no)
            tot += len(d)
            u = int((new == "Unk").sum())
            unk += u
            note = ""
            if pcol and pcol in d:
                diff = (new != d[pcol].astype(str))
                chg += int(diff.sum())
                note = f"changed={int(diff.sum())}"
            else:
                # CMM stores no parsed column; compare against its p5 rule.
                p5 = d[tcol].astype(str).str.strip().str.lower().str[:5]
                cmm = p5.apply(lambda s: "Yes" if "yes" in s
                               else ("No" if "no" in s else "Unk"))
                diff = (new != cmm)
                chg += int(diff.sum())
                note = f"vs-p5-differs={int(diff.sum())}"
            if u or "=0" not in note:
                print(f"  {os.path.basename(f):<52} n={len(d):<6} {note} unk={u}")
    print(f"\nTOTAL answers re-parsed: {tot}   label changes: {chg}   Unk: {unk}")
