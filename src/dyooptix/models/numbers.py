import re
import numpy as np

WIDTH = 512
NUM = r"(\d+(?:\.\d+)?)"
DEG = re.compile(NUM + r"\s*(?:°|deg(?:rees?)?\b)")
RAD = re.compile(NUM + r"\s*(?:radians?|rad)\b")
PX = re.compile(NUM + r"\s*(?:px\b|pixels?\b)")
PCT = re.compile(r"(to\s+)?" + NUM + r"\s*(?:%|percent\b)")
FAC = re.compile(NUM + r"\s*(?:x\b|times\b)|factor of " + NUM)
BARE = re.compile(r"(?<![\w.])" + NUM + r"(?![\w.%])(?!\s*(?:units?|radians?|rad)\b)")
ROT_WORD = re.compile(r"clockwise|counter|anti|\bleft\b|\bright\b")
SCALE_WORD = re.compile(r"scal|shrink|enlarg|bigger|larger|smaller|resiz|reduc|grow|expand|zoom")
FACTOR_WORD = re.compile(r"\b(double|twice|halve|half)\b")
FACTOR_VALUE = {"double": 2, "twice": 2, "halve": 0.5, "half": 0.5}


def wrap(deg):
    return (deg + 180) % 360 - 180


def fix_numbers(clause, pred, present):
    """pred = [tx, ty, cos, sin, log s]; present = (translate, rotate, scale) as booleans."""
    out = np.array(pred, dtype=float)
    text = clause.lower()
    t, r, sc = present
    scale_words = bool(SCALE_WORD.search(text))

    if r:
        deg = None
        if DEG.search(text):
            deg = float(DEG.search(text).group(1))
        elif RAD.search(text):
            deg = np.degrees(float(RAD.search(text).group(1))) 
        elif not (t or sc) and BARE.search(text):
            deg = float(BARE.search(text).group(1))
        if deg is not None:
            net = np.degrees(np.arctan2(out[3], out[2]))
            sign = (1 if net >= 0 else -1) if ROT_WORD.search(text) else 1
            a = np.radians(wrap(sign * deg))
            out[2], out[3] = np.cos(a), np.sin(a)

    if t:
        frac = None
        if PX.search(text):
            frac = float(PX.search(text).group(1)) / WIDTH
        elif not scale_words and PCT.search(text):
            frac = float(PCT.search(text).group(2)) / 100
        elif not (r or sc) and BARE.search(text):
            frac = float(BARE.search(text).group(1)) / WIDTH
        if frac is not None and np.hypot(*out[:2]) > 0:
            k = round(np.degrees(np.arctan2(out[1], out[0])) / 45)   # nearest of the 8 directions
            a = np.radians(45 * k)
            out[:2] = frac * np.array([np.cos(a), np.sin(a)])

    if sc:
        f = None
        m, p, w = FAC.search(text), PCT.search(text), FACTOR_WORD.search(text)
        if m:
            f = float(m.group(1) or m.group(2))
            if out[4] < 0 and f > 1:
                f = 1 / f
        elif p and (scale_words or not t):
            n = float(p.group(2)) / 100
            f = n if p.group(1) else (1 + n if out[4] >= 0 else 1 - n)
        elif w:
            f = FACTOR_VALUE[w.group(1)]
        if f and f > 0:
            out[4] = np.log(f)

    return out