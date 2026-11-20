"""Read object names and object-to-object relations ("until it touches the X", "beside the X") from an instruction."""
import re

ART = r"(?:the |a |an |this |that |my )?"
TOUCH = re.compile(r",?\s*until (?:it |they )?(?:touch|reach|hit|meet|contact)(?:es|s)?\s+" + ART + r"(?P<target>.+)$")
SIDE = re.compile(r",?\s*\b(?:to|on) the (?P<side>left|right)(?: side)? of\s+" + ART + r"(?P<target>.+)$")
BESIDE = re.compile(r",?\s*\b(?:beside|next to|alongside|near(?: to)?|close to|adjacent to|by the side of)\s+" + ART + r"(?P<target>.+)$")
EDGE = re.compile(r"^(?:(?P<side>left|right|top|upper|bottom|lower)\s+)?(?:edge|end|border|boundary|side)\b")
EDGE_SIDE = {"upper": "top", "lower": "bottom"}
DIRECTION = re.compile(r"\b(?:left|right|up|down)(?:wards?)?\b")
NAMED = re.compile(r"\b(?:the|this|that|my)\s+(?P<name>[a-z]+(?:\s+[a-z]+)?)")

WHOLE = {"image", "picture", "photo", "scene", "frame", "whole", "everything"}   # these mean: edit the whole image
NOT_NAME = {                                                                     # words that end an object's name
    "a", "an", "and", "then", "by", "to", "of", "in", "on", "at", "it", "its", "so", "until", "beside", "next", "near",
    "left", "right", "up", "down", "upward", "upwards", "downward", "downwards", "leftward", "leftwards", "rightward",
    "rightwards", "toward", "towards", "clockwise", "anticlockwise", "counterclockwise", "counter", "anti",
    "slightly", "little", "bit", "lot", "much", "gently", "mildly", "noticeably", "significantly", "considerably",
    "bigger", "smaller", "larger", "twice", "half", "times", "degrees", "px", "pixels", "percent",
}
VERBS = {                                                                        # edit verbs: never an object name
    "move", "shift", "slide", "push", "pull", "translate", "rotate", "turn", "spin", "scale", "resize", "shrink",
    "enlarge", "grow", "make", "bring", "place", "put", "drag", "expand", "reduce", "increase", "decrease", "zoom",
}
FILLER = {                                                                       # other words that are never a name
    "the", "as", "than", "is", "be", "x", "deg", "degree", "rad", "radian", "radians", "pixel", "around", "over", "out",
    "back", "away", "more", "less", "upside", "again", "also", "just", "about", "into", "from", "for", "with",
    "big", "small", "large", "tall", "wide", "short", "narrow", "size",
}


def loose_name(text):
    """Name written without 'the': the first word (and a following one) that is not a verb, number unit or direction."""
    skip = NOT_NAME | VERBS | FILLER
    words = re.findall(r"[a-z]+", text.lower())
    for i, w in enumerate(words):
        if w not in skip:
            if i + 1 < len(words) and words[i + 1] not in skip:
                w += " " + words[i + 1]
            return None if w.split()[0] in WHOLE else w
    return None


def name(text):
    """First object name in text: one or two words after the/this/that/my, else a bare name. None for 'it' or the whole image."""
    for m in NAMED.finditer(text):
        words = m.group("name").split()
        if words[0] in NOT_NAME:
            continue
        if len(words) == 2 and words[1] in NOT_NAME:
            words = words[:1]
        if words[0] in WHOLE:
            return None
        return " ".join(words)
    return loose_name(text)


def names_whole(text):
    """True if the text says 'image / picture / photo / whole ...': then the edit is for the whole image."""
    return any(w in WHOLE for w in re.findall(r"[a-z]+", text.lower()))


def target_name(text):
    """The target's name: the first one or two words of what follows the relation word."""
    words = text.split()[:2]
    if len(words) == 2 and words[1] in NOT_NAME:
        words = words[:1]
    return " ".join(words)


def parse(clause):
    """{"kind": "touch" | "edge" | "beside", "mover", "target", "side", "rest", "has_direction"}, or None for a plain edit.
    rest is the clause without the relation; it goes to the text model as usual."""
    for kind, pattern in (("touch", TOUCH), ("beside", SIDE), ("beside", BESIDE)):
        m = pattern.search(clause)
        if m:
            rest = clause[:m.start()].strip()
            out = {"kind": kind, "mover": name(rest), "target": target_name(m.group("target")),
                   "side": m.groupdict().get("side"), "rest": rest, "has_direction": bool(DIRECTION.search(rest))}
            edge = EDGE.match(m.group("target")) if kind == "touch" else None
            if edge:                                               # "until it touches the left end / the border"
                side = edge.group("side")
                out.update(kind="edge", target=None, side=EDGE_SIDE.get(side, side))
            return out
    return None