"""Task families for per-family evaluation, assigned by episode id (the source video). Used only at evaluation time.

Reset episodes (id ..._r<j>) -> reset; IMG_2513 / 2514 -> novel clean-up (not gated: no training examples of that family);
IMG_2516 -> fruit; IMG_2533-2548 -> tools (including the 'close the toolbox, then stack' t02 episodes); IMG_2549-2555 -> boxes;
IMG_2518 (bowls / mugs) -> other. The grouping was fixed after fold 0 had been scored."""
import re

GATED = ("fruit", "tools", "boxes", "reset")


def family(ep):
    v = ep[:8]
    if v in ("IMG_2513", "IMG_2514"):
        return "novel clean-up"
    if re.search(r"_r\d+$", ep):
        return "reset"
    n = int(v[4:]) if v[4:].isdigit() else -1
    if v == "IMG_2516":
        return "fruit"
    if 2533 <= n <= 2548:
        return "tools"
    if 2549 <= n <= 2555:
        return "boxes"
    return "other"
