"""Dense CoTracker3 point tracks of the SAM 3 objects, and a disk cache for the SAM track building they are seeded from."""
import pickle
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]


def cached_tracks(video):
    """scene.tracks.build_tracks() (slow, CPU) cached to disk -> (d, tracks, table, kept)."""
    from cleantables.perception.scene.tracks import build_tracks, load_dets
    p = _ROOT / "data/twin_table/flow" / f"{video}_samtracks.pkl"
    if p.exists():
        tracks, table, kept = pickle.load(open(p, "rb"))
        return load_dets(video), tracks, table, kept
    d, tracks, table, kept = build_tracks(video)
    p.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((tracks, table, kept), open(p, "wb"))
    return d, tracks, table, kept
