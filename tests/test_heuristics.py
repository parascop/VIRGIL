"""Heuristic tests on synthetic track sequences (no GPU, no models).
Sequence lengths are derived from the configured thresholds, not hard-coded."""
import copy
import math

import pytest

from helpers import box, cam, engine_for, person_down_script, run_engine
from vigil.config import heuristic_params
from vigil.heuristics import CooldownManager, HeuristicEngine, iou, point_in_polygon
from vigil.tracker import TrackBook
from vigil.types import Trigger

FPS = 10.0


def frames_for(seconds: float) -> int:
    return int(math.ceil(seconds * FPS)) + 2


def types(trigs):
    return {t.trigger_type for t in trigs}


# ---------------------------------------------------------------- person down
def test_person_down_positive(profile):
    hold = profile["heuristics"]["person_down"]["hold_s"]
    trigs = run_engine(engine_for(profile, cam()), person_down_script(n_still=frames_for(hold) + 5))
    assert "person_down" in types(trigs)
    t = next(t for t in trigs if t.trigger_type == "person_down")
    assert t.category == "person down" and 0 < t.heuristic_score <= 1 and t.track_ids == (1,)
    assert t.meta["held_s"] >= hold


def test_person_down_negative_sitting(profile):
    seq = [[(1, "person", box(300, 250, 40, 100))] for _ in range(20)]
    seq += [[(1, "person", box(300, 258, 80, 90))] for _ in range(60)]  # AR 0.89: sitting
    assert "person_down" not in types(run_engine(engine_for(profile, cam()), seq))


def test_person_down_negative_single_horizontal_box(profile):
    """A box that is horizontal from the very first frame is not a fall."""
    seq = [[(1, "person", box(310, 310, 100, 40))] for _ in range(80)]
    assert "person_down" not in types(run_engine(engine_for(profile, cam()), seq))


def test_person_down_negative_crawling(profile):
    """Horizontal but moving (crawling) must fail the stationarity test."""
    seq = [[(1, "person", box(300, 250, 40, 100))] for _ in range(20)]
    for k in range(60):
        seq.append([(1, "person", box(310 + 12 * k, 310, 100, 40))])
    assert "person_down" not in types(run_engine(engine_for(profile, cam()), seq))


def test_person_down_threshold_is_configurable(profile):
    p = copy.deepcopy(profile)
    p["heuristics"]["person_down"]["hold_s"] = 50.0  # longer than the clip
    assert "person_down" not in types(run_engine(engine_for(p, cam()), person_down_script()))


def test_person_down_tolerates_one_detector_glitch(profile):
    seq = person_down_script(n_still=40)
    seq[30] = [(1, "person", box(310, 310, 40, 100))]  # one jittery upright box
    assert "person_down" in types(run_engine(engine_for(profile, cam()), seq))


# ------------------------------------------------------------------- fighting
def _brawl(n: int, radius: float = 30.0):
    """Two people circling each other closely and quickly (sustained relative motion)."""
    out = []
    for k in range(n):
        th = 2 * math.pi * k / 12.0
        out.append([(1, "person", box(320 + radius * math.cos(th), 250 + radius * math.sin(th), 40, 100)),
                    (2, "person", box(320 - radius * math.cos(th), 250 - radius * math.sin(th), 40, 100))])
    return out


def test_fighting_positive(profile):
    n = frames_for(profile["heuristics"]["fighting"]["min_duration_s"]) + 20
    trigs = run_engine(engine_for(profile, cam()), _brawl(n))
    assert "fighting" in types(trigs)
    assert set(next(t for t in trigs if t.trigger_type == "fighting").track_ids) == {1, 2}


def test_fighting_negative_walking_together(profile):
    seq = [[(1, "person", box(100 + 8 * k, 250, 40, 100)), (2, "person", box(150 + 8 * k, 250, 40, 100))] for k in range(60)]
    assert "fighting" not in types(run_engine(engine_for(profile, cam()), seq))


def test_fighting_negative_close_but_still(profile):
    seq = [[(1, "person", box(300, 250, 40, 100)), (2, "person", box(335, 250, 40, 100))] for _ in range(60)]
    assert "fighting" not in types(run_engine(engine_for(profile, cam()), seq))


def test_fighting_threshold_configurable(profile):
    p = copy.deepcopy(profile)
    p["heuristics"]["fighting"]["motion_energy_min"] = 99.0
    assert "fighting" not in types(run_engine(engine_for(p, cam()), _brawl(60)))


# ----------------------------------------------------------------- tailgating
LINE = [{"name": "gate", "p1": [0.2, 0.55], "p2": [0.8, 0.55], "allowed_direction": "positive"}]


def _walker(tid, start_frame, n, y0=200, y1=340, x=320, direction=1):
    """Return {frame_index: det} for a walker crossing the line at y=264 (640x480)."""
    d = {}
    for k in range(n):
        y = y0 + (y1 - y0) * k / (n - 1) if direction > 0 else y1 - (y1 - y0) * k / (n - 1)
        d[start_frame + k] = (tid, "person", box(x, y - 50, 40, 100))  # foot = y
    return d


def _compose(total, *walkers):
    frames = [[] for _ in range(total)]
    for w in walkers:
        for k, det in w.items():
            if k < total:
                frames[k].append(det)
    return frames


def test_tailgating_positive(profile):
    c = cam(entry_lines=LINE)
    seq = _compose(60, _walker(1, 5, 14), _walker(2, 13, 14, x=340))  # ~0.8s apart
    trigs = run_engine(engine_for(profile, c), seq)
    assert "tailgating" in types(trigs)
    t = next(t for t in trigs if t.trigger_type == "tailgating")
    assert set(t.track_ids) == {1, 2} and "possible" in t.reason.lower()


def test_tailgating_negative_outside_window(profile):
    c = cam(entry_lines=LINE)
    win = profile["heuristics"]["tailgating"]["window_s"]
    gap_frames = int((win + 2.0) * FPS)
    seq = _compose(gap_frames + 40, _walker(1, 2, 14), _walker(2, 2 + gap_frames, 14, x=340))
    assert "tailgating" not in types(run_engine(engine_for(profile, c), seq))


def test_tailgating_negative_wrong_direction(profile):
    c = cam(entry_lines=LINE)
    seq = _compose(60, _walker(1, 5, 14), _walker(2, 13, 14, x=340, direction=-1))  # exits
    assert "tailgating" not in types(run_engine(engine_for(profile, c), seq))


def test_tailgating_needs_two_distinct_tracks(profile):
    c = cam(entry_lines=LINE)
    seq = _compose(60, _walker(1, 5, 14))
    assert "tailgating" not in types(run_engine(engine_for(profile, c), seq))


# ---------------------------------------------------------------------- crowd
def test_crowd_count_threshold(profile):
    limit = int(profile["heuristics"]["crowd"]["crowd_count"])
    dur = profile["heuristics"]["crowd"]["min_duration_s"]
    n = limit + 2
    seq = [[(i, "person", box(60 + 60 * i, 300, 30, 80)) for i in range(1, n + 1)] for _ in range(frames_for(dur) + 5)]
    trigs = run_engine(engine_for(profile, cam()), seq, wh=(1280, 720))
    assert "crowd_count" in types(trigs)
    t = next(t for t in trigs if t.trigger_type == "crowd_count")
    assert t.meta["count"] > limit and t.dedupe_ids == ()


def test_crowd_below_threshold_is_not_an_incident(profile):
    limit = int(profile["heuristics"]["crowd"]["crowd_count"])
    seq = [[(i, "person", box(60 + 60 * i, 300, 30, 80)) for i in range(1, limit + 1)] for _ in range(60)]
    assert "crowd_count" not in types(run_engine(engine_for(profile, cam()), seq, wh=(1280, 720)))


def test_crowd_dispersal(profile):
    d = profile["heuristics"]["crowd"]["dispersal"]
    n_calm = int((d["baseline_s"] + 2) * FPS)
    seq = []
    for k in range(n_calm):  # slow milling baseline
        seq.append([(i, "person", box(100 + 90 * i + (k % 2), 300, 30, 80)) for i in range(1, 5)])
    for k in range(40):  # everybody runs
        seq.append([(i, "person", box(100 + 90 * i + 40 * k * (1 if i % 2 else -1), 300, 30, 80)) for i in range(1, 5)])
    trigs = run_engine(engine_for(profile, cam()), seq, wh=(1920, 1080))
    assert "crowd_dispersal" in types(trigs)


# ----------------------------------------------------------------- suspicious
ZONES = [{"name": "door", "type": "restricted", "polygon": [[0.7, 0.2], [1.0, 0.2], [1.0, 1.0], [0.7, 1.0]], "loiter_s": 3},
         {"name": "vault", "type": "prohibited", "polygon": [[0.0, 0.0], [0.3, 0.0], [0.3, 0.3], [0.0, 0.3]]}]


def test_restricted_zone_loitering(profile):
    c = cam(zones=ZONES)
    n = frames_for(3) + 5
    seq = [[(1, "person", box(560, 300, 40, 100))] for _ in range(n)]  # foot (560,350) inside door zone
    trigs = run_engine(engine_for(profile, c), seq)
    assert "loitering" in types(trigs)
    assert next(t for t in trigs if t.trigger_type == "loitering").category == "suspicious activity"


def test_restricted_zone_short_visit_is_not_loitering(profile):
    c = cam(zones=ZONES)
    seq = [[(1, "person", box(560, 300, 40, 100))] for _ in range(10)]
    assert "loitering" not in types(run_engine(engine_for(profile, c), seq))


def test_prohibited_zone_entry(profile):
    c = cam(zones=ZONES)
    seq = [[(1, "person", box(100, 60, 40, 100))] for _ in range(frames_for(1.0))]  # foot (100,110) inside vault
    assert "zone_entry" in types(run_engine(engine_for(profile, c), seq))


def test_person_outside_zones_is_quiet(profile):
    c = cam(zones=ZONES)
    seq = [[(1, "person", box(320, 400, 40, 100))] for _ in range(100)]
    assert not run_engine(engine_for(profile, c), seq)


def test_abandoned_object_stretch(profile):
    p = copy.deepcopy(profile)
    p["heuristics"]["suspicious"]["abandoned"].update(enabled=True, stationary_s=3.0)
    seq = [[(1, "person", box(320, 300, 40, 100)), (9, "backpack", box(330, 340, 30, 30))] for _ in range(20)]
    seq += [[(9, "backpack", box(330, 340, 30, 30))] for _ in range(80)]
    assert "abandoned_object" in types(run_engine(engine_for(p, cam()), seq))


# ------------------------------------------------------------------- cooldown
def _trig(cid="cam1", ids=(1,), t=0.0, typ="person_down"):
    return Trigger(trigger_type=typ, category="person down", camera_id=cid, t=t, track_ids=ids, heuristic_score=0.5)


def test_cooldown_suppresses_and_extends_duration():
    cd = CooldownManager(30)
    act, e = cd.check(_trig(t=0))
    assert act == "new"
    act2, e2 = cd.check(_trig(t=12.5))
    assert act2 == "extend" and e2 is e and e.duration_s == pytest.approx(12.5)
    act3, e3 = cd.check(_trig(t=40.0))  # 27.5 s after the last sighting -> still inside cooldown
    assert act3 == "extend" and e3.duration_s == pytest.approx(40.0)
    act4, e4 = cd.check(_trig(t=75.0))  # 35 s gap -> new event
    assert act4 == "new" and e4 is not e


def test_cooldown_different_track_set_is_new_event():
    cd = CooldownManager(30)
    assert cd.check(_trig(ids=(1,), t=0))[0] == "new"
    assert cd.check(_trig(ids=(2,), t=1))[0] == "new"  # never silently merge unrelated people
    assert cd.check(_trig(ids=(1, 2), t=2, typ="fighting"))[0] == "new"


def test_cooldown_is_per_camera_and_per_type():
    cd = CooldownManager(30)
    assert cd.check(_trig(cid="cam1", t=0))[0] == "new"
    assert cd.check(_trig(cid="cam2", t=1))[0] == "new"
    assert cd.check(_trig(cid="cam1", t=1, typ="fighting"))[0] == "new"


def test_engine_rate_limits_persisting_candidate(profile):
    n = frames_for(profile["heuristics"]["person_down"]["hold_s"]) + 60  # long persistence
    trigs = [t for t in run_engine(engine_for(profile, cam()), person_down_script(n_still=n)) if t.trigger_type == "person_down"]
    assert 1 < len(trigs) < n  # re-emitted about once per emit_interval_s, not every frame


# --------------------------------------------------------- multi-camera / geom
def test_multi_camera_track_id_isolation(profile):
    """Same numeric track id on two cameras must never be mixed."""
    c1, c2 = cam("cam1"), cam("cam2")
    e1, e2 = engine_for(profile, c1), engine_for(profile, c2)
    b1, b2 = TrackBook("cam1"), TrackBook("cam2")
    upright = [[(1, "person", box(300, 250, 40, 100))] for _ in range(60)]
    t1 = run_engine(e1, person_down_script(), book=b1)
    t2 = run_engine(e2, upright, book=b2)
    assert any(t.camera_id == "cam1" for t in t1) and not t2
    assert b1.tracks[1] is not b2.tracks[1]
    assert all(t.camera_id == "cam1" for t in t1)


def test_geometry_helpers():
    assert point_in_polygon((0.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert not point_in_polygon((1.5, 0.5), [(0, 0), (1, 0), (1, 1), (0, 1)])
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_track_lifecycle_terminates_lost_tracks():
    book = TrackBook("cam1", lost_timeout_s=1.0)
    book.update(0.0, [(1, 0, "person", (0, 0, 10, 10), 0.9)])
    assert 1 in book.tracks
    book.update(0.5, [])
    assert 1 in book.tracks  # tolerated brief loss
    book.update(2.0, [])
    assert 1 not in book.tracks


def test_person_down_survives_track_id_switch_during_fall(profile):
    """Real trackers often drop the box mid-fall and restart a new, already-horizontal track."""
    seq = [[(1, "person", box(300, 250, 40, 100))] for _ in range(20)]
    seq += [[] for _ in range(3)]  # detector loses the falling person
    seq += [[(7, "person", box(330, 310, 100, 40))] for _ in range(40)]
    trigs = run_engine(engine_for(profile, cam()), seq)
    assert any(t.trigger_type == "person_down" and t.track_ids == (7,) for t in trigs)
    p = copy.deepcopy(profile)
    p["heuristics"]["person_down"]["reassociate"] = False  # configurable, and off => no claim without evidence
    assert "person_down" not in types(run_engine(engine_for(p, cam()), seq))


def test_new_horizontal_track_far_from_lost_upright_is_not_claimed(profile):
    seq = [[(1, "person", box(100, 250, 40, 100))] for _ in range(20)]
    seq += [[] for _ in range(3)]
    seq += [[(7, "person", box(600, 310, 100, 40))] for _ in range(40)]  # unrelated person lying far away
    assert "person_down" not in types(run_engine(engine_for(profile, cam()), seq))


def test_velocity_window_independent_of_timestamp_offset():
    """Regression: float rounding used to push the 0.4 s boundary observation out of the
    window on 15-40% of frames, silently turning a 4-frame velocity into a 5-frame one.
    Uses accelerating motion so the window length changes the result."""
    from vigil.types import Observation, TrackState

    def x(k):
        return 10.0 * k + 2.0 * k * k

    for offset in (0.0, 1792073729.0094368):
        tr = TrackState(1, "c", 0, "person")
        for k in range(300):
            t = offset + k / 10
            tr.history.append(Observation(t, (x(k), 0.0, x(k) + 50.0, 100.0)))
            tr.last_t = t
            if k >= 5:
                assert abs(tr.velocity(0.4)[0] - (x(k) - x(k - 4)) / 0.4) < 0.5, (offset, k)
