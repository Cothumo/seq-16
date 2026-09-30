#!/usr/bin/env python3
"""Hardware stress test for the seq-16 Grid sequencer (EN16 profile).

Acts as the DAW: sends MIDI clock and transport to the Grid, listens to the
notes it plays back, and checks that every step fires on time, nothing is
missed or doubled, and no notes hang after Stop.

Setup
  pip install mido python-rtmidi
  1. Close the DAW (or stop it sending clock to the Grid).
  2. Program a pattern on the EN16 - any notes, any step length.
  3. Make playback deterministic: every step at 100% chance, Euclid off,
     arp off, record off, clock division 1/16 (or pass --ticks-per-step).
  4. Don't touch the controller while the test runs.

Run
  python stress_test.py --list          # show MIDI ports
  python stress_test.py                 # full run (a few minutes)
  python stress_test.py --soak-minutes 30

It writes seq16-stress-log.json with every message sent and received.
"""
import argparse
import bisect
import json
import random
import statistics
import sys
import threading
import time
from collections import Counter

try:
    import mido
except ImportError:
    sys.exit("Needs mido and python-rtmidi:  pip install mido python-rtmidi")

TICKS_PER_16TH = 6  # SPP counts 16th notes; MIDI clock is 24 per quarter
CLOCK = mido.Message("clock")


class Rig:
    """Plays the DAW role and records everything that comes back."""

    def __init__(self, out_port, in_port, tps):
        self.tps = tps
        self.lock = threading.Lock()
        self.tx_lock = threading.Lock()
        self.clocks = []  # (time, song position or None when stopped, resumed)
        self.ons = []  # (time, note)
        self.active = set()
        self.log = []
        self.running = False
        self.pos = 0
        self.resumed = False
        self.t0 = time.perf_counter()
        self.out = out_port
        in_port.callback = self._rx

    def _rx(self, msg):
        t = time.perf_counter()
        with self.lock:
            if msg.type == "note_on" and msg.velocity > 0:
                self.ons.append((t, msg.note))
                self.active.add(msg.note)
            elif msg.type in ("note_on", "note_off"):
                self.active.discard(msg.note)
            self.log.append([round(t - self.t0, 6), "rx", str(msg)])

    def send(self, msg, logged=True):
        with self.tx_lock:
            self.out.send(msg)
        if logged:
            with self.lock:
                self.log.append([round(time.perf_counter() - self.t0, 6), "tx", str(msg)])

    # transport, tracking the song position the way a DAW would
    def start(self):
        self.send(mido.Message("start"))
        with self.lock:
            self.running, self.pos, self.resumed = True, 0, True

    def cont(self):
        self.send(mido.Message("continue"))
        with self.lock:
            self.running, self.resumed = True, True

    def stop(self):
        self.send(mido.Message("stop"))
        with self.lock:
            self.running = False

    def songpos(self, p):
        self.send(mido.Message("songpos", pos=p))
        with self.lock:
            self.pos = p * TICKS_PER_16TH

    def clock(self, bpm, n, at=None):
        """Send n clock ticks at bpm; at maps tick index -> callable run just before it."""
        dt = 60.0 / (bpm * 24)
        nxt = time.perf_counter()
        for i in range(n):
            if at and i in at:
                at[i]()
            while True:
                now = time.perf_counter()
                if now >= nxt:
                    break
                if nxt - now > 0.002:
                    time.sleep(nxt - now - 0.0015)
            # record before sending so a fast reply can never predate its own clock
            t = time.perf_counter()
            with self.lock:
                if self.running:
                    boundary = self.pos % self.tps == 0
                    self.clocks.append((t, self.pos, self.resumed and boundary))
                    if boundary:
                        self.resumed = False
                    self.pos += 1
                else:
                    self.clocks.append((t, None, False))
            self.send(CLOCK, logged=False)
            nxt += dt

    def idle(self, bpm, seconds):
        self.clock(bpm, max(1, round(seconds * bpm * 24 / 60)))

    def mark(self):
        with self.lock:
            return len(self.clocks), len(self.ons)


def attribute(clocks, ons, tps):
    """Map each received note-on to the step-boundary clock that triggered it."""
    times = [c[0] for c in clocks]
    got, lat, strays = {}, [], []
    for t, note in ons:
        k = bisect.bisect_right(times, t) - 1
        j = k
        while j >= 0 and not (clocks[j][1] is not None and clocks[j][1] % tps == 0):
            j -= 1
        if j < 0 or k - j >= tps:
            strays.append(note)
            continue
        got.setdefault(j, []).append(note)
        lat.append(t - clocks[j][0])
    return got, lat, strays


class Result:
    def __init__(self, name):
        self.name = name
        self.steps = self.missing = self.extra = self.doubled = self.strays = 0
        self.hanging = []
        self.lat = []
        self.notes = []

    def failed(self):
        return any([self.missing, self.extra, self.doubled, self.strays, self.hanging])

    def line(self):
        s = f"{'FAIL' if self.failed() else 'PASS'}  {self.name:<22} steps={self.steps:<5}"
        s += f" missing={self.missing} extra={self.extra} doubled={self.doubled} stray={self.strays}"
        s += f" hanging={len(self.hanging)}"
        if self.lat:
            ms = sorted(x * 1000 for x in self.lat)
            p99 = ms[min(len(ms) - 1, int(len(ms) * 0.99))]
            s += f"  latency ms med={statistics.median(ms):.1f} p99={p99:.1f} max={ms[-1]:.1f}"
        for n in self.notes[:8]:
            s += "\n        " + n
        if len(self.notes) > 8:
            s += f"\n        ... {len(self.notes) - 8} more in the log"
        return s

    def summary(self):
        return {k: v for k, v in self.__dict__.items() if k != "lat"} | {"latency_ms": [round(x * 1000, 2) for x in self.lat]}


def check(rig, since, ref, name):
    """Compare every step the device should have played since `since` with the reference pattern."""
    c0, o0 = since
    with rig.lock:
        clocks, ons = rig.clocks[:], rig.ons[o0:]
    got, lat, strays = attribute(clocks, ons, rig.tps)
    r = Result(name)
    r.lat, r.strays = lat, len(strays)
    if strays:
        r.notes.append(f"notes off the step grid: {strays[:10]}")
    period = len(ref)
    for j in range(c0, len(clocks)):
        _, pos, resumed = clocks[j]
        if pos is None or pos % rig.tps:
            continue
        r.steps += 1
        step = (pos // rig.tps) % period
        g = Counter(got.get(j, []))
        exp = ref[step]
        miss = exp - set(g)
        extra = set() if resumed else set(g) - exp  # tied notes retrigger on the first step after Start/Continue
        dbl = [n for n, c in g.items() if c > 1]
        r.missing += len(miss)
        r.extra += len(extra)
        r.doubled += len(dbl)
        if miss or extra or dbl:
            r.notes.append(f"step {step + 1} (song pos {pos}): expected {sorted(exp)} got {sorted(g.elements())}")
    return r


def hang_check(rig, r, label):
    time.sleep(0.05)
    with rig.lock:
        left = sorted(rig.active)
    if left:
        r.hanging.append(label)
        r.notes.append(f"notes still on after {label}: {left}")


# ---------------------------------------------------------------- tests

def learn(rig, bpm):
    """Play 64 steps from Start and work out the repeating pattern."""
    since = rig.mark()
    rig.start()
    rig.clock(bpm, 64 * rig.tps)
    rig.stop()
    rig.idle(bpm, 0.3)
    with rig.lock:
        clocks, ons = rig.clocks[:], rig.ons[since[1]:]
    got, lat, strays = attribute(clocks, ons, rig.tps)
    seq = [frozenset(got.get(j, [])) for j in range(since[0], len(clocks))
           if clocks[j][1] is not None and clocks[j][1] % rig.tps == 0]
    if strays:
        sys.exit(f"Notes arrived off the step grid {strays[:10]}. Turn the arp off and check --ticks-per-step.")
    if not any(seq):
        sys.exit("No notes came back. Program a pattern, check the port, and make sure the EN16 profile is loaded.")
    for p in range(1, 33):
        if all(seq[i] == seq[i + p] for i in range(1, len(seq) - p)):
            ref = [set(seq[p + j]) for j in range(p)]
            print(f"learned pattern: repeats every {p} steps, {sum(len(s) for s in ref)} note-ons per loop")
            return ref
    sys.exit("Pattern doesn't repeat. Set every step to 100% chance and turn Euclid/arp/record off.")


def tempo_sweep(rig, ref, bpms):
    out = []
    for bpm in bpms:
        since = rig.mark()
        rig.start()
        rig.clock(bpm, 2 * len(ref) * rig.tps)
        rig.stop()
        rig.idle(bpm, 0.2)
        r = check(rig, since, ref, f"tempo {bpm} BPM")
        hang_check(rig, r, "stop")
        out.append(r)
    return out


def transport_storm(rig, ref, n, bpm, rnd):
    """Random Start / Stop / Continue / Song Position jumps at random moments."""
    since = rig.mark()
    rig.start()
    hangs = Result("")
    for i in range(n):
        rig.clock(bpm, rnd.randint(3, 8 * rig.tps))
        rig.stop()
        rig.clock(bpm, rnd.randint(1, 6))
        hang_check(rig, hangs, f"stop #{i + 1}")
        action = rnd.choice(["continue", "songpos", "start"])
        if action == "songpos":
            rig.songpos(rnd.randint(0, 255))
            rig.clock(bpm, rnd.randint(0, 3))
            rig.cont()
        elif action == "continue":
            rig.cont()
        else:
            rig.start()
    rig.clock(bpm, 4 * rig.tps)
    rig.stop()
    rig.idle(bpm, 0.2)
    r = check(rig, since, ref, f"transport storm x{n}")
    r.hanging, r.notes = hangs.hanging, hangs.notes + r.notes
    return r


def input_flood(rig, ref, bpm, rate, rnd):
    """Hammer the MIDI input with notes/CC/pitch bend while the sequencer plays."""
    stop = threading.Event()
    sent = set()

    def flood():
        dt = 1.0 / rate
        while not stop.is_set():
            k = rnd.random()
            if k < 0.5:
                n = rnd.randint(0, 127)
                rig.send(mido.Message("note_on", channel=rnd.randint(0, 15), note=n, velocity=rnd.randint(1, 127)), logged=False)
                sent.add(n)
            elif k < 0.7 and sent:
                n = sent.pop()
                rig.send(mido.Message("note_off", note=n), logged=False)
            elif k < 0.9:
                rig.send(mido.Message("control_change", control=rnd.choice([1, 7, 10, 64, 74]), value=rnd.randint(0, 127)), logged=False)
            else:
                rig.send(mido.Message("pitchwheel", pitch=rnd.randint(-8192, 8191)), logged=False)
            time.sleep(dt)

    since = rig.mark()
    th = threading.Thread(target=flood, daemon=True)
    rig.start()
    th.start()
    rig.clock(bpm, 4 * len(ref) * rig.tps)
    stop.set()
    th.join()
    for n in sent:
        rig.send(mido.Message("note_off", note=n), logged=False)
    rig.stop()
    rig.idle(bpm, 0.2)
    r = check(rig, since, ref, f"input flood {rate}/s")
    hang_check(rig, r, "stop")
    return r


def all_notes_off(rig, ref, bpm, rnd):
    """CC 123 while running must silence everything straight away."""
    r = Result("all-notes-off CC123")
    tries = 0
    rig.start()
    for i in range(40):
        if tries >= 10:
            break
        rig.clock(bpm, rig.tps * rnd.randint(1, 4) + 1)  # one tick after a step fires
        with rig.lock:
            sounding = bool(rig.active)
        if not sounding:
            continue
        tries += 1
        rig.send(mido.Message("control_change", control=123, value=0))
        hang_check(rig, r, f"CC123 #{tries}")
    rig.stop()
    rig.idle(bpm, 0.2)
    r.steps = tries
    if not tries:
        r.notes.append("skipped: no notes were sounding when CC123 was sent")
    return r


def soak(rig, ref, bpm, minutes):
    since = rig.mark()
    rig.start()
    steps = int(minutes * 60 * bpm * 24 / 60 / rig.tps)
    rig.clock(bpm, steps * rig.tps)
    rig.stop()
    rig.idle(bpm, 0.2)
    r = check(rig, since, ref, f"soak {minutes:g} min")
    hang_check(rig, r, "stop")
    return r


# ---------------------------------------------------------------- main

def pick(names, want):
    hits = [n for n in names if want.lower() in n.lower()]
    if not hits:
        sys.exit(f"No MIDI port matching '{want}'. Found: {names}. Use --port.")
    return hits[0]


def run(rig, a):
    rnd = random.Random(a.seed)
    print("learning the pattern at 120 BPM ...")
    ref = learn(rig, 120)
    results = []
    print("tempo sweep ...")
    results += tempo_sweep(rig, ref, [int(x) for x in a.bpms.split(",")])
    print("transport storm ...")
    results.append(transport_storm(rig, ref, a.storm, 140, rnd))
    print("input flood ...")
    results.append(input_flood(rig, ref, 160, a.flood_rate, rnd))
    print("all notes off ...")
    results.append(all_notes_off(rig, ref, 120, rnd))
    if a.soak_minutes:
        print(f"soak {a.soak_minutes:g} min ...")
        results.append(soak(rig, ref, 120, a.soak_minutes))
    rig.send(mido.Message("stop"))
    print()
    for r in results:
        print(r.line())
    bad = [r for r in results if r.failed()]
    print(f"\n{'ALL PASSED' if not bad else f'{len(bad)} TEST(S) FAILED'}")
    with open(a.log, "w") as f:
        json.dump({"seed": a.seed, "ticks_per_step": rig.tps, "pattern": [sorted(s) for s in ref],
                   "results": [r.summary() for r in results], "events": rig.log}, f)
    print(f"log written to {a.log}")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="list MIDI ports and exit")
    ap.add_argument("--port", default="grid", help="part of the Grid's MIDI port name (default: grid)")
    ap.add_argument("--ticks-per-step", type=int, default=6, help="clock ticks per step: 6 = 1/16 (default), 12 = 1/8")
    ap.add_argument("--bpms", default="60,90,120,160,200,240,300", help="tempos for the sweep")
    ap.add_argument("--storm", type=int, default=100, help="number of random transport changes")
    ap.add_argument("--flood-rate", type=int, default=300, help="incoming MIDI messages per second during the flood")
    ap.add_argument("--soak-minutes", type=float, default=0, help="add a long run at 120 BPM")
    ap.add_argument("--seed", type=int, default=random.randrange(1 << 30), help="random seed, to repeat a run")
    ap.add_argument("--log", default="seq16-stress-log.json")
    a = ap.parse_args()

    if a.list:
        print("inputs: ", mido.get_input_names())
        print("outputs:", mido.get_output_names())
        return 0
    out_name = pick(mido.get_output_names(), a.port)
    in_name = pick(mido.get_input_names(), a.port)
    print(f"out: {out_name}\nin:  {in_name}\nseed: {a.seed}")
    with mido.open_output(out_name) as out, mido.open_input(in_name) as inp:
        rig = Rig(out, inp, a.ticks_per_step)
        try:
            return run(rig, a)
        finally:
            out.send(mido.Message("stop"))
            out.send(mido.Message("control_change", control=123, value=0))


if __name__ == "__main__":
    sys.exit(main())
