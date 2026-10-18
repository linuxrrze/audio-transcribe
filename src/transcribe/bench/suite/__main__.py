"""Benchmark suite: "who said what" on many recordings, conditions and settings.

    python -m transcribe.bench.suite build ami --split test            # 16 meetings
    python -m transcribe.bench.suite build voxconverse --split dev --limit 50
    python -m transcribe.bench.suite build synth --lang de --split dev --count 20
    python -m transcribe.bench.suite build synth --lang de --commonvoice ~/cv-corpus/de
    python -m transcribe.bench.suite add-session <session> [--dataset local --split test]
    python -m transcribe.bench.suite list

    python -m transcribe.bench.suite run NAME --data ami synth-de-piper --split dev
        [--conditions clean noise:10 phone] [--set speakers.diarize_threshold=0.4,0.5]
        [--set asr.model=small,large-v3-turbo] [--live] [--full] [--limit N]
    python -m transcribe.bench.suite report NAME [--baseline FILE --tolerance 0.02]
        [--compare SECTION.KEY=VALUE]

Tune on ``--split dev``; report ``--split test`` once settings are chosen. Results go to
<bench dir>/results/NAME (results.jsonl, summary.json, summary.md); runs are resumable
and Whisper/diarization outputs are cached, so adding settings or conditions only
computes what is new. See docs/BENCHMARKS.md, "Benchmark suite".
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transcribe.bench.suite import run, sources
from transcribe.bench.suite.data import Dataset, available, bench_dir
from transcribe.core.config import Config


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m transcribe.bench.suite",
                                 description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="download/convert or generate a dataset")
    b.add_argument("source", choices=["ami", "voxconverse", "synth"])
    b.add_argument("--split", default="dev", choices=["dev", "test"])
    b.add_argument("--limit", type=int, help="at most this many recordings")
    b.add_argument("--lang", default="de", choices=["de", "en"], help="synth: language")
    b.add_argument("--count", type=int, default=10, help="synth: conversations")
    b.add_argument("--speakers", default="2-6", help="synth: min-max speakers")
    b.add_argument("--turns", default="20-40", help="synth: min-max turns")
    b.add_argument("--commonvoice", type=Path,
                   help="synth: Common Voice folder (validated.tsv, clips/) for real voices")
    b.add_argument("--name", help="synth: dataset name (default synth-<lang>-<source>)")

    a = sub.add_parser("add-session", help="use a corrected session as reference")
    a.add_argument("sessions", nargs="+", type=Path)
    a.add_argument("--dataset", default="local")
    a.add_argument("--split", default="test", choices=["dev", "test"])

    sub.add_parser("list", help="datasets that are built")

    r = sub.add_parser("run", help="evaluate datasets")
    r.add_argument("name", help="name of this run (results folder)")
    r.add_argument("--data", nargs="+", required=True)
    r.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    r.add_argument("--conditions", nargs="+", default=["clean"])
    r.add_argument("--set", action="append", default=[], metavar="SECTION.KEY=V1,V2",
                   help="setting to vary (repeatable; all combinations are run)")
    r.add_argument("--live", action="store_true", help="also score live speaker labels")
    r.add_argument("--full", action="store_true",
                   help="the complete offline pipeline (slower, no separate caching)")
    r.add_argument("--limit", type=int, help="at most this many recordings per dataset")
    r.add_argument("--user-config", action="store_true",
                   help="start from your config.toml instead of the defaults")

    p = sub.add_parser("report", help="summary table (and regression check)")
    p.add_argument("name")
    p.add_argument("--baseline", type=Path, help="summary.json to compare with")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="allowed absolute worsening per rate (default 2 points)")
    p.add_argument("--save-baseline", type=Path, help="write this run's summary as baseline")
    p.add_argument("--compare", metavar="SECTION.KEY=VALUE",
                   help="also compare every setting with this one on the same recordings "
                        "(paired; e.g. asr.vad_threshold_narrowband=0.25)")

    args = ap.parse_args(argv)
    if args.cmd == "build":
        if args.source == "ami":
            ds = sources.build_ami(args.split, args.limit)
        elif args.source == "voxconverse":
            ds = sources.build_voxconverse(args.split, args.limit)
        else:
            lo, hi = (int(x) for x in args.speakers.split("-"))
            t0, t1 = (int(x) for x in args.turns.split("-"))
            ds = sources.build_synth(args.lang, args.split, args.count, (lo, hi), (t0, t1),
                                     args.commonvoice, args.name)
        print(f"{ds.name}: {len(list(ds.recordings()))} recordings in {ds.dir}")
    elif args.cmd == "add-session":
        for s in args.sessions:
            rec = sources.add_session(s, args.dataset, args.split)
            print(f"{args.dataset}/{rec.id}: {len(rec.turns)} turns, "
                  f"{len(rec.speakers)} speakers")
    elif args.cmd == "list":
        for name in available():
            recs = list(Dataset(name).recordings())
            hours = sum(r.duration for r in recs) / 3600
            splits = {s: sum(r.split == s for r in recs) for s in ("dev", "test")}
            print(f"{name:20} {len(recs):4} recordings {hours:6.1f} h  "
                  f"dev {splits['dev']}  test {splits['test']}")
    elif args.cmd == "run":
        base = Config.load() if args.user_config else Config()
        out = bench_dir() / "results" / args.name
        run.run(args.data, args.split, args.conditions, run.parse_grid(args.set), base, out,
                args.limit, args.live, args.full)
        return _report(out, None, 0.0, None)
    else:
        return _report(bench_dir() / "results" / args.name, args.baseline, args.tolerance,
                       args.save_baseline, args.compare)
    return 0


def _report(out: Path, baseline: Path | None, tolerance: float,
            save: Path | None, reference: str | None = None) -> int:
    results = run.load_results(out / "results.jsonl")
    rows = run.summarize(results)
    (out / "summary.json").write_text(json.dumps(rows, indent=1))
    table = run.markdown(rows)
    (out / "summary.md").write_text(table)
    print(table)
    if reference:
        try:
            pairs = run.paired(results, reference)
        except ValueError as e:
            print(e, file=sys.stderr)
            return 2
        (out / "paired.json").write_text(json.dumps(pairs, indent=1))
        paired_table = run.paired_markdown(pairs)
        (out / "paired.md").write_text(paired_table)
        print(paired_table)
    if save:
        save.write_text(json.dumps(rows, indent=1))
        print(f"Baseline written to {save}")
    if baseline:
        problems = run.compare(rows, json.loads(baseline.read_text()), tolerance)
        for p in problems:
            print(f"REGRESSION: {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"No regression against {baseline} (tolerance {100 * tolerance:.0f} points)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
