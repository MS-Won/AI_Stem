"""명령줄 진입점: python -m mrengine.cli <input> --out <dir>"""
from __future__ import annotations

import argparse
from pathlib import Path

from mrengine.pipeline import PipelineConfig, run_pipeline
from mrengine.separate import DEFAULT_MODELS, MODEL_DIR, AudioSeparatorBackend


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mrengine", description="합창을 인식하는 MR 생성")
    p.add_argument("input", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--ensemble", choices=["avg", "min_mag", "max_mag"], default="avg")
    p.add_argument("--models", default=",".join(DEFAULT_MODELS), help="쉼표로 구분한 모델 파일명")
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    work = args.out / "_work"
    separators = [AudioSeparatorBackend(m, MODEL_DIR, work) for m in models]
    cfg = PipelineConfig(strength=args.strength, ensemble_method=args.ensemble)
    rep = run_pipeline(args.input, args.out, separators, cfg)

    secs = rep["sections"]
    choir = sum(1 for s in secs if s["kind"] == "choir")
    total = sum(rep["timings_s"].values())
    print(f"출력: {args.out}")
    print(f"구간 {len(secs)}개 (합창 {choir}개), 총 {total:.1f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
