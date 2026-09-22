"""Create a portable visual review bundle for CubiCasa records skipped by export."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from PIL import Image


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def resolve(value: str, root: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def make_thumbnail(source: Path, target: Path) -> tuple[int, int]:
    # These are local CubiCasa source files selected by the supplied manifest,
    # not uploaded/untrusted image bytes. Large architectural plans are valid.
    previous = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = None
    try:
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.thumbnail((1200, 900))
            image.save(target, "JPEG", quality=86)
            return image.size
    finally:
        Image.MAX_IMAGE_PIXELS = previous


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="The graph manifest used for canonical export.")
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output.expanduser().resolve()
    assets = output / "assets"
    if (output / "index.html").exists():
        raise FileExistsError(f"Review exists: {output / 'index.html'}")
    assets.mkdir(parents=True, exist_ok=False)
    root = args.corpus_root.expanduser().resolve()
    by_id = {row["plan_id"]: row for row in read_jsonl(args.manifest.expanduser().resolve())}
    cards = []
    review_rows = []
    for number, error in enumerate(read_jsonl(args.errors.expanduser().resolve()), start=1):
        plan = by_id.get(error["plan_id"])
        if plan is None:
            cards.append(f"<article><h2>{html.escape(error['plan_id'])}</h2><pre>{html.escape(error['error'])}</pre><p>Missing from manifest.</p></article>")
            continue
        image = resolve(plan["image_path"], root)
        thumb_name = f"{number:04d}_{error['plan_id']}.jpg"
        thumbnail_size = make_thumbnail(image, assets / thumb_name)
        identity = plan.get("metadata", {}).get("source_dir", "unknown")
        review = {"plan_id": error["plan_id"], "source_identity": identity, "image_path": str(image),
                  "svg_path": str(resolve(plan["svg_path"], root)), "error": error["error"], "thumbnail": f"assets/{thumb_name}"}
        review_rows.append(review)
        cards.append(f'''<article><h2>{html.escape(error['plan_id'])}</h2>
<img src="assets/{html.escape(thumb_name)}" alt="{html.escape(error['plan_id'])} raster thumbnail">
<p><b>Source identity:</b> {html.escape(str(identity))}</p>
<p><b>Error:</b> {html.escape(error['error'])}</p>
<details><summary>Source paths</summary><pre>{html.escape(json.dumps(review, indent=2))}</pre></details></article>''')
    payload = json.dumps(review_rows, indent=2)
    document = f'''<!doctype html><html><head><meta charset="utf-8"><title>CubiCasa export errors</title>
<style>body{{font:16px system-ui;margin:28px;max-width:1200px}}article{{border:1px solid #ccd;padding:18px;margin:16px 0;border-radius:10px}}img{{max-width:100%;border:1px solid #aaa}}pre{{white-space:pre-wrap}}.note{{color:#555}}</style>
</head><body><h1>CubiCasa records skipped from EGTR training</h1><p class="note">These records were excluded from the canonical silver-training corpus. Review before any manual repair; source files are unchanged.</p>{''.join(cards)}
<script type="application/json" id="records">{html.escape(payload)}</script></body></html>'''
    (output / "index.html").write_text(document, encoding="utf-8")
    (output / "review_records.json").write_text(payload + "\n", encoding="utf-8")
    print(json.dumps({"errors_reviewed": len(review_rows), "output": str(output / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
