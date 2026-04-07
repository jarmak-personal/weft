#!/usr/bin/env python3
"""Generate an HTML dashboard for WEFT experiment reports."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import shutil
from typing import Any


def _fmt(v: Any, ndigits: int = 4) -> str:
    if v is None:
        return "-"
    try:
        x = float(v)
    except Exception:
        return html.escape(str(v))
    return f"{x:.{ndigits}f}"


def _exists_link(path_str: str | None) -> tuple[str, str]:
    if not path_str:
        return "-", ""
    p = Path(path_str)
    if not p.exists():
        return "(missing)", ""
    return p.name, p.resolve().as_uri()


def _img_thumb(path_str: str | None, height: int = 72) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.exists():
        return ""
    return f'<img src="{p.resolve().as_uri()}" style="height:{height}px;border-radius:6px;border:1px solid #ddd" loading="lazy" />'


def _safe_name(s: str) -> str:
    out = []
    for ch in s:
        if ch.isalnum() or ch in ("-", "_", "."):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def build_dashboard(
    report: dict[str, Any],
    report_path: Path,
    *,
    bundle_dir: Path | None = None,
    bundle_href_prefix: str = "",
) -> str:
    leaderboard = report.get("leaderboard", [])
    results = report.get("results", [])

    top_cards: list[str] = []
    for row in leaderboard[:5]:
        top_cards.append(
            """
            <div class="card">
              <div class="name">{name}</div>
              <div class="kpi">objective {obj}</div>
              <div class="sub">bpp {bpp} | psnr {psnr} | decode {dms} ms</div>
            </div>
            """.format(
                name=html.escape(str(row.get("profile", "-"))),
                obj=_fmt(row.get("avg_objective"), 6),
                bpp=_fmt(row.get("avg_bpp"), 4),
                psnr=_fmt(row.get("avg_psnr"), 3),
                dms=_fmt(row.get("avg_decode_ms"), 2),
            )
        )

    lb_rows: list[str] = []
    for row in leaderboard:
        lb_rows.append(
            """
            <tr>
              <td>{profile}</td>
              <td data-sort="{obj}">{obj_txt}</td>
              <td data-sort="{bpp}">{bpp_txt}</td>
              <td data-sort="{psnr}">{psnr_txt}</td>
              <td data-sort="{edge}">{edge_txt}</td>
              <td data-sort="{enc}">{enc_txt}</td>
              <td data-sort="{dec}">{dec_txt}</td>
            </tr>
            """.format(
                profile=html.escape(str(row.get("profile", "-"))),
                obj=float(row.get("avg_objective") or 1e99),
                obj_txt=_fmt(row.get("avg_objective"), 6),
                bpp=float(row.get("avg_bpp") or 1e99),
                bpp_txt=_fmt(row.get("avg_bpp"), 4),
                psnr=float(row.get("avg_psnr") or -1e99),
                psnr_txt=_fmt(row.get("avg_psnr"), 3),
                edge=float(row.get("avg_edge_mse") or 1e99),
                edge_txt=_fmt(row.get("avg_edge_mse"), 6),
                enc=float(row.get("avg_encode_ms") or 1e99),
                enc_txt=_fmt(row.get("avg_encode_ms"), 2),
                dec=float(row.get("avg_decode_ms") or 1e99),
                dec_txt=_fmt(row.get("avg_decode_ms"), 2),
            )
        )

    detail_rows: list[str] = []
    copied = 0
    for r in results:
        enc = (r.get("metadata") or {}).get("encode") or {}
        dec = (r.get("metadata") or {}).get("decode") or {}
        src_path = str((Path(report.get("dataset_dir", "")) / r.get("image", "")).resolve())
        weft_path = enc.get("output_path")
        dec_path = dec.get("output_path")

        if bundle_dir is not None:
            profile = _safe_name(str(r.get("profile", "profile")))
            image = _safe_name(str(r.get("image", "image")))

            def _copy_to_bundle(path_str: str | None, suffix: str) -> tuple[str, str]:
                nonlocal copied
                if not path_str:
                    return "(missing)", ""
                p = Path(path_str)
                if not p.exists():
                    return "(missing)", ""
                dst = bundle_dir / f"{profile}__{image}__{suffix}{p.suffix}"
                if not dst.exists():
                    shutil.copy2(p, dst)
                    copied += 1
                href = f"{bundle_href_prefix}{dst.name}" if bundle_href_prefix else dst.name
                return dst.name, html.escape(href)

            src_name, src_href = _copy_to_bundle(src_path, "src")
            dec_name, dec_href = _copy_to_bundle(dec_path, "decoded")
            weft_name, weft_href = _copy_to_bundle(weft_path, "weft")
            src_thumb = f'<img src="{src_href}" style="height:72px;border-radius:6px;border:1px solid #ddd" loading="lazy" />' if src_href else "-"
            dec_thumb = f'<img src="{dec_href}" style="height:72px;border-radius:6px;border:1px solid #ddd" loading="lazy" />' if dec_href else "-"
        else:
            weft_name, weft_href = _exists_link(weft_path)
            dec_name, dec_href = _exists_link(dec_path)
            src_name, src_href = _exists_link(src_path)
            src_thumb = _img_thumb(src_path) or "-"
            dec_thumb = _img_thumb(dec_path) or "-"

        detail_rows.append(
            """
            <tr>
              <td>{profile}</td>
              <td>{image}</td>
              <td>{src_thumb}</td>
              <td>{dec_thumb}</td>
              <td data-sort="{weft_bytes}">{weft_bytes_txt}</td>
              <td data-sort="{bpp}">{bpp_txt}</td>
              <td data-sort="{psnr}">{psnr_txt}</td>
              <td data-sort="{obj}">{obj_txt}</td>
              <td data-sort="{enc_ms}">{enc_ms_txt}</td>
              <td data-sort="{dec_ms}">{dec_ms_txt}</td>
              <td>{src_link}</td>
              <td>{dec_link}</td>
              <td>{weft_link}</td>
            </tr>
            """.format(
                profile=html.escape(str(r.get("profile", "-"))),
                image=html.escape(str(r.get("image", "-"))),
                src_thumb=src_thumb,
                dec_thumb=dec_thumb,
                weft_bytes=int(r.get("weft_bytes") or 0),
                weft_bytes_txt=str(int(r.get("weft_bytes") or 0)),
                bpp=float(r.get("bpp") or 1e99),
                bpp_txt=_fmt(r.get("bpp"), 4),
                psnr=float(r.get("psnr") or -1e99),
                psnr_txt=_fmt(r.get("psnr"), 3),
                obj=float(r.get("objective") or 1e99),
                obj_txt=_fmt(r.get("objective"), 6),
                enc_ms=float(r.get("encode_ms") or 1e99),
                enc_ms_txt=_fmt(r.get("encode_ms"), 2),
                dec_ms=float(r.get("decode_ms") or 1e99),
                dec_ms_txt=_fmt(r.get("decode_ms"), 2),
                src_link=(f'<a href="{src_href}">{html.escape(src_name)}</a>' if src_href else html.escape(src_name)),
                dec_link=(f'<a href="{dec_href}">{html.escape(dec_name)}</a>' if dec_href else html.escape(dec_name)),
                weft_link=(f'<a href="{weft_href}">{html.escape(weft_name)}</a>' if weft_href else html.escape(weft_name)),
            )
        )

    copied_txt = f" | bundled assets: {copied}" if bundle_dir is not None else ""

    return f"""<!doctype html>
<html lang=\"en\"> 
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width,initial-scale=1\" />
  <title>WEFT Experiment Dashboard</title>
  <style>
    :root {{ --bg:#f7f9fc; --fg:#0f172a; --muted:#475569; --card:#ffffff; --line:#dbe3ef; }}
    body {{ margin:0; font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; background:var(--bg); color:var(--fg); }}
    .wrap {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
    .meta {{ color:var(--muted); font-size:13px; margin-bottom: 16px; }}
    .cards {{ display:grid; grid-template-columns: repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-bottom: 18px; }}
    .card {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; }}
    .name {{ font-weight:700; font-size:14px; margin-bottom:6px; }}
    .kpi {{ font-size:13px; }}
    .sub {{ font-size:12px; color:var(--muted); margin-top:3px; }}
    .panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px; margin-bottom: 14px; overflow:auto; }}
    table {{ width:100%; border-collapse:collapse; font-size:12px; }}
    th, td {{ padding:7px 8px; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
    th {{ cursor:pointer; position:sticky; top:0; background: #eef4fb; }}
    tr:hover td {{ background:#f3f8ff; }}
    a {{ color:#0369a1; text-decoration:none; }}
    a:hover {{ text-decoration:underline; }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>WEFT Experiment Dashboard</h1>
    <div class=\"meta\">report: {html.escape(str(report_path))} | dataset: {html.escape(str(report.get('dataset_dir','-')))} | profiles: {len(report.get('profiles',[]))} | results: {len(results)}{copied_txt}</div>

    <div class=\"cards\">{''.join(top_cards)}</div>

    <div class=\"panel\">
      <h3>Leaderboard</h3>
      <table class=\"sortable\">
        <thead><tr><th>profile</th><th>objective</th><th>bpp</th><th>psnr</th><th>edge_mse</th><th>encode_ms</th><th>decode_ms</th></tr></thead>
        <tbody>{''.join(lb_rows)}</tbody>
      </table>
    </div>

    <div class=\"panel\">
      <h3>Per-Result Detail (Visual + Size)</h3>
      <table class=\"sortable\">
        <thead><tr>
          <th>profile</th><th>image</th><th>source</th><th>decoded</th>
          <th>weft_bytes</th><th>bpp</th><th>psnr</th><th>objective</th><th>encode_ms</th><th>decode_ms</th>
          <th>source_path</th><th>decoded_path</th><th>weft_path</th>
        </tr></thead>
        <tbody>{''.join(detail_rows)}</tbody>
      </table>
    </div>
  </div>
  <script>
    document.querySelectorAll('table.sortable').forEach((table) => {{
      table.querySelectorAll('th').forEach((th, col) => {{
        let asc = true;
        th.addEventListener('click', () => {{
          const tbody = table.querySelector('tbody');
          const rows = Array.from(tbody.querySelectorAll('tr'));
          rows.sort((a,b) => {{
            const da = a.children[col].dataset.sort;
            const db = b.children[col].dataset.sort;
            const va = (da !== undefined) ? parseFloat(da) : a.children[col].innerText.trim().toLowerCase();
            const vb = (db !== undefined) ? parseFloat(db) : b.children[col].innerText.trim().toLowerCase();
            if (typeof va === 'number' && !Number.isNaN(va) && typeof vb === 'number' && !Number.isNaN(vb)) {{
              return asc ? va - vb : vb - va;
            }}
            return asc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
          }});
          rows.forEach(r => tbody.appendChild(r));
          asc = !asc;
        }});
      }});
    }});
  </script>
</body>
</html>
"""


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate HTML dashboard from WEFT experiment_report.json")
    ap.add_argument("report_json", help="Path to experiment_report.json")
    ap.add_argument("output_html", nargs="?", default=None, help="Output HTML path (default: <report_dir>/dashboard.html)")
    ap.add_argument("--bundle-dir", default=None, help="Optional directory to copy source/decoded/weft files for portable dashboard")
    args = ap.parse_args()

    report_path = Path(args.report_json).resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))

    output_html = Path(args.output_html).resolve() if args.output_html else (report_path.parent / "dashboard.html").resolve()
    bundle_dir = Path(args.bundle_dir).resolve() if args.bundle_dir else None
    if bundle_dir is not None:
        bundle_dir.mkdir(parents=True, exist_ok=True)
    bundle_href_prefix = ""
    if bundle_dir is not None:
        bundle_href_prefix = f"{bundle_dir.name}/"

    html_doc = build_dashboard(
        report,
        report_path,
        bundle_dir=bundle_dir,
        bundle_href_prefix=bundle_href_prefix,
    )
    output_html.write_text(html_doc, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_html": str(output_html),
                "leaderboard_rows": len(report.get("leaderboard", [])),
                "result_rows": len(report.get("results", [])),
                "bundle_dir": str(bundle_dir) if bundle_dir is not None else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
