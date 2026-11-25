"""Figma 슬라이드 화면(reports/slides/png/slide_XX.png)을 한 장씩 꽉 채운 16:9 PPTX로 묶는다.

PNG는 Figma 슬라이드를 1920×1080으로 내려받은 것이다. 글자를 고칠 수 있는 PPTX가 필요하면
Figma에서 File → Export → PowerPoint로 내보낸다.
사용: PYTHONPATH=<python-pptx 설치 경로> env-gpu/bin/python scripts/make_slides_pptx.py
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "reports" / "slides" / "png"
OUT = ROOT / "reports" / "slides" / "gpu_reclaim_final.pptx"

prs = Presentation()
prs.slide_width, prs.slide_height = Emu(12192000), Emu(6858000)  # 13.333 × 7.5 in (16:9)
blank = prs.slide_layouts[6]
pngs = sorted(SRC.glob("slide_*.png"))
for png in pngs:
    slide = prs.slides.add_slide(blank)
    slide.shapes.add_picture(str(png), 0, 0, width=prs.slide_width, height=prs.slide_height)
prs.save(OUT)
print("written", OUT, f"{OUT.stat().st_size / 1e6:.1f} MB, {len(pngs)} slides")
