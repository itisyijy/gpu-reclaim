"""reports/slide_report/README.md → 그림을 본문에 넣은 A4 세로 PDF (reports/slide_report/slide_report.pdf).

마크다운을 HTML로 바꾸고, 본문에서 `figures/*.png`를 처음 언급한 줄 바로 뒤에 그 그림을 넣은 뒤,
headless Chrome으로 인쇄한다. 사용: env-gpu/bin/python scripts/report_to_pdf.py
"""
import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports" / "slide_report"
SRC, HTML, PDF = REP / "README.md", REP / "slide_report.html", REP / "slide_report.pdf"

text = SRC.read_text()
# 파일 목록·재현 절(3절)은 그림을 다시 넣지 않는다
body, _, tail = text.partition("\n## 3. 파일 목록")
seen, out, pending = set(), [], []


def flush():
    for fig in pending:
        out.extend(["", f'<div class="fig"><img src="figures/{fig}"></div>', ""])
    pending.clear()


for line in body.splitlines():
    # 그림을 언급한 목록 항목과 그 하위 항목(들여쓴 줄)이 끝난 뒤에 그림을 넣는다 (목록 중간에 넣으면 목록이 깨짐)
    if pending and not line.startswith(" "):
        flush()
    out.append(line)
    for fig in re.findall(r"figures/(fig\w+\.png)", line):
        if fig not in seen:
            seen.add(fig)
            pending.append(fig)
flush()
md_text = "\n".join(out) + ("\n## 3. 파일 목록" + tail if tail else "")

# Python-Markdown 규칙 보정: (1) 목록·표 앞에 빈 줄이 없으면 앞 문단에 붙으므로 빈 줄 삽입
# (2) 하위 목록 들여쓰기는 4칸이어야 하므로 2칸 단위를 4칸으로 (3) 굵은 글씨로 시작하는 연속 줄은 줄바꿈 유지
LIST = re.compile(r"^(\s*)([-*]|\d+\.)\s")
fixed, prev, in_code = [], "", False
for line in md_text.splitlines():
    if line.startswith("```"):
        in_code = not in_code
    if not in_code:
        m = LIST.match(line)
        if m:
            line = " " * (len(m.group(1)) * 2) + line.lstrip()
        starts_block = bool(m) or line.startswith("|")
        prev_block = bool(LIST.match(prev)) or prev.startswith("|") or prev.startswith("  ")
        if starts_block and prev.strip() and not prev_block:
            fixed.append("")
        if line.startswith("**") and prev.startswith("**") and fixed and fixed[-1].strip():
            fixed[-1] += "  "          # 마크다운 강제 줄바꿈
    fixed.append(line)
    prev = line
md_text = "\n".join(fixed)

html_body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists"])
# 슬라이드 구성안(S1…)과 부록(A1…)은 제목 앞에서 쪽이 넘어가도 되게 표시
html_body = re.sub(r"<h3>(S\d+\.|A\d+\.)", r'<h3 class="slide">\1', html_body)

css = """
@page { size: A4 portrait; margin: 14mm 13mm 14mm 13mm; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body { font-family: 'Noto Sans CJK KR', 'Noto Sans KR', sans-serif; font-size: 9.6pt; line-height: 1.55; color: #1d1d1f; }
h1 { font-size: 17pt; margin: 0 0 4mm; border-bottom: 2px solid #1f6fb4; padding-bottom: 2mm; }
h2 { font-size: 13.5pt; margin: 7mm 0 3mm; color: #1f6fb4; break-after: avoid; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 5mm 0 2mm; padding: 1.2mm 2.5mm; background: #eef4fb; border-left: 3px solid #1f6fb4;
     break-after: avoid; page-break-after: avoid; }
p, li { margin: 1mm 0; }
ul, ol { padding-left: 5mm; margin: 1mm 0; }
blockquote { margin: 2mm 0; padding: 2mm 3.5mm; background: #f6f8fa; border-left: 3px solid #8c8c8c; }
blockquote p { margin: 0.6mm 0; }
table { border-collapse: collapse; width: 100%; margin: 2mm 0 3mm; font-size: 8.6pt; break-inside: auto; }
tr { break-inside: avoid; page-break-inside: avoid; }
thead { display: table-header-group; }
th, td { border: 1px solid #cfd6de; padding: 1mm 1.6mm; vertical-align: top; word-break: keep-all; overflow-wrap: anywhere; }
th { background: #f0f3f6; text-align: left; }
code { font-family: 'Noto Sans Mono CJK KR', monospace; font-size: 8.4pt; background: #f3f4f6; padding: 0 1mm; border-radius: 2px; }
pre { background: #f3f4f6; padding: 2mm 3mm; font-size: 8.2pt; white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
.fig { margin: 2mm 0 3mm; text-align: center; break-inside: avoid; page-break-inside: avoid; }
.fig img { max-width: 100%; max-height: 105mm; }
hr { border: none; border-top: 1px solid #d0d7de; margin: 4mm 0; }
"""
HTML.write_text(f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>통합 결과 보고서</title>'
                f"<style>{css}</style></head><body>{html_body}</body></html>")

subprocess.run(["google-chrome", "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
                f"--print-to-pdf={PDF}", HTML.as_uri()], check=True, capture_output=True, timeout=180)
print("written", PDF, f"{PDF.stat().st_size / 1e6:.2f} MB, 그림 {len(seen)}장")
