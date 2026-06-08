#!/usr/bin/env bash
# 本地预览构建：权威字体 SimSun/SimHei/KaiTi 仅 Windows/MiKTeX 可出，
# 本机临时替换为 Noto CJK 编译，仅供内容与页数预览（非最终字体稿）。
# 用法：bash build_preview.sh
set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD="$DIR/.preview_build"
OUT="$DIR/残差选编版_preview_noto.pdf"

rm -rf "$BUILD"; mkdir -p "$BUILD"
cp "$DIR/main.tex" "$BUILD/"
cp -r "$DIR/figures" "$BUILD/"
sed -i \
  -e 's/\\setCJKmainfont{SimSun}\[AutoFakeBold=2.5\]/\\setCJKmainfont{Noto Serif CJK SC}[AutoFakeBold=2.5]/' \
  -e 's/\\setCJKfamilyfont{zhsong}{SimSun}\[AutoFakeBold=2.5\]/\\setCJKfamilyfont{zhsong}{Noto Serif CJK SC}[AutoFakeBold=2.5]/' \
  -e 's/\\setCJKfamilyfont{zhhei}{SimHei}/\\setCJKfamilyfont{zhhei}{Noto Sans CJK SC}/' \
  -e 's/\\setCJKfamilyfont{zhkai}{KaiTi}/\\setCJKfamilyfont{zhkai}{Noto Serif CJK SC}/' \
  "$BUILD/main.tex"

cd "$BUILD"
latexmk -xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
latexmk -xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
cp "$BUILD/main.pdf" "$OUT"

echo "预览 PDF 已生成（Noto 字体，仅供内容/页数预览）："
echo "  $OUT"
pdfinfo "$OUT" 2>/dev/null | grep -E "Pages" || true
UND=$(grep -ci "Citation.*undefined\|Reference.*undefined\|There were undefined" "$BUILD"/main.log || true)
echo "  undefined refs/cites: $UND"
