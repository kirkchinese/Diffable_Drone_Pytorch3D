#!/usr/bin/env bash
# 本地预览构建：权威源使用 SimSun/SimHei/KaiTi（仅 Windows/MiKTeX 可出），
# 本机缺这些字体，故复制一份临时替换为 Noto CJK 编译，仅供内容预览（非最终字体稿）。
# 用法：bash testscript/build_preview.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/docs/论文相关/5.29需要修订论文/thesis/thesis-latex"
BUILD="$SRC/.preview_build"
OUT="$ROOT/docs/论文相关/5.29需要修订论文/thesis/main_preview_noto.pdf"

rm -rf "$BUILD"; mkdir -p "$BUILD"
cp "$SRC"/main.tex "$BUILD"/
cp -r "$SRC"/chapters "$SRC"/figures "$BUILD"/
# 临时把权威字体替换为本机可用的 Noto CJK（不改动权威源）
sed -i \
  -e 's/\\setCJKmainfont{SimSun}\[AutoFakeBold=2.5\]/\\setCJKmainfont{Noto Serif CJK SC}[AutoFakeBold=2.5]/' \
  -e 's/\\setCJKsansfont{SimHei}/\\setCJKsansfont{Noto Sans CJK SC}/' \
  -e 's/\\setCJKmonofont{KaiTi}/\\setCJKmonofont{Noto Sans CJK SC}/' \
  "$BUILD"/main.tex

cd "$BUILD"
latexmk -xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
latexmk -xelatex -interaction=nonstopmode main.tex >/dev/null 2>&1 || true
cp "$BUILD"/main.pdf "$OUT"

echo "预览 PDF 已生成（Noto 字体，仅供内容预览）："
echo "  $OUT"
pdfinfo "$OUT" 2>/dev/null | grep -E "Pages" || true
UND=$(grep -ci "Citation.*undefined\|Reference.*undefined\|There were undefined" "$BUILD"/main.log || true)
echo "  undefined refs/cites: $UND"
