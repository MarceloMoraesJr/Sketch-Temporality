curl -L "https://github.com/KeLi-SketchX/SketchX-PRIS-Dataset/archive/refs/heads/master.zip" -o "repo.zip"
mkdir -p "./data/spg/raw"
unzip -q "repo.zip" "SketchX-PRIS-Dataset-master/Perceptual Grouping/*.ndjson" -d "./data/spg/raw"
unzip -q "repo.zip" "SketchX-PRIS-Dataset-master/Group ID/*.txt" -d "./data/spg/categories"
mv ./data/spg/raw/SketchX-PRIS-Dataset-master/Perceptual\ Grouping/* ./data/spg/raw
mv ./data/spg/categories/SketchX-PRIS-Dataset-master/Group\ ID/* ./data/spg/categories
rm -rf "repo.zip" "./data/spg/raw/SketchX-PRIS-Dataset-master/" "./data/spg/categories/SketchX-PRIS-Dataset-master/"

for file in ./data/spg/categories/*-*; do
  [ -e "$file" ] || continue
  new_name="${file//-/_}"
  mv -- "$file" "$new_name"
done