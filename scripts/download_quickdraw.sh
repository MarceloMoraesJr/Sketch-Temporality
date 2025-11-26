#categories list from: https://github.com/leosampaio/sketchformer/blob/master/prep_data/quickdraw/list_quickdraw.txt

while read p; do
  wget "https://storage.googleapis.com/quickdraw_dataset/sketchrnn/$p.npz" -P ./data/quickdraw/raw/
done < ./scripts/categories.txt
