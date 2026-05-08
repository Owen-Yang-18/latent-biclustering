# Real-World Data

Run commands from `release/`. Preprocessors write:

```text
data/processed/<dataset>/X.npz
data/processed/<dataset>/row_mapping.json
data/processed/<dataset>/col_mapping.json
data/processed/<dataset>/stats.json
```

Install optional preprocessing dependencies:

```bash
pip install -r requirements-realworld.txt
mkdir -p data/raw data/processed
```

## Online Retail

Source: https://archive.ics.uci.edu/dataset/352/online+retail

```bash
curl -L -o data/raw/online_retail.zip \
  "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
unzip -j data/raw/online_retail.zip -d data/raw/online_retail
python scripts/preprocess_realworld.py \
  --dataset retail \
  --input "data/raw/online_retail/Online Retail.xlsx" \
  --output-dir data/processed \
  --min-user 5 \
  --min-item 5
```

## Last.fm 2K

Source: https://grouplens.org/datasets/hetrec-2011/

```bash
curl -L -o data/raw/hetrec2011-lastfm-2k.zip \
  https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
unzip data/raw/hetrec2011-lastfm-2k.zip -d data/raw
python scripts/preprocess_realworld.py \
  --dataset lastfm \
  --input data/raw/hetrec2011-lastfm-2k/user_artists.dat \
  --output-dir data/processed \
  --min-user 5 \
  --min-item 5
```

## MovieLens 1M

Source: https://grouplens.org/datasets/movielens/1m/

```bash
curl -L -o data/raw/ml-1m.zip \
  https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip data/raw/ml-1m.zip -d data/raw
python scripts/preprocess_realworld.py \
  --dataset movielens \
  --input data/raw/ml-1m/ratings.dat \
  --output-dir data/processed \
  --min-user 5 \
  --min-item 5 \
  --movielens-min-rating 4
```

## Amazon Reviews 2023: All Beauty

Source: https://amazon-reviews-2023.github.io/

```bash
curl -L -o data/raw/All_Beauty.jsonl.gz \
  https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/raw/review_categories/All_Beauty.jsonl.gz
python scripts/preprocess_realworld.py \
  --dataset amazon \
  --input data/raw/All_Beauty.jsonl.gz \
  --output-dir data/processed \
  --min-user 5 \
  --min-item 5 \
  --amazon-item-key asin
```

## Model Run

```bash
python scripts/run_matrix_pipeline.py \
  --matrix data/processed/lastfm/X.npz \
  --config configs/realworld_pipeline.json \
  --output-dir outputs/lastfm
```
