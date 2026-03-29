# contempArt Data

Data from the contempArt dataset (Huckle, Garcia, Nakashima, ECCV Workshop 2020).
442 early-career artists from 15 German art schools. 14,559 artworks.

## Structure

```
data/
  images/
    visart2020/             ← artwork images organized by artist (537 folders)
  metadata/
    artists.csv             ← per-artist demographics and Instagram metrics (cleaned, comma-separated)
    images.csv              ← per-image metadata: path, timestamps, likes, dimensions (cleaned, comma-separated)
    artists_raw.csv         ← original finalData.csv (semicolon-separated, 35 cols)
    images_raw.csv          ← original imageData.csv (semicolon-separated, 13 cols)
    artists_original.xlsx   ← original spreadsheet
  social/
    edgelist.csv            ← full Instagram follower graph (target, source, type). 456k edges.
    edgelist_raw.csv        ← original with capitalized column names
    followers/              ← per-artist CSVs listing their followers (860 files)
    following/              ← per-artist CSVs listing who they follow (918 files)
```

## artists.csv columns (25, cleaned from 35)

| Column | Coverage | Description |
|---|---|---|
| artist_id | 442/442 | Unique artist identifier (folder name) |
| full_name | 442/442 | Artist's full name |
| school | 442/442 | Art school (15 schools) |
| east_german | 442/442 | Whether the school is in former East Germany |
| professor_class | 420/442 | Professor/class name at the school |
| gender | 440/442 | F or M |
| country_iso3 | 234/442 | ISO 3166-1 alpha-3 country code |
| continent | 234/442 | Europe, Asia, Americas |
| region | 234/442 | World Bank region grouping |
| instagram_handle | 366/442 | Instagram username |
| follower_count | 359/442 | Number of Instagram followers |
| following_count | 359/442 | Number of Instagram following |
| posts_count | 359/442 | Number of Instagram posts |
| img_count | 442/442 | Number of images in dataset |
| avg_likes | 200/442 | Mean likes per image (Instagram only) |
| avg_comments | 200/442 | Mean comments per image |
| avg_file_size | 442/442 | Mean file size in bytes |
| avg_width | 442/442 | Mean image width in pixels |
| avg_height | 442/442 | Mean image height in pixels |
| avg_aspect_ratio | 442/442 | Mean width/height ratio |

Dropped from original: univNumeric, good, studentStatus, eurocontrolStatfor,
website2, createTime, nationality (kept iso3c as country_iso3), avgNocomm,
duplicate instagram handle columns, region23.

## images.csv columns (11, cleaned from 13)

| Column | Coverage | Description |
|---|---|---|
| image_path | 14559/14559 | Relative path to image |
| artist_id | 14559/14559 | Artist identifier |
| timestamp | 5478/14559 | When posted on Instagram |
| likes | 5478/14559 | Instagram likes |
| comments | 5478/14559 | Instagram comments |
| comments_disabled | 5478/14559 | Whether comments were disabled |
| instagram_handle | 5478/14559 | Instagram username |
| file_size | 14559/14559 | File size in bytes |
| width | 14559/14559 | Image width in pixels |
| height | 14559/14559 | Image height in pixels |
| aspect_ratio | 14559/14559 | Width / height |

## Notes

- 5,478 images have Instagram metadata (timestamp, likes, comments).
  The remaining 9,081 were collected from artist websites.
- Nationality data available for 234 of 442 artists.
- Raw originals preserved as *_raw.csv for reproducibility.
