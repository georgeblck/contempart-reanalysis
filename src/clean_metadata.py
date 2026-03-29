"""
One-time cleanup of contempArt metadata CSVs.

Reads the raw files, fixes issues, writes cleaned versions.
Originals are kept as *_raw.csv for reference.

Usage:
    uv run python src/clean_metadata.py
"""

from pathlib import Path

import pandas as pd


def clean_artists(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Merge two instagram handle columns into one
    df["instagram_handle"] = df["instagramHandle.y"].fillna(df["instagramHandle.x"])

    # Fix bool columns
    for col in ["isBus", "isPriv"]:
        df[col] = df[col].map({True: True, False: False, "True": True, "False": False})

    # Rename columns to snake_case
    rename = {
        "ID": "artist_id",
        "imgCount": "img_count",
        "avgLikes": "avg_likes",
        "avgComm": "avg_comments",
        "avgSize": "avg_file_size",
        "avgW": "avg_width",
        "avgH": "avg_height",
        "avgNocomm": "avg_no_comments",
        "avgAR": "avg_aspect_ratio",
        "university": "school",
        "univEastGerman": "east_german",
        "name": "full_name",
        "iso3c": "country_iso3",
        "class": "professor_class",
        "gender": "gender",
        "instagramPrivate": "instagram_private",
        "instagramPrivateAllowed": "instagram_private_allowed",
        "website1": "website",
        "followCount": "follower_count",
        "followingCount": "following_count",
        "isBus": "is_business",
        "isPriv": "is_private",
        "postsCount": "posts_count",
        "continent": "continent",
        "region": "region",
    }

    # Select and rename (drops redundant columns)
    keep_cols = list(rename.keys())
    keep_cols.append("instagram_handle")  # already renamed above
    df = df.rename(columns=rename)

    final_cols = [
        "artist_id",
        "full_name",
        "school",
        "east_german",
        "professor_class",
        "gender",
        "country_iso3",
        "continent",
        "region",
        "instagram_handle",
        "instagram_private",
        "instagram_private_allowed",
        "is_business",
        "is_private",
        "follower_count",
        "following_count",
        "posts_count",
        "website",
        "img_count",
        "avg_likes",
        "avg_comments",
        "avg_file_size",
        "avg_width",
        "avg_height",
        "avg_aspect_ratio",
    ]

    df = df[[c for c in final_cols if c in df.columns]]

    # Fix east_german to bool
    df["east_german"] = df["east_german"].map({"Yes": True, "No": False})

    # Fix instagram_private columns to bool
    for col in ["instagram_private", "instagram_private_allowed"]:
        if col in df.columns:
            df[col] = df[col].map({"Yes": True, "No": False})

    print(f"artists: {df.shape[0]} rows, {df.shape[1]} cols (was 35 cols)")
    print(f"  Dropped: univNumeric, good, studentStatus, eurocontrolStatfor,")
    print(f"           website2, createTime, nationality, avgNocomm,")
    print(f"           instagramHandle.x/.y (merged), region23")
    return df


def clean_images(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")

    # Fix bool column
    df["noComm"] = df["noComm"].map({True: True, False: False, "True": True, "False": False})

    # Update image paths to new structure
    df["imgPath"] = df["imgPath"].str.replace("visart2020/", "images/visart2020/", n=1)

    rename = {
        "imgPath": "image_path",
        "ID": "artist_id",
        "timeStamp": "timestamp",
        "likes": "likes",
        "comments": "comments",
        "noComm": "comments_disabled",
        "instagramHandle": "instagram_handle",
        "fileSize": "file_size",
        "width": "width",
        "height": "height",
        "aspectRatio": "aspect_ratio",
    }

    df = df.rename(columns=rename)

    # Drop redundant columns
    final_cols = [
        "image_path",
        "artist_id",
        "timestamp",
        "likes",
        "comments",
        "comments_disabled",
        "instagram_handle",
        "file_size",
        "width",
        "height",
        "aspect_ratio",
    ]

    df = df[[c for c in final_cols if c in df.columns]]

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    print(f"images: {df.shape[0]} rows, {df.shape[1]} cols (was 13 cols)")
    print(f"  Dropped: basePath, imgUrl")
    return df


def clean_edgelist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    print(f"edgelist: {df.shape[0]} rows, columns renamed to lowercase")
    return df


def main():
    data_dir = Path("data")
    meta_dir = data_dir / "metadata"
    social_dir = data_dir / "social"

    # Backup originals
    artists_path = meta_dir / "artists.csv"
    images_path = meta_dir / "images.csv"
    edgelist_path = social_dir / "edgelist.csv"

    artists_path.rename(meta_dir / "artists_raw.csv")
    images_path.rename(meta_dir / "images_raw.csv")
    edgelist_path.rename(social_dir / "edgelist_raw.csv")

    print("Originals backed up as *_raw.csv\n")

    # Clean and save
    artists = clean_artists(meta_dir / "artists_raw.csv")
    artists.to_csv(artists_path, index=False)
    print()

    images = clean_images(meta_dir / "images_raw.csv")
    images.to_csv(images_path, index=False)
    print()

    edgelist = clean_edgelist(social_dir / "edgelist_raw.csv")
    edgelist.to_csv(edgelist_path, index=False)
    print()

    # Summary
    print("=" * 50)
    print("Cleaned files saved. Raw originals kept as *_raw.csv.")
    print(f"\nartists.csv:  {artists.shape[1]} cols (was 35)")
    print(f"  school:         {artists['school'].nunique()} schools, 0 nulls")
    print(f"  gender:         {artists['gender'].notna().sum()}/{len(artists)} have data")
    print(f"  country_iso3:   {artists['country_iso3'].notna().sum()}/{len(artists)} have data")
    print(f"  professor_class:{artists['professor_class'].notna().sum()}/{len(artists)} have data")
    print(f"  instagram:      {artists['instagram_handle'].notna().sum()}/{len(artists)} have data")
    print(f"\nimages.csv:   {images.shape[1]} cols (was 13)")
    print(f"  with timestamp: {images['timestamp'].notna().sum()}/{len(images)}")
    print(f"  with likes:     {images['likes'].notna().sum()}/{len(images)}")


if __name__ == "__main__":
    main()
