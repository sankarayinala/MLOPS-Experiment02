import os
import pandas as pd

from config.paths_config import DF, RATING_DF


def convert_file(csv_path: str, parquet_path: str, usecols=None, dtype=None, chunksize: int = 200000):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    print(f"Converting: {csv_path} -> {parquet_path}")

    chunks = []
    for chunk in pd.read_csv(csv_path, usecols=usecols, dtype=dtype, chunksize=chunksize, low_memory=True):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.to_parquet(parquet_path, index=False, engine="pyarrow")
    print(f"Done: {parquet_path}")


def main():
    convert_file(
        csv_path=DF,
        parquet_path=DF.replace(".csv", ".parquet"),
        usecols=["anime_id", "eng_version", "Genres", "Members"],
        dtype={
            "anime_id": "int32",
            "eng_version": "string",
            "Genres": "string",
            "Members": "float32",
        },
    )

    convert_file(
        csv_path=RATING_DF,
        parquet_path=RATING_DF.replace(".csv", ".parquet"),
        usecols=["user_id", "anime_id", "rating"],
        dtype={
            "user_id": "int32",
            "anime_id": "int32",
            "rating": "float32",
        },
    )


if __name__ == "__main__":
    main()
