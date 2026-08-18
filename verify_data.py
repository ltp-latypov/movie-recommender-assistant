import polars as pl 

df = pl.read_parquet("/workspaces/movie-recommender-assistant/data/movies_with_embeddings.parquet")

print(df.filter(pl.col("title") == "Prisoners").select(pl.col("title", "overview", "keywords")).row(0))