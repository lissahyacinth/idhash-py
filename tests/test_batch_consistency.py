import pandas as pd
import pyarrow as pa

from idhash import id_hash

def hash_pd(df: pd.DataFrame, max_chunksize=1000) -> int:
    dtypes = [str(df.dtypes[x]) for x in df.columns]
    df_batches = pa.Table.from_pandas(df).to_batches(max_chunksize=max_chunksize)
    return id_hash(df_batches, df.columns, dtypes)


def test_batch_consistency(mixed_type_data):
    assert hash_pd(mixed_type_data, 1000) == hash_pd(mixed_type_data, 1)