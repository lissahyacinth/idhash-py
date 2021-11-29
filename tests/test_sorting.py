import pandas as pd
import pyarrow as pa

from idhash import id_hash

def hash_pd(df: pd.DataFrame, max_chunksize=1000) -> int:
    dtypes = [str(df.dtypes[x]) for x in df.columns]
    df_batches = pa.Table.from_pandas(df).to_batches(max_chunksize=max_chunksize)
    return id_hash(df_batches, df.columns, dtypes)


def test_batch_consistency(mixed_type_data):
    initial_data = mixed_type_data
    sorted_data = mixed_type_data.sort_values(['a', 'b', 'c', 'd'])
    assert not all(
        pd.util.hash_pandas_object(initial_data.reset_index()) == 
        pd.util.hash_pandas_object(sorted_data.reset_index())
    )
    assert hash_pd(initial_data, 1000) == hash_pd(sorted_data, 1)