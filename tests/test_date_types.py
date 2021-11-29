import pandas as pd
import pyarrow as pa

from idhash import id_hash, IDHasher
from typing import List

def hash_pd(df: pd.DataFrame) -> int:
    dtypes = [str(df.dtypes[x]) for x in df.columns]
    df_batches = pa.Table.from_pandas(df).to_batches()
    return id_hash(df_batches, df.columns, dtypes)

def create_hasher(columns: List[str], dtypes: pd.Series) -> IDHasher:
    dtypes = [str(dtypes[x]) for x in columns]
    return IDHasher(field_names=columns, field_types=dtypes)

def test_create_hasher(date_data):
    print(f"Start Hasher")
    hasher = create_hasher(date_data.columns, date_data.dtypes)
    print(f"Created Hasher")
    hasher.write_batches(pa.Table.from_pandas(date_data).to_batches(), delta="Add")
    print(f"Finished Hasher {hasher}")
    print(f"Hasher Res {hasher.finalize()}")
    assert type(hasher.finalize()) is int

def test_hash_dataframe(date_data):
    assert isinstance(hash_pd(date_data.copy(deep=True)), int)
