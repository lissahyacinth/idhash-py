# IDHash

Efficiently create an identifiable hash for a dataset that is;
* Independent of row ordering
* Dependent of column ordering

Quickly identify if two datasets are identical by comparing their hashes, without needing to presort the values.

IDHash is based upon UNFv6, and details around UNF can be found [here](https://guides.dataverse.org/en/latest/developers/unf/index.html), where the major differences are that UNF is column-invariant but row-dependent, and IDHash is column-dependent and row-invariant.


## Example - Hash Entire Dataset
```python
import pandas as pd
import pyarrow as pa
from idhash import id_hash

def hash_pd(df: pd.DataFrame) -> int:
    dtypes = [str(df.dtypes[col]) for col in df.columns]
    df_batches = pa.Table.from_pandas(df).to_batches()
    return id_hash(df_batches, df.columns, dtypes)

x=pd.DataFrame.from_dict({'a': [1,2,3]})
print(hash_pd(x))
> 259167810065665855969772359546814925541
```

## Example - Hash Iteratively
```python
import pandas as pd
import pyarrow as pa
from idhash import id_hash, IDHasher
from typing import List

def create_hasher(columns: List[str], dtypes: pd.Series) -> IDHasher:
    dtypes = [str(dtypes[col]) for col in columns]
    return IDHasher(field_names=columns, field_types=dtypes)

df = pd.DataFrame.from_dict({'a': [1,2,3]})
hasher = create_hasher(df.columns, df.dtypes)
batches = pa.Table.from_pandas(df).to_batches(max_chunksize=1)
for batch in batches:
    hasher.write_batches([batch], delta="Add")
print(hasher.finalize())
> 259167810065665855969772359546814925541
```

Iterative hashing has an additional benefit - it's possible to verify a delta between two datasets, i.e.

```python
dataset_a: pd.DataFrame, dataset_b: pd.DataFrame, delta: List[pa.RecordBatch] = load_data()
hasher_a = create_hasher(dataset_a.columns, dataset_a.dtypes)
hasher_a.write_batches(pa.Table.from_pandas(dataset_a), delta="Add")
hasher_b = create_hasher(dataset_b.columns, dataset_b.dtypes)
hasher_b.write_batches(pa.Table.from_pandas(dataset_b), delta="Add")

assert hasher_a.write_batches(delta, delta="Add").finalize() == hasher_b.finalize()
```

## Preprocessing
Each column has specific pre-processing according to the UNF definition. This mostly consists of ensuring that floating point values, datetimes, and timestamps are representable consistently across datasets when taking into account floating point epsilon. 

## Hash Generation
Each row is taken as a single bytestream, and hashed using Murmurhash128. Murmurhash is a non-cryptographically secure hash function that produces a well distributed hash for each individual value. By adding (wrapping around f64::max) the individual hashed primitives, a final hash can be produced for the final dataset that does not take into account duplicates.  

## Checking for Equality + Delta
As the hashed rows are added to each other to produce the final value, it is also possible to remove rows against the final hash by producing a row hash in the same manner as was originally performed. 

## Data Processing
IDHash operates over Apache Arrow RecordBatches and can process with zero-copy over the batches.
