# IDHash

Efficiently create an identifiable hash for a dataset that is independent of row ordering but dependent of column ordering. The core purpose of IDHash is to identify quickly if two datasets are the same without requiring them to both be on the same machine, but also expands to being able to check if one dataset is a modified version of the other dataset if the modification is known upfront.

IDHash is based upon UNFv6, and details around UNF can be found [here](https://guides.dataverse.org/en/latest/developers/unf/index.html), where the major differences are that UNF is column-invariant but row-dependent, and IDHash is column-dependent and row-invariant.


## Examples
```python
import pandas as pd
import pyarrow as pa
from idhash import id_hash

def hash_pd(df: pd.DataFrame) -> int:
    dtypes = [str(df.dtypes[x]) for x in df.columns]
    df_batches = pa.Table.from_pandas(x).to_batches()
    return id_hash(df_batches, df.columns, dtypes)

x=pd.DataFrame.from_dict({'a': [1,2,3]})
print(hash_pd(x))
> 259167810065665855969772359546814925541
```

## Method Drawbacks

### Duplicate Identification
Due to the requirement for row-invariance, a dataset of
```
    A    B
    1    2
    1    2
    1    2
```
will produce the same hash as;
```
    A    B
    1    2
```
But not the same as;
```
    A    B
    1    2
    1    2
```

In practice, this is relatively unlikely, and for the core purpose of datasets within Machine Learning, it is not a primary issue.

## Preprocessing
Each column has specific pre-processing according to the UNF definition. This mostly consists of ensuring that floating point values and timestamps (currently unsupported in IDHash) are representable consistently across datasets when taking into account floating point epsilon. 

## Hash Generation
Each row is taken as a single bytestream, and hashed using Murmurhash128. Murmurhash is a non-cryptographically secure hash function that produces a well distributed hash for each individual value. By XORing the individual hashed primitives, a final hash can be produced for the final dataset that does not take into account duplicates.  

## Checking for Equality + Delta
As the hashed rows are XORed against each other to produce the final value, it is also possible to remove rows against the final hash by producing a row hash in the same manner as was originally performed. 

## Data Processing
IDHash operates over Apache Arrow RecordBatches and can process with zero-copy over the batches.