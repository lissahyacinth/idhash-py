import pandas as pd
import pytest

@pytest.fixture(scope="session")
def date_data() -> pd.DataFrame:
    example_data = pd.DataFrame.from_dict({
        'a' : pd.to_datetime(['2021-01-01', '2022-02-01'])
        })
    return example_data


@pytest.fixture(scope="session")
def mixed_type_data() -> pd.DataFrame:
    example_data = pd.DataFrame.from_dict({
        'a' : pd.to_datetime(['2021-01-01', '2022-02-01', '2020-01-01']),
        'b' : [0, 1, 3],
        'c' : ['0', '1', '0'],
        'd' : [True, False, True]
        })
    return example_data
