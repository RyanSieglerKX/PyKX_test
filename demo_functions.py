import os
import pandas as pd



def fix_pandas_date_and_time(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f")
    return df
