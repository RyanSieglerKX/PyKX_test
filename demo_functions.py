import os

os.environ["PYKX_ENABLE_PANDAS_API"] = "true"
import pykx as kx
import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import gc


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def count_records(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["time"] = df["time"].dt.time
    print(f"Num Records: {len(df):,d}")
    return df.head()


def fix_pandas_date_and_time(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S.%f")
    return df


def plot_spreads_volatility_and_trade_volume(ax, sym: str, plot_num: int, spread_tab: kx.Table, volatility_dict: dict, trad_vol_tab: kx.Table) -> None:

    xtick_labels = ["9am", "10am", "11am", "12pm", "1pm", "2pm", "3pm", "4pm", "5pm"]
    low = datetime(2022, 3, 31, 9, 0, 0)
    high = datetime(2022, 3, 31, 18, 0, 0)
    xtick_vals = np.arange(start=low, stop=high, step=timedelta(minutes=60))

    # assign colors
    spread_col = "blue"
    volatility_col = "darkorange"
    volume_col = "darkgreen"

    # plot spreads
    ax[plot_num, 0].set_title(sym)
    ax[plot_num, 0].plot(spread_tab["datetime"], spread_tab["spread"], color=spread_col)
    ax[plot_num, 0].set_ylabel("Spread (Ask - Bid)", color=spread_col)
    ax[plot_num, 0].set_yticks([])
    ax[plot_num, 0].set_xticks(xtick_vals)
    ax[plot_num, 0].set_xticklabels(xtick_labels)

    # plot volatility
    ax2 = ax[plot_num, 0].twinx()
    ax2.plot(volatility_dict["datetime"], volatility_dict["volatility"], color=volatility_col)
    ax2.set_ylabel("Volatility", color=volatility_col)
    ax2.set_yticks([])

    # plot liquidity
    ax[plot_num, 1].set_title(sym)
    ax[plot_num, 1].plot(trad_vol_tab["time"], trad_vol_tab["volume"]/1000000, color=volume_col)
    ax[plot_num, 1].set_ylabel("Trading Volume", color=volume_col)
    ax[plot_num, 1].set_yticks([])
    ax[plot_num, 1].set_xticks(xtick_vals)
    ax[plot_num, 1].set_xticklabels(xtick_labels)


def plot_slippage(df) -> None:
    bin_size = 10_000
    bins = [bin_size*i for i in range(10)]
    volume_bins = pd.cut(df['size']*df['price'], bins)
    plot_df = df.groupby(['venue', volume_bins])['slippage'].mean().unstack().T

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_df.plot(ax=ax)
    fig.suptitle("Slippage Per Venue For Each Trading Volume Amount", fontsize=16)
    plt.xlabel('Trading Volume ($)', fontsize=12)
    plt.ylabel('Splippage (BPS)', fontsize=12)


def collect_garbage() -> None:
    kx.q(".Q.gc[]"),gc.collect()


def fix_res_string(v) -> str:
    return v.stdout.split("\n")[0].replace("+-", "±")


def create_res_dict(pandas_time_res, pykx_time_res, pandas_mem_res, pykx_mem_res) -> pd.DataFrame:
    return pd.DataFrame({
        "Time":
            {
                "Pandas": fix_res_string(pandas_time_res),
                "PyKX": fix_res_string(pykx_time_res),
            },
        "Memory":
            {
                "Pandas": fix_res_string(pandas_mem_res),
                "PyKX": fix_res_string(pykx_mem_res),
            },
    })


def parse_vals(pandas_time_res, pykx_time_res, pandas_mem_res, pykx_mem_res) -> tuple[pd.DataFrame]:
    res_df = create_res_dict(pandas_time_res, pykx_time_res, pandas_mem_res, pykx_mem_res)
    return parse_time_vals(dict(res_df["Time"])), parse_memory_vals(dict(res_df["Memory"]))


def parse_memory_vals(d: dict) -> pd.DataFrame:
    all_rows = pd.DataFrame()
    for k, v in d.items():

        if "," not in v:
            one_row = pd.DataFrame({"total_memory": "", "memory_increment": ""}, index=[k])

        else:
            peak, increment = v.split(", ")
            peak_mem_val = " ".join(peak.split(" ")[-2:])
            increment_val = " ".join(increment.split(" ")[-2:])
            one_row = pd.DataFrame({"total_memory": peak_mem_val, "memory_increment": increment_val}, index=[k])

        all_rows = pd.concat([all_rows, one_row])

    return all_rows.reset_index().rename(columns={"index": "syntax"})


def parse_time_vals(d: dict) -> pd.DataFrame:
    all_rows = pd.DataFrame()
    for k, v in d.items():

        if "±" not in v:
            one_row = pd.DataFrame({"avg_time": "", "avg_dev": "", "runs": "", "loops": ""}, index=[k])

        else:
            avg_time, rest1, rest2 = v.split(" ± ")
            avg_dev, _ = rest1.split(" per loop")
            rest2 = rest2.strip(" std. dev. of ")
            rest2 = rest2.strip(" loops each)")
            runs, loops = rest2.split(" runs, ")
            one_row = pd.DataFrame({"avg_time": avg_time, "avg_dev": avg_dev, "runs": runs, "loops": loops}, index=[k])

        all_rows = pd.concat([all_rows, one_row])

    return all_rows.reset_index().rename(columns={"index": "syntax"})


def fix_time(l: list[str]) -> list[int]:
    fixed_l = []
    for v in l:
        if v.endswith(" ns"):
            fixed_l.append(float(v.strip(" ns")) / 1000 / 1000)
        elif v.endswith(" µs"):
            fixed_l.append(float(v.strip(" µs")) / 1000)
        elif v.endswith(" us"):
            fixed_l.append(float(v.strip(" us")) / 1000)
        elif v.endswith(" ms"):
            fixed_l.append(float(v.strip(" ms")))
        elif v.endswith(" s"):
            fixed_l.append(float(v.strip(" s")) * 1000)
        elif "min " in v:
            mins, secs = v.split("min ")
            total_secs = (float(mins) * 60) + float(secs.strip(" s"))
            fixed_l.append(total_secs * 1000)
        else:
            fixed_l.append(v)
            
    return fixed_l


def compare(df: pd.DataFrame, metric: str, syntax1: str, syntax2: str) -> float:
    factor = float(df[df["syntax"] == syntax1][metric].values) / float(df[df["syntax"] == syntax2][metric].values)
    str_factor = f"{factor:,.2f} times less" if factor > 1 else f"{1/factor:,.2f} time more"
    print(color.BOLD + f"\nThe '{metric}' for '{syntax2}' is {str_factor} than '{syntax1}'.\n" + color.END)
    return factor


def graph_time_data(df_to_graph: pd.DataFrame) -> pd.DataFrame:
    print(tabulate(df_to_graph, headers='keys', tablefmt='psql'))

    fig, ax = plt.subplots(figsize=(7, 3))

    df = df_to_graph.copy()
    df["avg_time"] = fix_time(df["avg_time"])
    df["avg_dev"] = fix_time(df["avg_dev"])
    df["upper_dev"] = df["avg_time"] + df["avg_dev"]
    df["lower_dev"] = df["avg_time"] - df["avg_dev"]

    _ = compare(df, "avg_time", "Pandas", "PyKX")

    df.plot(ax=ax, kind="bar", x="syntax", y="avg_time", rot=0)
    #df.plot(ax=ax, kind="scatter", x="syntax", y="upper_dev", color="yellow")
    #df.plot(ax=ax, kind="scatter", x="syntax", y="lower_dev", color="orange")

    ax.set_title("Pandas VS Pykx Time Taken")
    ax.set_ylabel("Average Time (ms)")
    ax.get_legend().remove()
    ax.set(xlabel=None)

    plt.show()

    return df


def fix_memory(l: list[str]) -> list[float]:
    fixed_l = []
    for v in l:
        if v.endswith(" MiB"):
            fixed_l.append(float(v.strip(" MiB")))
        else:
            fixed_l.append(v)

    return fixed_l


def graph_memory_data(df_to_graph: pd.DataFrame) -> pd.DataFrame:
    print(tabulate(df_to_graph, headers='keys', tablefmt='psql'))

    fig, ax = plt.subplots(figsize=(7, 3))

    df = df_to_graph.copy()
    df["total_memory"] = fix_memory(df["total_memory"])
    df["memory_increment"] = fix_memory(df["memory_increment"])

    _ = compare(df, "memory_increment", "Pandas", "PyKX")

    df.plot(ax=ax, kind="bar", x="syntax", y="memory_increment", rot=0)

    ax.set_title("Pandas Vs Pykx Memory Footprint")
    ax.set_ylabel("Memory Usage (MiB)")
    ax.get_legend().remove()
    ax.set(xlabel=None)

    plt.show()

    return df