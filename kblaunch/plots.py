import pandas as pd
import plotext as plt
import typer

from kblaunch.cli import app

FILE_PATH = "/nfs/user/s2234411-infk8s/cluster_gpu_usage.json"


def get_data() -> pd.DataFrame:
    df = pd.read_json(FILE_PATH)

    def add_gpu_id(gpu_usage):
        new_list = []
        for i, g in enumerate(gpu_usage):
            g["gpu_id"] = i
            new_list.append(g)
        return new_list

    df["gpu_usage"] = df["gpu_usage"].apply(add_gpu_id)
    df = df.explode("gpu_usage")
    df = pd.concat([df, df.gpu_usage.apply(pd.Series)], axis=1).drop(
        "gpu_usage", axis=1
    )
    df["gpu_mem_used"] = df["memory_used"] / df["memory_total"] * 100
    df["inactive"] = df["gpu_mem_used"] < 1
    return df


def plot_gpu_usage_per_user(df: pd.DataFrame):
    plt.clear_figure()
    latest = df[df["timestamp"] == df["timestamp"].max()]
    usage = latest.groupby("username")["gpu_name"].count()

    plt.bar(usage.index.tolist(), usage.values.tolist())
    plt.title("Current GPU Usage per User")
    plt.xlabel("Username")
    plt.ylabel("Number of GPUs")
    plt.show()


def plot_memory_usage_per_user(df: pd.DataFrame):
    plt.clear_figure()
    latest = df[df["timestamp"] == df["timestamp"].max()]
    memory = latest.groupby("username")["gpu_mem_used"].mean()

    plt.bar(memory.index.tolist(), memory.values.tolist())
    plt.title("Current GPU Memory Usage per User (%)")
    plt.xlabel("Username")
    plt.ylabel("Memory Usage (%)")
    plt.show()


def plot_usage_over_time(df: pd.DataFrame):
    plt.clear_figure()
    usage_over_time = df.groupby("timestamp")["gpu_name"].count()

    plt.plot(
        [t.strftime("%Y-%m-%d %H:%M") for t in usage_over_time.index],
        usage_over_time.values.tolist(),
    )
    plt.title("GPU Usage Over Time")
    plt.xlabel("Time")
    plt.ylabel("Number of GPUs")
    plt.show()


def print_current_stats(df: pd.DataFrame):
    latest = df[df["timestamp"] == df["timestamp"].max()]

    print("\nCurrent GPU Statistics:")
    print("-" * 50)

    # GPU counts
    print("\nGPU Count by Type:")
    for gpu_type, count in latest["gpu_name"].value_counts().items():
        print(f"{gpu_type}: {count}")

    # User statistics
    print("\nUser Statistics:")
    user_stats = (
        latest.groupby("username")
        .agg({"gpu_name": "count", "gpu_mem_used": "mean"})
        .round(2)
    )

    for user, row in user_stats.iterrows():
        print(f"\n{user}:")
        print(f"  GPUs in use: {row['gpu_name']}")
        print(f"  Avg memory usage: {row['gpu_mem_used']}%")


@app.command()
def monitor(
    view: str = typer.Option(
        "current", help="View to display (current, usage, memory, time)"
    ),
):
    """
    Display GPU usage statistics and plots in the terminal.
    Views:
    - current: Show current statistics
    - usage: Show GPU usage per user
    - memory: Show memory usage per user
    - time: Show usage over time
    """
    try:
        df = get_data()

        if view == "current":
            print_current_stats(df)
        elif view == "usage":
            plot_gpu_usage_per_user(df)
        elif view == "memory":
            plot_memory_usage_per_user(df)
        elif view == "time":
            plot_usage_over_time(df)
        else:
            print(f"Invalid view option: {view}")
            print("Available views: current, usage, memory, time")
    except Exception as e:
        print(f"Error displaying monitor: {e}")


if __name__ == "__main__":
    monitor()
