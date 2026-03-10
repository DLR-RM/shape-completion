import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px


def visualize_csv_files(
    directory_or_paths: Path | list[Path],
    file: str = "test_eval_0.50.csv",
    cls: str = "mean",
    metric: str = "f1",
) -> None:
    if isinstance(directory_or_paths, Path):
        file_paths = list(directory_or_paths.rglob(file))
    else:
        file_paths = directory_or_paths

    values: list[float] = []
    dir_names: list[str] = []
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        if cls in df["category name"].values:
            value = df.loc[df["category name"] == cls, metric].iloc[0]
            values.append(value)
            dir_names.append(file_path.parent.parent.name if "generation" in str(file_path) else file_path.parent.name)

    df = pd.DataFrame({"Experiment": dir_names, metric.upper(): values})
    df_sorted = df.sort_values(metric.upper(), ascending=False)
    fig = px.bar(
        df_sorted,
        x="Experiment",
        y=metric.upper(),
        color="Experiment",
        title=f"{cls.capitalize()} {metric.upper()} by Experiment",
    )
    fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize metrics from CSV files in a directory")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--directory",
        metavar="DIR",
        type=Path,
        help="path to directory to search recursively for CSV files",
    )
    group.add_argument(
        "--files",
        metavar="FILES",
        type=Path,
        nargs="+",
        help="list of file paths to visualize",
    )
    parser.add_argument(
        "--file",
        metavar="FILE",
        type=str,
        default="test_eval_0.50.csv",
        help="file to visualize (default: test_eval_0.50.csv)",
    )
    parser.add_argument(
        "--cls",
        metavar="CLASS",
        type=str,
        default="mean",
        help="class to visualize (default: mean)",
    )
    parser.add_argument(
        "--metric",
        metavar="METRIC",
        type=str,
        default="f1",
        help="metric to visualize (default: f1)",
    )
    args = parser.parse_args()

    directory_or_paths = args.directory if args.files is None else args.files
    visualize_csv_files(directory_or_paths, args.file, args.cls, args.metric)


if __name__ == "__main__":
    main()
