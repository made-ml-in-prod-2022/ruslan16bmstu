from faker_generator import gen_fake_dataset
import click
import os


@click.command(name="generator")
@click.argument("output_dir")
@click.argument("n_rows", default=100)
@click.argument("target", default=True)
def main(output_dir: str, n_rows: int, target: bool) -> None:
    df, targets = gen_fake_dataset(n_rows, target)
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/features.csv", index=False)
    targets.to_csv(f"{output_dir}/targets.csv", index=False)


if __name__ == "__main__":
    main()
