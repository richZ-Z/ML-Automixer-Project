"""
join_csvs.py — Join two CSV files on a shared column (like SQL JOINs)

Usage:
    python join_csvs.py file1.csv file2.csv --on column_name [--how left|right|inner|outer]

Examples:
    python join_csvs.py customers.csv orders.csv --on customer_id
    python join_csvs.py customers.csv orders.csv --on customer_id --how inner
    python join_csvs.py customers.csv orders.csv --on customer_id --how left --output merged.csv
"""

import argparse
import pandas as pd
import sys


def join_csvs(
    file1: str,
    file2: str,
    on: str,
    how: str = "outer",
    output: str = "joined_output.csv",
    left_on: str = None,
    right_on: str = None,
) -> pd.DataFrame:
    """
    Join two CSV files.

    Args:
        file1:     Path to the left CSV file
        file2:     Path to the right CSV file
        on:        Column name to join on (must exist in both files)
        how:       Join type — 'inner', 'left', 'right', or 'outer' (default: outer)
        output:    Output file path (default: joined_output.csv)
        left_on:   Column name in file1 if columns have different names
        right_on:  Column name in file2 if columns have different names

    Returns:
        Merged DataFrame
    """
    print(f"Reading '{file1}'...")
    df1 = pd.read_csv(file1)

    print(f"Reading '{file2}'...")
    df2 = pd.read_csv(file2)

    print(f"\nFile 1 — {len(df1)} rows, columns: {list(df1.columns)}")
    print(f"File 2 — {len(df2)} rows, columns: {list(df2.columns)}")

    # Determine join keys
    join_kwargs = {"how": how}
    if left_on and right_on:
        join_kwargs["left_on"] = left_on
        join_kwargs["right_on"] = right_on
        print(f"\nJoining on '{left_on}' (left) ↔ '{right_on}' (right), type: {how}")
    elif on:
        join_kwargs["on"] = on
        # Validate the column exists in both files
        if on not in df1.columns:
            raise ValueError(
                f"Column '{on}' not found in {file1}. Available: {list(df1.columns)}"
            )
        if on not in df2.columns:
            raise ValueError(
                f"Column '{on}' not found in {file2}. Available: {list(df2.columns)}"
            )
        print(f"\nJoining on '{on}', type: {how}")
    else:
        raise ValueError("Provide either --on, or both --left-on and --right-on")

    merged = pd.merge(df1, df2, **join_kwargs)

    print(f"Result  — {len(merged)} rows, columns: {list(merged.columns)}")

    merged.to_csv(output, index=False)
    print(f"\nSaved to '{output}'")

    return merged


def main():
    parser = argparse.ArgumentParser(description="Join two CSV files.")

    parser.add_argument("file1", help="Path to the left (first) CSV file")
    parser.add_argument("file2", help="Path to the right (second) CSV file")

    key_group = parser.add_mutually_exclusive_group(required=True)
    key_group.add_argument(
        "--on", help="Column name to join on (same name in both files)"
    )
    key_group.add_argument(
        "--left-on", help="Column name in file1 (use with --right-on)"
    )

    parser.add_argument("--right-on", help="Column name in file2 (use with --left-on)")
    parser.add_argument(
        "--how",
        choices=["inner", "left", "right", "outer"],
        default="outer",
        help="Join type (default: outer)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="joined_output.csv",
        help="Output file path (default: joined_output.csv)",
    )

    args = parser.parse_args()

    if args.left_on and not args.right_on:
        parser.error("--left-on requires --right-on")

    try:
        join_csvs(
            file1=args.file1,
            file2=args.file2,
            on=args.on,
            how=args.how,
            output=args.output,
            left_on=args.left_on,
            right_on=args.right_on,
        )
    except (ValueError, FileNotFoundError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Quick demo
    # def demo():
    """Run a quick demo with sample data."""
    import io

    customers_csv = """customer_id,name,email
1,Alice,alice@example.com
2,Bob,bob@example.com
3,Carol,carol@example.com
4,Dave,dave@example.com"""

    orders_csv = """customer_id,order_id,amount
1,1001,49.99
1,1002,19.99
2,1003,89.50
5,1004,15.00"""

    df_customers = pd.read_csv(io.StringIO(customers_csv))
    df_orders = pd.read_csv(io.StringIO(orders_csv))

    for join_type in ["inner", "left", "right", "outer"]:
        result = pd.merge(df_customers, df_orders, on="customer_id", how=join_type)
        print(
            f"\n── {join_type.upper()} JOIN ({'SQL: ' + join_type.upper() + ' JOIN'}) ──"
        )
        print(result.to_string(index=False))


if __name__ == "__main__":
    import sys

    main()

    # if len(sys.argv) == 1:
    # print("No arguments given — running demo with sample data.\n")
    # demo()
    # else:
    # main()

    # run with
    # Same column name in both files (outer join by default)
    # python join_csvs.py customers.csv orders.csv --on customer_id

    # Specify join type
    # python join_csvs.py customers.csv orders.csv --on customer_id --how left

    # Different column names in each file
    # python join_csvs.py file1.csv file2.csv --left-on id --right-on user_id --how inner

    # Custom output path
    # python join_csvs.py a.csv b.csv --on id --output merged.csv
