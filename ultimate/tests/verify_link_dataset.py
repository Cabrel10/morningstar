import pandas as pd
from pathlib import Path


def verify_link_dataset():
    """Verify LINK dataset matches SHIB dataset structure and quality."""
    # Load datasets
    link_path = Path("data/processed/link_usdt_binance_4h.parquet")
    shib_path = Path("data/processed/shib_usdt_binance_4h_processed.parquet")

    try:
        link_df = pd.read_parquet(link_path)
        shib_df = pd.read_parquet(shib_path)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return False

    # 1. Structure comparison
    print("\n=== Structure Comparison ===")
    link_cols = set(link_df.columns)
    shib_cols = set(shib_df.columns)

    # Check for missing columns
    missing_in_link = shib_cols - link_cols
    missing_in_shib = link_cols - shib_cols

    if missing_in_link:
        print(f"WARNING: Missing columns in LINK: {missing_in_link}")
    if missing_in_shib:
        print(f"WARNING: Extra columns in LINK: {missing_in_shib}")

    # Check data types
    type_mismatches = []
    for col in link_cols & shib_cols:
        if link_df[col].dtype != shib_df[col].dtype:
            type_mismatches.append((col, link_df[col].dtype, shib_df[col].dtype))

    if type_mismatches:
        print("\nData type mismatches:")
        for col, link_type, shib_type in type_mismatches:
            print(f"{col}: LINK={link_type}, SHIB={shib_type}")

    # 2. Data validation
    print("\n=== Data Validation ===")
    # Check for NaN values
    link_nans = link_df.isna().sum()
    shib_nans = shib_df.isna().sum()

    print("\nNaN values in LINK:")
    print(link_nans[link_nans > 0])
    print("\nNaN values in SHIB:")
    print(shib_nans[shib_nans > 0])

    # Check timestamp continuity
    print("\nTimestamp checks:")
    link_times = pd.to_datetime(link_df.index)
    shib_times = pd.to_datetime(shib_df.index)

    print(f"LINK time range: {link_times.min()} to {link_times.max()}")
    print(f"SHIB time range: {shib_times.min()} to {shib_times.max()}")

    link_freq = pd.infer_freq(link_times)
    shib_freq = pd.infer_freq(shib_times)
    print(f"LINK frequency: {link_freq}")
    print(f"SHIB frequency: {shib_freq}")

    # 3. Feature completeness
    print("\n=== Feature Completeness ===")
    required_tech_indicators = 38  # As specified in task
    actual_tech_indicators = len([c for c in link_cols if c.startswith("ta_")])
    print(f"Technical indicators in LINK: {actual_tech_indicators}/{required_tech_indicators}")

    # Check for LLM and MCP features
    has_llm = any(c.startswith("llm_") for c in link_cols)
    has_mcp = any(c.startswith("mcp_") for c in link_cols)
    print(f"LLM features present: {has_llm}")
    print(f"MCP features present: {has_mcp}")

    return True


if __name__ == "__main__":
    verify_link_dataset()
