"""Optional: save the built-in NOISY demo dataset to data/raw/ (so you can open it in Excel / VS Code).

    python generate_demo_data.py
"""
from pathlib import Path

from src.data_loader import generate_noisy_dataset

dest = Path(__file__).resolve().parent / "data" / "raw" / "demo_noisy_social_media_accounts.csv"
dest.parent.mkdir(parents=True, exist_ok=True)
df = generate_noisy_dataset(5000, 42)
df.to_csv(dest, index=False)
print(f"Saved {len(df):,} rows x {df.shape[1]} columns to {dest}")
