from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

train = pd.read_csv(ROOT / "Train.csv")
audit = pd.read_csv(ROOT / "outputs/audit/train_image_audit.csv")

# Match the audit hash to the original observations.
df = train.merge(
    audit[["SequenceHash"]],
    left_index=True,
    right_index=True,
)

# Find sequences appearing more than once with different labels.
duplicate_hashes = (
    df.groupby("SequenceHash")
    .filter(lambda group: len(group) > 1 and group["RootVolume"].nunique() > 1)[
        "SequenceHash"
    ]
    .unique()
)

columns = [
    "ID",
    "FolderName",
    "PlantNumber",
    "Side",
    "Start",
    "End",
    "RootVolume",
    "Genotype",
    "Stage",
    "SequenceHash",
]

for sequence_hash in duplicate_hashes[:5]:
    group = df[df["SequenceHash"] == sequence_hash]

    print("\n" + "=" * 80)
    print("DUPLICATE IMAGE SEQUENCE")
    print("=" * 80)

    print(group[columns].to_string(index=False))
