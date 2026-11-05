"""Build hub operating profiles, K-Means segments and tier mismatch evidence."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUTPUTS = ROOT / "outputs"
PROFILE_COLUMNS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]


def main():
    OUTPUTS.mkdir(exist_ok=True)
    shipments = pd.read_csv(DATA / "modelling_table.csv")
    hubs = pd.read_csv(DATA / "hubs.csv")
    profile = shipments.groupby("hub_id", sort=True).agg(
        shipments=("hub_id", "size"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index().sort_values("hub_id", kind="stable")
    profile.round(4).to_csv(OUTPUTS / "hub_profile.csv", index=False)

    scaled = StandardScaler().fit_transform(profile[PROFILE_COLUMNS])
    labels_by_k = {}
    separation_rows = []
    for k in [2, 3, 4, 5]:
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(scaled)
        labels_by_k[k] = labels
        across, within = [], []
        for left, right in combinations(range(len(profile)), 2):
            distance = np.linalg.norm(scaled[left] - scaled[right])
            if labels[left] == labels[right]:
                within.append(distance)
            else:
                across.append(distance)
        sizes = sorted(np.bincount(labels).tolist(), reverse=True)
        separation_rows.append({
            "K": k, "across_over_within": np.mean(across) / np.mean(within),
            "cluster_sizes": "-".join(str(size) for size in sizes),
        })
    pd.DataFrame(separation_rows).round({"across_over_within": 4}).to_csv(
        OUTPUTS / "part_d_separation.csv", index=False
    )

    profile["cluster"] = labels_by_k[3]
    cluster_order = profile.groupby("cluster")["delay_rate"].mean().sort_values(kind="stable").index.tolist()
    names = {cluster: name for cluster, name in zip(cluster_order, ["Standard", "Watch", "Escalate"])}
    profile["segment"] = profile["cluster"].map(names)
    hub_segments = profile[["hub_id", "segment"]].merge(
        hubs[["hub_id", "hub_city", "hub_tier"]], on="hub_id", how="left", validate="one_to_one"
    ).sort_values("hub_id", kind="stable")
    hub_segments.to_csv(OUTPUTS / "hub_segments.csv", index=False)

    segment_profile = profile.groupby("segment", sort=False).agg(
        hubs=("hub_id", "size"),
        mean_delivery_hours=("mean_delivery_hours", "mean"), delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"), mean_n_stops=("mean_n_stops", "mean"),
    ).reindex(["Standard", "Watch", "Escalate"]).reset_index()
    segment_profile.round(4).to_csv(OUTPUTS / "segment_profile.csv", index=False)

    expected_segment = {1: "Standard", 2: "Watch", 3: "Escalate"}
    mismatch = hub_segments[hub_segments.apply(
        lambda row: row["segment"] != expected_segment[row["hub_tier"]], axis=1
    )]
    mismatch.to_csv(OUTPUTS / "tier_vs_segment_mismatch.csv", index=False)


if __name__ == "__main__":
    main()