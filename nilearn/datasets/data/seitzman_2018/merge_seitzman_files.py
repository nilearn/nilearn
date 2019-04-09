"""This script takes original files from Seitzman atlas
    - ROIs_300inVol_MNI_allInfo.txt
    - ROIs_anatomicalLabels.txt
and creates one nicely formatted csv file (seitzman_2018_rois.csv) with
coordinates, network and region names
"""
import pandas as pd


def merge_seitzman_files():
    rois = (pd.read_csv("ROIs_300inVol_MNI_allInfo.txt", sep=" ").
            rename(columns={"netName": "network", "radius(mm)": "radius"})
            )

    labels = pd.read_csv("ROIs_anatomicalLabels.txt", sep=" ")

    region_mapping = {}
    for r in labels.columns[0].split(","):
        i, region = r.split("=")
        region_mapping[int(i)] = region

    labels.columns = ["region"]
    labels.replace(to_replace=region_mapping, inplace=True)

    rois = pd.merge(rois, labels, left_index=True, right_index=True)
    rois.to_csv("seitzman_2018_rois.csv", index=False)


if __name__ == "__main__":
    merge_seitzman_files()
