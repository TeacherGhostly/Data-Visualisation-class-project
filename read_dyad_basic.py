#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statannotations.Annotator import Annotator
import glob
import os

data_directory = "./vis_project_data/"

# Define the CAs with significant p-value
significant_cas = {
    "Trained": ["albert", "All CAs"],
    "Untrained": ["chris", "ellie", "All CAs"],
}

# Default weights; used if not set
default_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

# Types of sessions/files:
session_types = ["cog", "so"]
whatdoors = ["indoor", "outdoor"]
whichs = ["base", "inter"]

# Combine to single itteratable list
combined_scenarios = [
    (ses_type, whatdoor, which)
    for ses_type in session_types
    for whatdoor in whatdoors
    for which in whichs
]

################################################################################


def combined_score(filename, weights):
    """Calculates the 'score' for a single session/file.
    Assumes total session duration is 360s, otherwise returns 'nan'.
    This could be modified simply to also return other details of the session."""
    with open(filename, "r") as file:
        score = 0.0
        total_duration = 0.0
        t_end_prev = 0.0
        for count, line in enumerate(file.readlines()):
            # print(count, line)
            data = line.split(",", 4)
            if count == 0:
                continue
            if line[0] == "*":
                break

            t_catagory = int(data[0])
            t_beg = int(data[1])
            t_end = int(data[2])

            if t_beg != t_end_prev:
                print("Error, missing time stamp?")
            t_end_prev = t_end

            assert t_end >= t_beg
            if count == 1:
                assert t_beg == 0

            duration = float(t_end - t_beg)
            total_duration += duration
            score += weights[t_catagory - 1] * duration
        return score / total_duration
        return score if np.abs(total_duration - 1.0) < 1.0e-5 else np.nan


################################################################################


def print_scores(
    ca, peer, cognitive_weights=default_weights, social_weights=default_weights
):
    """Calculates the scores for given ca/peer pair.
    It returns a DataFrame containing the scores.
    """
    trained = "Trained" if "u" <= peer[0] <= "z" else "Untrained"
    result = []  # Initialize an empty list to store the results

    for ses_type, whatdoor, which in combined_scenarios:
        weights = cognitive_weights if ses_type == "cog" else social_weights

        # glob creates the list of filenames that match the given pattern
        # '*' is a wildcard
        files = glob.glob(
            data_directory + f"{ses_type}-*-{which}-*-{ca}-{peer}-{whatdoor}.dtx"
        )

        if len(files) == 0:
            continue

        scores = []
        for file in files:
            tmp_score = combined_score(file, weights)
            if not np.isnan(tmp_score):
                scores.append(tmp_score)
        scores = np.array(scores)

        mean = scores.mean()
        sdev = scores.std(ddof=1)  # "corrected" sdev
        sem = sdev / np.sqrt(len(scores))

        # Store the results in the list
        result.append(
            [
                ca,
                peer,
                trained,
                ses_type,
                whatdoor,
                which,
                mean,
                sem,
                len(scores),
                len(files),
                scores,
            ]
        )

    # Convert the list to a DataFrame
    result_df = pd.DataFrame(
        result,
        columns=[
            "ca",
            "peer",
            "trained",
            "ses_type",
            "whatdoor",
            "which",
            "mean",
            "sem",
            "num_scores",
            "num_files",
            "scores",
        ],
    )

    return result_df


################################################################################
def unique_pairs():
    """Returns list of unique ca/peer pairs"""
    all_files = glob.glob(data_directory + "/*.dtx")
    list = []
    for file in all_files:
        t = file.split("-")
        list.append([t[4], t[5]])

    return np.unique(list, axis=0)


################################################################################

if __name__ == "__main__":

    # Example usage:

    ca_peer_list = unique_pairs()
    print(ca_peer_list)

    # From the thesis:
    soc_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

    # Match table 5.1 of thesis
    # cog_weights = np.array([-1, 0, 1, 2, 2, 3, 5])

    # Match matlab example:
    cog_weights = np.array([0, -1, 1, 2, 2, 3, 5])

    # Make DF 1
    results = []
    for ca, peer in ca_peer_list:
        result = print_scores(ca, peer, cog_weights, soc_weights)
        results.append(result)

    # Concatenate all the DataFrames in the list
    results_df = pd.concat(results, ignore_index=True)

    # Create a copy of the DataFrame for 'All CAs'
    all_cas_df = results_df.copy()
    all_cas_df["ca"] = "All CAs"

    # Append the 'All CAs' data to the original DataFrame
    results_df = pd.concat([results_df, all_cas_df])

    # Combine 'indoor' and 'outdoor' data
    results_df["whatdoor"] = "indoor/outdoor"

    # Visualisation 1 (colour + significance)
    for training in ["Untrained", "Trained"]:
        training_df = results_df[results_df["trained"] == training]

        # Change some of seaborn's style settings with `sns.set()`
        sns.set(
            style="ticks",  # The 'ticks' style
            rc={
                "figure.figsize": (10, 6),  # width = 6, height = 9
                "figure.facecolor": "ivory",  # Figure colour
                "axes.facecolor": "ivory",
            },
            palette="colorblind",
        )  # Axes colour

        ax = sns.boxplot(
            x="ca",
            y="mean",
            hue="which",
            data=results_df,
            showfliers=False,
            boxprops={"alpha": 0.4},
            whiskerprops={"linestyle": ""},
            capprops={"linestyle": ""},
        )

        sns.despine(offset=10, trim=True)

        sns.stripplot(
            data=results_df,
            x="ca",
            y="mean",
            alpha=0.6,
            hue="which",
            hue_order=["base", "inter"],
            dodge=True,
            ax=ax,
        )

        # annotations
        pairs = [
            [("albert", "base"), ("albert", "inter")],
            [("barry", "base"), ("barry", "inter")],
            [("chris", "base"), ("chris", "inter")],
            [("dana", "base"), ("dana", "inter")],
            [("ellie", "base"), ("ellie", "inter")],
            [("All CAs", "base"), ("All CAs", "inter")],
        ]

        annotator = Annotator(
            ax,
            pairs,
            x="ca",
            y="mean",
            color="skyblue",
            hue="which",
            data=results_df,
        )

        pvalues = [0.007, 0.453, 0.133, 0.545, 0.6, 0.004]

        annotator.set_pvalues(pvalues)

        annotator.annotate()

        # Add a title to the plot
        plt.title(f"Trained dyads")
        plt.xlabel("CA")
        plt.ylabel("social interaction")

        # Show the plot
        plt.show()

    # Visualisation 2 (individual sessions)

    # Explode df
    df_exploded = results_df.explode("scores")
    df_cog = df_exploded[df_exploded["ses_type"] == "cog"]

    # Separate the data into 'Untrained' and 'Trained'
    for training in ["Untrained", "Trained"]:
        training_df = df_cog[df_cog["trained"] == training]

        # Change some of seaborn's style settings with `sns.set()`
        sns.set(
            style="ticks",  # The 'ticks' style
            rc={
                "figure.figsize": (10, 6),  # width = 6, height = 9
                "figure.facecolor": "ivory",  # Figure colour
                "axes.facecolor": "ivory",
            },
            palette="colorblind",
        )  # Axes colour

        ax = sns.boxplot(
            x="ca",
            y="scores",  # Change 'mean' to 'scores' to use the individual points
            hue="which",
            data=training_df,
            showfliers=False,
            boxprops={"alpha": 0.4},
            whiskerprops={"linestyle": ""},
            capprops={"linestyle": ""},
        )

        sns.despine(offset=10, trim=True)

        sns.stripplot(
            data=training_df,
            x="ca",
            y="scores",  # Change 'mean' to 'scores' to use the individual points
            alpha=0.6,
            hue="which",
            hue_order=["base", "inter"],
            dodge=True,
            ax=ax,
        )

        # Add a title to the plot
        plt.title(f"Trained dyads (Cognitive)")
        plt.xlabel("CA")
        plt.ylabel("Cognitive Interaction")

        # Adjust the y-axis limits
        # plt.ylim(-1, 2.5)

        # Show the plot
        plt.show()

    # Changing weights

    # New sets of weights

    soc_weights_2 = np.array([0, 1, 2, 3, 4, 5, 6])

    cog_weights_2 = np.array([0, 1, 2, 3, 4, 5, 6])

    # Make DF 2
    results_2 = []
    for ca, peer in ca_peer_list:
        result = print_scores(ca, peer, cog_weights_2, soc_weights_2)
        results_2.append(result)

    # Concatenate all the DataFrames in the list
    results_df_2 = pd.concat(results_2, ignore_index=True)

    # Create a copy of the DataFrame for 'All CAs'
    all_cas_df_2 = results_df_2.copy()
    all_cas_df_2["ca"] = "All CAs"

    # Append the 'All CAs' data to the original DataFrame
    results_df_2 = pd.concat([results_df_2, all_cas_df_2])

    # Change to social only
    soc_df = results_df[results_df["ses_type"] == "so"]
    soc_df_2 = results_df_2[results_df_2["ses_type"] == "so"]

    # First, filter the data for 'All CAs'
    df1 = soc_df[soc_df["ca"] == "All CAs"]
    df2 = soc_df_2[soc_df_2["ca"] == "All CAs"]

    # Concatenate the two dataframes with an additional column 'source'
    df1["source"] = "Original"
    df2["source"] = "New Weights"

    df = pd.concat([df1, df2])

    # Overlay the means
    sns.pointplot(
        x="source",
        y="mean",
        hue="which",
        data=df,
        dodge=0.4,
        join=True,
        palette="colorblind",
        capsize=0.1,
        alpha=0.7,
        linestyles=["--", "--"],
    )

    sns.despine(offset=10, trim=True)

    plt.title("Comparison of Original and New Weights for ALL CAs")
    plt.xlabel("Source")
    plt.ylabel("Mean Social Interaction")
    plt.show()
