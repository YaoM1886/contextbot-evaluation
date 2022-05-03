import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns

from preprocess import preprocess_tables_surveys

pd.set_option("display.max_rows", None)


def extract_conditional_df(total_df, condition_dict, condition_boxes):

    # extract conditional variables and create list for data
    for key in condition_dict.keys():
        for condition in condition_boxes:
            # for within task type and interacted types
            condition_l = list(total_df.loc[(total_df["stage"]=="main_history"+"_"+key) & (total_df["interacted"] == condition)].drop_duplicates(subset=["id"])["spent_time"].values)
            condition_dict[key].append(condition_l)

            # for task types
            # condition_l = list(total_df.loc[total_df["stage"]=="main_"+key+"_"+condition].drop_duplicates(subset=["id"])["spent_time"].values)

    return condition_dict


def boxplot_stage_time(condition_dict, labels, colors):

    # boxplot of the nine stages and their spent time of the task
    x_positions_fmt = ["history-early", "history-half", "history-late"]

    bplot = plt.boxplot(condition_dict["early"], labels=labels, positions=(1, 1.4), widths=0.3, patch_artist=True)
    bplot2 = plt.boxplot(condition_dict["half"], labels=labels, positions=(2.5, 2.9), widths=0.3, patch_artist=True)
    bplot3 = plt.boxplot(condition_dict["late"], labels=labels, positions=(4, 4.4), widths=0.3, patch_artist=True)


    for bplot in (bplot, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_positions = [1, 2.5, 4]

    plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)

    plt.ylabel("execution time (seconds)")
    plt.legend(bplot["boxes"], labels, loc="best")
    plt.show()


def remove_append_utterances(total_df):
    utterance_df = total_df.loc[:, ["id", "worker_utterance", "message_time", "msg_status", "mood"]].drop_duplicates().sort_values(by=["id", "message_time"])

    unique_worker_id = list(utterance_df["id"].unique())
    for id in unique_worker_id:
        id_df = utterance_df.loc[(utterance_df["id"] == id)]

        # first, we examine if a worker has multiple utterance status
        if ((len(id_df) != 1) & ("Deleted" in id_df["msg_status"].values)):
            # for each worker, we examine if an utterance was added and then deleted, if so, there must be two duplicated utterances
            dup_utt = id_df.duplicated(subset=["id", "worker_utterance"], keep=False)
            wait_to_check_status = list(id_df[dup_utt]["msg_status"].values)
            for i in range(0, len(wait_to_check_status), 2):
                if ((i%2 == 0) & (wait_to_check_status[i] == "Added") & ((i+1)%2!=0) & (wait_to_check_status[i+1] == "Deleted")):
                    indexes = list(id_df[dup_utt].index)
                    utterance_df.drop(index = indexes, inplace=True)

        # append all of the added utterances from one worker
        total_df.loc[total_df["id"] == id, "final_msg"] = " ".join(total_df.loc[total_df["id"] == id]["worker_utterance"].values)

    return total_df


def len_utterance(total_df):
    utterance_df = total_df.loc[:, ["id", "final_msg", "mood", "interacted"]].drop_duplicates().sort_values(by=["id"])
    utterance_df["len_msg"] = utterance_df["final_msg"].apply(lambda x: len(re.findall(r'\w+', x)))
    print("The average length of the worker utterances: ", utterance_df["len_msg"].sum()/len(utterance_df))

    return utterance_df


def mood_interacted_df(utterance_df):
    # relation between the mood and the interacted type
    interacted_mood = utterance_df.groupby(["mood", "interacted"])["id"].nunique().reset_index()
    bar_x = list(interacted_mood["mood"].unique())

    interacted_mood.loc[17] = [8, "Yes, I did.", 0]

    interacted_mood["mood"] = interacted_mood["mood"].astype(int)
    interacted_mood = interacted_mood.sort_values(by="mood")

    bar_y1 = list(interacted_mood[(interacted_mood["interacted"]=="Yes, I did.")]["id"].values)
    bar_y2 = list(interacted_mood[(interacted_mood["interacted"]=="No, I did not.")]["id"].values)

    return bar_x, (bar_y1, bar_y2)


def task_cxt_df(total_df, condition_dict):
    scale_to_score = {
        "Strongly Disagree": 0,
        "Disagree": 1,
        "Somewhat disagree": 2,
        "Neither agree nor disagree": 3,
        "Somewhat agree": 4,
        "Agree": 5,
        "Strongly agree": 6
    }

    # bar_x = ["familiar_tech", "use_freq", "get_help", "explain_cxt", "cxt_flow", "reply_btn", "feel_control", "ling_cxt",  "seman_cxt", "cogn_cxt"]

    # only focus on the three dimensions of context in the following bar plot
    bar_x = ["ling_cxt",  "seman_cxt", "cogn_cxt"]

    for col in bar_x:
        total_df[col] = total_df[col].map(scale_to_score)

    # extract conditional variables and create list for data
    for key in condition_dict.keys():
        for col in bar_x:
            # for within task type and interacted types
            condition_df = total_df.loc[(total_df["stage"]=="main_"+key) & (total_df["interacted"] == "Yes, I did.")].drop_duplicates(subset=["id"])
            condition_l = condition_df[col].sum(axis=0)/len(condition_df)
            condition_dict[key].append(condition_l)

    dict_keys_list = list(condition_dict.keys())
    bar_y1 = condition_dict[dict_keys_list[0]]
    bar_y2 = condition_dict[dict_keys_list[1]]
    bar_y3 = condition_dict[dict_keys_list[2]]

    return bar_x, (bar_y1, bar_y2, bar_y3)


def barplot(labels, xlabel, ylabel, bar_x, *args):

    # aligned boxplots
    fig, ax= plt.subplots()
    x = np.arange(len(bar_x))+1
    width=0.2
    for i in range(len(labels)):
        ax.bar(x+i*width, args[i], width, label=labels[i])

    ax.set_xticks(x)
    ax.set_xticklabels(bar_x)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()
    plt.show()



if __name__ == "__main__":
    total_df = preprocess_tables_surveys()
    bar_x, bar_y = task_cxt_df(remove_append_utterances(total_df), {"MI_early":[], "MI_half":[], "MI_late":[]})
    barplot(["MI_early", "MI_half", "MI_late"], "context type", "mean score", bar_x, *bar_y)

    # task type v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"MI": [], "non_MI": [], "history": []}, ["early", "half", "late"]), ["early", "half", "late"], ['lightblue', 'lightgreen', 'lightyellow'])

    # # within each task type, interacted v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"early": [], "half": [], "late": []}, ["Yes, I did.", "No, I did not."]), ["interacted", "not interacted"], ['lightblue', 'lightgreen'])