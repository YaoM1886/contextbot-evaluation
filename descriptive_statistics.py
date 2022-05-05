import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
import seaborn as sns

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

    # this may filter out some messages that might appear to be empty in the system(due to the db connection issue maybe)
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
                if (((i+1)>=len(wait_to_check_status)) & (wait_to_check_status[i] == "Added")):
                    continue
                elif ((i%2 == 0) & (wait_to_check_status[i] == "Added") & ((i+1)%2!=0) & (wait_to_check_status[i+1] == "Deleted")):
                    indexes = list(id_df[dup_utt].index)[i:i+2]
                    utterance_df.drop(index = indexes, inplace=True)

        # append all of the added utterances from one worker
        utterance_df.loc[utterance_df["id"] == id, "final_msg"] = " ".join(utterance_df.loc[utterance_df["id"] == id]["worker_utterance"].values)
    total_df = pd.merge(total_df, utterance_df.loc[:, ["id", "final_msg"]], on=["id"], how="outer")
    total_df["final_msg"].fillna(value=total_df["worker_utterance"], inplace=True)
    return total_df


def len_utterance(total_df):
    utterance_df = total_df.loc[:, ["id", "final_msg", "mood", "interacted"]].drop_duplicates().sort_values(by=["id"])
    utterance_df["len_msg"] = utterance_df["final_msg"].apply(lambda x: len(re.findall(r'\w+', str(x))))
    print("The average length of the worker utterances: ", utterance_df["len_msg"].sum(axis=0)/len(utterance_df))
    # # length of message v.s. pre-task mood
    # sns.boxplot(x=utterance_df["mood"], y=utterance_df["len_msg"])
    # plt.show()

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


def behavior_satis_df(total_df):
    # check the relation between behavior and satisfaction score

    # extract all the workers who had behavior, order by the behavior time
    b_df = total_df[total_df["interacted"]=="Yes, I did."].loc[:, ["id", "stage", "b_name", "behavior_time", "message_time", "final_msg", "interacted", "mood", "satis_score"]].drop_duplicates().sort_values(by=["id", "behavior_time"])

    # validity check: check if the first message time is behind the first behavior
    unique_worker_id = list(b_df["id"].unique())
    not_valid_id = []
    for id in unique_worker_id:
        id_df = b_df.loc[(b_df["id"] == id)]
        min_msg_time = id_df["message_time"].min()
        min_b_time = id_df["behavior_time"].min()
        if min_msg_time < min_b_time:
            not_valid_id.append(id)
    print("Workers who responded first and then interacted (partially) with ContextBot: ", not_valid_id)

    # message time column might repeat due to the message status
    b_df.drop(["message_time"], axis=1, inplace=True)
    response_after_help_df = b_df[~b_df["id"].isin(not_valid_id)].drop_duplicates()

    # give one-hot encoding of all the behavior types
    behavior_dummies = pd.get_dummies(response_after_help_df[["b_name"]])
    b_name_cols = behavior_dummies.columns
    behavior_dummies_id = pd.concat([response_after_help_df["id"], behavior_dummies], axis=1)
    unique_b_id = list(response_after_help_df["id"].unique())
    # create a new df to store the behavior names, for each participant, we have one record, and the behavior name is the col name
    b_name_df = pd.DataFrame(columns=b_name_cols)
    for id in unique_b_id:
        b_name_df.loc[id] = behavior_dummies_id[behavior_dummies_id["id"]==id].loc[:,b_name_cols].sum()
    b_name_df.reset_index(inplace=True)
    b_name_df.rename(columns={"index": "id"}, inplace=True)


    # count the number of finishing each dimension of context

    # check if each dimension of cxt was interacted with
    cxt_type_dict = {"interacted_social_cxt":["b_name_Hmm...I think so.", "b_name_Sure, let us begin!"], "interacted_ready_ling_cxt":["b_name_Yes, very clear!", "b_name_Well...I still don't understand."],
                     "interacted_ling_cxt":["b_name_Clicked the back button", "b_name_Clicked the linked text", "b_name_Clicked the next button"],
                     "interacted_seman_cxt":["b_name_Not really...I am confused.", "b_name_Yes, I'm totally aware now!"],
                     "interacted_ready_cog_cxt":['b_name_Yes, I am ready!', 'b_name_Okay, share what you have with me.'],
                     "interacted_cog_cxt":["b_name_Your intention", "b_name_User's intention", "b_name_Previous coach's intention"], "interacted_prompt_cxt":["b_name_Wow, I think I know enough now!", "b_name_Okay, anything else?"]}

    for cxt_type, cxt_option in cxt_type_dict.items():
        if len(cxt_option) == 2:
            b_name_df.loc[(b_name_df[cxt_option[0]]>0)|(b_name_df[cxt_option[1]]>0), cxt_type] = 1
            b_name_df.loc[(b_name_df[cxt_option[0]]==0)&(b_name_df[cxt_option[1]]==0), cxt_type] = 0
        else:
            b_name_df.loc[(b_name_df[cxt_option[0]]>0)|(b_name_df[cxt_option[1]]>0) |(b_name_df[cxt_option[2]]>0), cxt_type] = 1
            b_name_df.loc[(b_name_df[cxt_option[0]]==0)&(b_name_df[cxt_option[1]]==0)&(b_name_df[cxt_option[2]]==0), cxt_type] = 0


    # check the workers who finished the whole interaction
    b_name_df.loc[(b_name_df["interacted_social_cxt"]>0)&(b_name_df["interacted_ready_ling_cxt"]>0) & (b_name_df["interacted_ling_cxt"]) &(b_name_df["interacted_seman_cxt"]>0) & (b_name_df[ "interacted_ready_cog_cxt"]>0)& (b_name_df["interacted_cog_cxt"]>0) & (b_name_df["interacted_prompt_cxt"]>0), "interacted_all_cxt"] = 1

    # check the workers who only clicked the icon
    b_name_df.loc[(b_name_df["b_name_Clicked the ContextBot Icon"]>0), "only_bot_icon"] = 1

    interacted_cxt_types = ["only_bot_icon"]
    interacted_cxt_types.extend(list(cxt_type_dict.keys()))
    interacted_cxt_types.append("interacted_all_cxt")

    for interacted_cxt_type in interacted_cxt_types:
        print("Number of workers who finished %s: "%(interacted_cxt_type), len(b_name_df[b_name_df[interacted_cxt_type]==1]))

    # # check which worker explored which cognitive cxt
    # id_user_intention = b_name_df[b_name_df["b_name_User's intention"]>0]["id"].values
    # id_prev_coach_intention = b_name_df[b_name_df["b_name_Previous coach's intention"]>0]["id"].values
    # id_your_intention = b_name_df[b_name_df["b_name_Your intention"]>0]["id"].values

    def boxplot_cxt_satis(b_name_df):
        cxt_type_dict = ["only_bot_icon", "interacted_social_cxt", "interacted_ready_ling_cxt"]
        box_dict = dict.fromkeys(cxt_type_dict)
        for i in range(0, len(cxt_type_dict)-1):
            box_dict[cxt_type_dict[i]] = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df[cxt_type_dict[i]]==1) & (b_name_df[cxt_type_dict[i+1]]==0)]["id"])]["satis_score"].values)

        box_dict["soc_ling"] = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df["interacted_ling_cxt"]==1) & (b_name_df["interacted_seman_cxt"]==0)]["id"])]["satis_score"].values)
        box_dict["soc_ling_seman"] = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df["interacted_ling_cxt"]==1) & (b_name_df["interacted_cog_cxt"]==0) & (b_name_df["interacted_seman_cxt"]==1)]["id"])]["satis_score"].values)
        box_dict["soc_ling_seman_cog"] = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df["interacted_ling_cxt"]==1) & (b_name_df["interacted_cog_cxt"]==1) & (b_name_df["interacted_prompt_cxt"]==0)]["id"])]["satis_score"].values)
        box_dict["all_cxt"] = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df["interacted_all_cxt"]==1)]["id"])]["satis_score"].values)

        del box_dict["interacted_ready_ling_cxt"]

        df = pd.DataFrame({key:pd.Series(value) for key, value in box_dict.items() })

        df.boxplot(grid=False)
        plt.xticks(rotation=10)
        plt.xlabel("interacted context type")
        plt.ylabel("satisfaction score")

        plt.show()


    b_name_df = pd.merge(response_after_help_df.loc[:, ["id", "satis_score", "stage"]], b_name_df, on=["id"], how="inner").drop_duplicates()
    b_name_df["interacted_all_cxt"].fillna(0, inplace=True)
    boxplot_cxt_satis(b_name_df)





if __name__ == "__main__":
    total_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/final_df.csv", index_col=0)

    # the following process is based on the principle that one worker has one final utterance
    total_df = remove_append_utterances(total_df)


    # behavior v.s. satisfaction score
    behavior_satis_df(total_df)





    # task type v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"MI": [], "non_MI": [], "history": []}, ["early", "half", "late"]), ["early", "half", "late"], ['lightblue', 'lightgreen', 'lightyellow'])

    # # within each task type, interacted v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"early": [], "half": [], "late": []}, ["Yes, I did.", "No, I did not."]), ["interacted", "not interacted"], ['lightblue', 'lightgreen'])

    # context type v.s. mean score of agreement
    # bar_x, bar_y = task_cxt_df(remove_append_utterances(total_df), {"MI_early":[], "MI_half":[], "MI_late":[]})
    # barplot(["MI_early", "MI_half", "MI_late"], "context type", "mean score", bar_x, *bar_y)