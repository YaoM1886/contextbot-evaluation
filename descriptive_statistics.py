import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np
from scipy import stats
from scipy.stats import levene
import scikit_posthocs as sp
from preprocess import total_eval_df
from statistic_test import within_group_test, between_groups_test, normality_test
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def extract_conditional_df(total_df, condition_dict, condition_boxes):

    # extract conditional variables and create list for data
    for key in condition_dict.keys():
        for condition in condition_boxes:
            # for within task type and interacted types
            # condition_l = list(total_df.loc[(total_df["stage"]=="main_non_MI"+"_"+key) & (total_df["actual_interacted"] == condition)].drop_duplicates(subset=["id"])["mean_task_load"].values)

            #  for task types
            condition_l = list(total_df.loc[((total_df["stage"]=="main_MI_"+key) | (total_df["stage"]=="main_non_MI"+"_"+key)) & (total_df["interacted"] == condition)].drop_duplicates(subset=["id"])["mean_pragmatic"].values)
            condition_dict[key].append(condition_l)

    return condition_dict

def boxplot_three_groups_three_each(condition_dict, labels, colors, ylabel):
    # boxplot of the nine stages and their spent time of the task
    x_positions_fmt = ["Early*", "Half", "Late"]

    bplot = plt.boxplot(condition_dict["early"], labels=labels, positions=(1, 1.4, 1.8), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot2 = plt.boxplot(condition_dict["half"], labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot3 = plt.boxplot(condition_dict["late"], labels=labels, positions=(4, 4.4, 4.8), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")


    for bplot in (bplot, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_positions = [1, 2.5, 4]

    plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)

    plt.ylabel(ylabel)
    plt.legend(bplot["boxes"], labels, loc="best", prop={"size":11})
    plt.show()


def boxplot_three_groups_two_each(condition_dict, labels, colors, ylabel):
    # boxplot of the nine stages and their spent time of the task
    x_positions_fmt = ["Early*", "Half", "Late"]

    bplot = plt.boxplot(condition_dict["early"], labels=labels, positions=(1, 1.4), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot2 = plt.boxplot(condition_dict["half"], labels=labels, positions=(2.5, 2.9), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot3 = plt.boxplot(condition_dict["late"], labels=labels, positions=(4, 4.4), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")


    for bplot in (bplot, bplot2, bplot3):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_positions = [1, 2.5, 4]

    plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)

    plt.ylabel(ylabel)
    plt.legend(bplot["boxes"], labels, loc="best")
    plt.show()

def boxplot_two_groups_two_each(condition_dict, labels, colors, ylabel):
    # boxplot of the nine stages and their spent time of the task
    x_positions_fmt = ["MI", "Non-MI"]

    bplot = plt.boxplot(condition_dict["MI"], labels=labels, positions=(1, 1.4), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot2 = plt.boxplot(condition_dict["non_MI"], labels=labels, positions=(2.5, 2.9), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")


    for bplot in (bplot, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_positions = [1, 2.5]

    plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)

    plt.ylabel(ylabel)
    plt.legend(bplot["boxes"], labels, loc="best")
    plt.show()


def boxplot_two_groups_three_each(condition_dict, labels, colors, ylabel):
    # boxplot of the nine stages and their spent time of the task
    x_positions_fmt = ["MI", "Non-MI"]

    bplot = plt.boxplot(condition_dict["MI"], labels=labels, positions=(1, 1.4, 1.8), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    bplot2 = plt.boxplot(condition_dict["non_MI"], labels=labels, positions=(2.5, 2.9, 3.3), widths=0.3, patch_artist=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")


    for bplot in (bplot, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)

    x_positions = [1, 2.5]

    plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)

    plt.ylabel(ylabel)
    plt.legend(bplot["boxes"], labels, loc="best")
    plt.show()



def len_utterance(total_df):

    utterance_df = total_df.loc[:, ["id", "final_msg", "mood", "interacted", "stage"]].drop_duplicates().sort_values(by=["id"])
    utterance_df["len_msg"] = utterance_df["final_msg"].apply(lambda x: len(re.findall(r'\w+', str(x))))
    # find an outlier that has multiple repeated inputs
    worker_repeated_input = utterance_df[utterance_df["len_msg"] == utterance_df["len_msg"].max()]["id"].values[0]
    repeated_input_df = total_df[total_df["id"]==worker_repeated_input][["worker_utterance", "message_time"]].sort_values(by=["message_time"]).drop_duplicates()[:2]
    utterance_df.loc[utterance_df["id"] == worker_repeated_input, "final_msg"] = " ".join(repeated_input_df["worker_utterance"].values)

    # modified the length of that outlier message, and recalculated the message lenghth
    utterance_df["len_msg"] = utterance_df["final_msg"].apply(lambda x: len(re.findall(r'\w+', str(x))))
    print("The average length of the worker utterances: ", utterance_df["len_msg"].sum(axis=0)/len(utterance_df))


    # length of message v.s. pre-task mood
    # utterance_df = map_mood_cat(utterance_df)
    # sns.boxplot(x=utterance_df["mood"], y=utterance_df["len_msg"])
    # plt.show()
    return utterance_df


def map_mood_cat(utterance_df):
    mood_to_cat = {1: "pleasant", 2: "pleasant", 3: "pleasant", 4: "pleasant", 5: "unpleasant", 6:"unpleasant", 7: "unpleasant", 8:"unpleasant", 9: "neutral"}
    utterance_df["mood"] = utterance_df["mood"].map(mood_to_cat)
    return utterance_df


def mood_interacted_df(utterance_df):
    # map the mood, 1-4 mapped to pleasant, 5-8 mapped to unpleasant, 9 mapped to neutral
    utterance_df = map_mood_cat(utterance_df)
    # relation between the mood and the interacted type, cluster the interacted type based on the mood score
    interacted_mood = utterance_df.groupby(["mood", "interacted"])["id"].nunique().unstack("interacted")
    print(interacted_mood)
    interacted_mood.plot(kind="bar", stacked=True)
    plt.xticks(rotation=0)
    plt.xlabel("mood type")
    plt.ylabel("number of participants")
    plt.show()


def task_cxt_df(total_df, condition_dict):

    # bar_x = ["familiar_tech", "use_freq", "get_help", "explain_cxt", "cxt_flow", "reply_btn", "feel_control", "ling_cxt",  "seman_cxt", "cogn_cxt"]

    # only focus on the three dimensions of context in the following bar plot
    bar_x = ["ling_cxt",  "seman_cxt", "cogn_cxt"]

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
    b_df = total_df[(total_df["interacted"]=="Yes, I did.")].loc[:, ["id", "stage", "b_name", "behavior_time", "message_time", "final_msg", "interacted", "mood", "satis_score"]].drop_duplicates().sort_values(by=["id", "behavior_time"])

    response_after_help_df = b_df.drop_duplicates()

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

    # check the workers who clicked the icon
    b_name_df.loc[(b_name_df["b_name_Clicked the ContextBot Icon"]>0), "click_bot_icon"] = 1

    # reorder the name of the cxt type names
    interacted_cxt_types = ["click_bot_icon"]
    interacted_cxt_types.extend(list(cxt_type_dict.keys()))
    interacted_cxt_types.append("interacted_all_cxt")

    for interacted_cxt_type in interacted_cxt_types:
        print("Number of workers who finished %s: "%(interacted_cxt_type), len(b_name_df[b_name_df[interacted_cxt_type]==1]))


    # check who only clicked the bot icon, we named it as not actually interated
    only_bot_icon_worker = list(b_name_df[b_name_df["id"].isin(b_name_df[(b_name_df["interacted_prompt_cxt"]==1)]["id"])]["id"])
    print(only_bot_icon_worker)

    # # check which worker explored which cognitive cxt
    # id_user_intention = b_name_df[b_name_df["b_name_User's intention"]>0]["id"].values
    # id_prev_coach_intention = b_name_df[b_name_df["b_name_Previous coach's intention"]>0]["id"].values
    # id_your_intention = b_name_df[b_name_df["b_name_Your intention"]>0]["id"].values

    def boxplot_cxt_satis(b_name_df):
        cxt_type_dict = ["click_bot_icon", "interacted_social_cxt", "interacted_ready_ling_cxt"]
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

    # plot the interaction stage with the satisfaction score
    # boxplot_cxt_satis(b_name_df)

    return b_name_df.loc[:, ["id", "interacted_social_cxt", "interacted_ready_ling_cxt", "interacted_ling_cxt", "interacted_seman_cxt", "interacted_ready_cog_cxt", "interacted_cog_cxt", "interacted_prompt_cxt", "interacted_all_cxt", "click_bot_icon"]]



def plain_three_groups_boxplot(condition_dict, ylabel, labels):
    boxes = list(condition_dict.values())
    plt.boxplot(boxes, showmeans=True, labels=labels, meanprops = {'marker':'*'})
    plt.ylabel(ylabel)
    plt.show()


def find_low_consis_patterns(total_eval_df, b_name_df):
    def cxt_type_ratio(stage_pattern, b_name_df, total_eval_df):
        # calculate the ratio of interaction cxt type
        interacted_cxt_types = list(b_name_df.columns)
        interacted_cxt_types.remove("id")
        for cxt in interacted_cxt_types:
            cxt_ratio = round(b_name_df.loc[b_name_df["id"].isin(list(stage_pattern["id"].values))][cxt].sum(axis=0)/len(stage_pattern),3)
            print("Ratio of cxt %s in group %s"%(cxt, stage_pattern["stage"].unique()), cxt_ratio)

        var_df = total_eval_df.loc[total_eval_df["id"].isin(list(stage_pattern["id"].values))].drop_duplicates(["id"])
        print("Number of samples in this group: ", len(var_df))
        var_pattern = var_df[["spent_time", "satis_score", "mean_ueq", "mean_pragmatic", "mean_hedonic", "mean_task_load", "familiar_tech", "use_freq", "get_help", "explain_cxt", "cxt_flow", "reply_btn", "feel_control", "ling_cxt",  "seman_cxt", "cogn_cxt"]].mean(axis=0)
        print(var_pattern)


        var_dict = {}
        for var in ["spent_time", "satis_score", "mean_ueq", "mean_pragmatic", "mean_hedonic", "mean_task_load", "familiar_tech", "use_freq", "get_help", "explain_cxt", "cxt_flow", "reply_btn", "feel_control", "ling_cxt",  "seman_cxt", "cogn_cxt"]:
            var_dict.setdefault(var, list(var_df[var].values))
        return var_dict

    # list avg_consistency scores for each group across stages and interacted types
    avg_consis_groups = pd.pivot_table(total_eval_df, index=["stage", "actual_interacted"], values=["avg_consis",  "spent_time", "satis_score", "mean_ueq", "mean_pragmatic", "mean_hedonic", "mean_task_load"], aggfunc=np.mean)
    print("Var scores for each group: ", avg_consis_groups)


    # we finally included one history-late group as the control group
    stage_history = total_eval_df.loc[(total_eval_df["stage"]=="main_history_early")].drop_duplicates(["id"])
    stage_history["pattern"] = "history"
    print(len(stage_history))
    history_dict = {}
    for var in ["spent_time", "satis_score", "mean_ueq", "mean_pragmatic", "mean_hedonic", "mean_task_load"]:
        history_dict.setdefault(var, list(stage_history[var].values))

    # "main_MI_early", "main_MI_half", "main_non_MI_early", "main_non_MI_half",
    for stage in ["main_MI_early", "main_non_MI_early"]:
        avg_consis = avg_consis_groups.loc[stage, "TRUE"]["avg_consis"]

        # e.g. define two groups where one is lower than the avg consistency and the other one is higher. So then we check what happened in each group in terms of other variables;
        stage_low = total_eval_df.loc[(total_eval_df["avg_consis"]<avg_consis) & (total_eval_df["stage"]==stage) & (total_eval_df["actual_interacted"]=="TRUE")].drop_duplicates(["id"])
        stage_high = total_eval_df.loc[(total_eval_df["avg_consis"]>=avg_consis) & (total_eval_df["stage"]==stage) & (total_eval_df["actual_interacted"]=="TRUE")].drop_duplicates(["id"])


        print("Low group: ")
        low_dict = cxt_type_ratio(stage_low, b_name_df, total_eval_df)
        print("\n")
        print("High group: ")
        high_dict = cxt_type_ratio(stage_high, b_name_df, total_eval_df)
        for var in low_dict.keys():
            print(stats.mannwhitneyu(x=low_dict[var], y=high_dict[var]))
        stage_low["pattern"] = "low"
        stage_high["pattern"] = "high"

        for var in history_dict.keys():
            dunn_df = pd.concat([stage_low, stage_high, stage_history], axis=0)
            print(sp.posthoc_dunn(dunn_df,  var, "pattern", "bonferroni"))


            print(var)
            print("shapiro for low",stats.shapiro(low_dict[var]))
            print("shapiro for high",stats.shapiro(high_dict[var]))
            print("shapiro for history", stats.shapiro(history_dict[var]))
            print("\n")
            print("Variance check", stats.levene(high_dict[var], history_dict[var]))
            print("\n")
            print(stats.mannwhitneyu(x=high_dict[var], y=history_dict[var]))
            print("T-test", stats.ttest_ind(high_dict[var], history_dict[var]))
            print("Kruskal", stats.kruskal(low_dict[var], high_dict[var], history_dict[var]))
            print("One-way", stats.f_oneway(low_dict[var], high_dict[var], history_dict[var]))
            print("\n\n")
        print("\n\n")




    # print("Comments of ContextBot or the system", MI_late_low[["id", "comment_contextbot", "comment_system", "preprogrammed"]])

    # print("Comments of ContextBot or the system", MI_late_high[["id", "comment_contextbot", "comment_system", "preprogrammed"]])


def trade_off_consis_time(total_eval_df):
    '''
    divided the group into highly and lowly consistent, test the time difference
    :param total_eval_df:
    :return:
    '''
    highly_consis = total_eval_df[(total_eval_df["actual_interacted"] == "TRUE") & (total_eval_df["avg_consis"]>4)].drop_duplicates("id")
    lowly_consis = total_eval_df[(total_eval_df["actual_interacted"] == "TRUE") & (total_eval_df["avg_consis"]<=4)].drop_duplicates("id")

    # for history, highly and lowly consistent
    highly_consis_his = total_eval_df[(total_eval_df["actual_interacted"] == "Unknown") & (total_eval_df["avg_consis"]>4)].drop_duplicates("id")
    lowly_consis_his = total_eval_df[(total_eval_df["actual_interacted"] == "Unknown") & (total_eval_df["avg_consis"]<=4)].drop_duplicates("id")
    print(highly_consis_his)
    print(lowly_consis_his)


    for stage in ["main_history_early", "main_history_half", "main_history_late"]:
        highly_g_h = list(highly_consis_his[highly_consis_his["stage"]==stage]["spent_time"].values)
        lowly_g_h = list(lowly_consis_his[lowly_consis_his["stage"]==stage]["spent_time"].values)


        print(stage)
        print("avg highly", np.mean(highly_g_h), np.std(highly_g_h))
        print("avg lowly", np.mean(lowly_g_h), np.std(lowly_g_h))

        if len(highly_g_h)>=3:
            print("shapiro for high",stats.shapiro(highly_g_h))
        if len(lowly_g_h)>=3:

            print("shapiro for low",stats.shapiro(lowly_g_h))
        print("\n")
        print(stats.mannwhitneyu(highly_g_h, lowly_g_h))


    # for stage in ["main_MI_early", "main_MI_half", "main_MI_late", "main_non_MI_early", "main_non_MI_half", "main_non_MI_late"]:
    #     highly_g = list(highly_consis[highly_consis["stage"]==stage]["spent_time"].values)
    #     lowly_g = list(lowly_consis[lowly_consis["stage"]==stage]["spent_time"].values)
    #
    #     print(stage)
    #     print("avg highly", np.mean(highly_g), np.std(highly_g))
    #     print("avg lowly", np.mean(lowly_g), np.std(lowly_g))
    #
    #     if len(highly_g)>=3:
    #         print("shapiro for high",stats.shapiro(highly_g))
    #     if len(lowly_g)>=3:
    #
    #         print("shapiro for low",stats.shapiro(lowly_g))
    #
    #     print("\n")
    #     if ((len(highly_g)>=3) & (len(lowly_g)>=3)):
    #         # print("Variance check", stats.levene(highly_g, lowly_g))
    #         print("\n")
    #         print(stats.kruskal(highly_g, lowly_g, history_late_time))
    #         # print("T-test", stats.ttest_ind(highly_g, lowly_g))
    #         print("\n\n")
    #


def qualitative_feedback(total_eval_df):

    feedback_df = total_eval_df.dropna(subset=["comment_system", "comment_contextbot"], how="all", axis=0).drop_duplicates("id").loc[:, ["id", "stage", "spent_time", "final_msg", "interacted", "comment_system", "comment_contextbot"]]
    print(feedback_df[["comment_system"]].values)


if __name__ == "__main__":
    total_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/total_df.csv", index_col=0)

    eval_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/sampled_eval_consis_psych.csv", index_col=0, usecols=["id", "stage", "interacted", "avg_consis"]).reset_index()

    b_name_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/b_name_df.csv", index_col=0)

    # remove outliers of consistency scores
    eval_df = eval_df.drop(eval_df[(eval_df["stage"]=="MI_early") & (eval_df["interacted"]=="Yes, I did.") & (eval_df["avg_consis"]==2.0)].index)
    eval_df = eval_df.drop(eval_df[(eval_df["stage"]=="MI_late") & (eval_df["interacted"]=="No, I did not.") & (eval_df["avg_consis"]==1.7)].index)
    # boxplot_three_groups_three_each(within_group_test(eval_df, "avg_consis"), ["Interacted (MI)", "Not interacted (MI)", "History"], ['dimgrey', 'silver', 'whitesmoke'], "Consistency score")
    total_eval = total_eval_df(total_df, eval_df)
    # find_low_consis_patterns(total_eval, b_name_df)
    # trade_off_consis_time(total_eval)
    qualitative_feedback(total_eval)






    # task_load_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/task_load.csv", index_col=0)
    # ueq_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/ueq.csv", index_col=0)


    # eval_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/sampled_eval_consis_psych.csv", index_col=0, usecols=["id", "stage", "interacted", "avg_consis"]).reset_index().dropna(axis=0, subset=["avg_consis"], how="all")

    # eval_df = eval_df.drop(eval_df[(eval_df["stage"]=="non_MI_early") & (eval_df["interacted"]=="Yes, I did.") & (eval_df["avg_consis"]==1.7)].index)
    # eval_df = eval_df.drop(eval_df[(eval_df["stage"]=="MI_late") & (eval_df["interacted"]=="No, I did not.") & (eval_df["avg_consis"]==2.0)].index)

    # eval_df = eval_df.drop(eval_df[(eval_df["interacted"]=="No, I did not.") & (eval_df["avg_consis"]==1.3)].index)

    # all_interacted_worker = [35, 45, 63, 64, 70, 73, 91, 109, 118, 119, 120, 123, 125, 131, 137, 145, 185, 193, 223, 235, 236, 239, 242, 245, 246, 262, 269, 271, 276, 281, 282, 309, 321, 336, 341, 354, 356, 361, 373, 391]
    # eval_df = eval_df.loc[(eval_df["id"].isin(all_interacted_worker)) | (eval_df["interacted"] == "No, I did not.") | (eval_df["stage"] == "history_early") | (eval_df["stage"] == "history_half") | (eval_df["stage"] == "history_late")]
    # merge_dict = {'MI': [6.0, 7.0, 3.5, 6.5, 5.0, 5.0, 6.0, 5.5, 6.5, 5.0, 6.0, 5.5, 5.5, 6.0], 'non_MI': [2.0, 4.5, 5.5, 5.0, 5.0, 6.0, 3.5, 5.0, 6.0, 6.0, 6.0, 3.5, 3.5, 5.0, 4.5]}
    # plt.boxplot([merge_dict["MI"], merge_dict["non_MI"]], labels=["MI", "Non_MI"], showfliers=True, showmeans=True, meanprops={'marker':'o', "markerfacecolor":"black", "markeredgecolor": "black"}, sym="+")
    # plt.show()

    # boxplot_two_groups_three_each(within_group_test(eval_df, "avg_psych"), ["Early", "Half", "Late"], ['dimgrey', 'silver', 'whitesmoke'], "Professional score")



    # task type v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"MI": [], "non_MI": [], "history": []}, ["early", "half", "late"]), ["early", "half", "late"], ['lightblue', 'lightgreen', 'lightyellow'])

    # # within each task type, interacted v.s. execution time
    # boxplot_stage_time(extract_conditional_df(total_df, {"early": [], "half": [], "late": []}, ["True", "False"]), ["interacted", "not interacted"], ['lightblue', 'lightgreen'])

    # mood v.s. length of msg
    # len_utterance(total_df)

    # mood v.s. interacted type
    # mood_interacted_df(len_utterance(total_df))

    # context type v.s. mean score of agreement
    # bar_x, bar_y = task_cxt_df(remove_append_utterances(total_df), {"MI_early":[], "MI_half":[], "MI_late":[]})
    # barplot(["MI_early", "MI_half", "MI_late"], "context type", "mean score", bar_x, *bar_y)


    # behavior v.s. satisfaction score

    # b_name_df = behavior_satis_df(total_df)
    # b_name_df.to_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/b_name_df.csv")
    # b_name_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/b_name_df.csv", index_col=0)
    # behavior_agreement_df(total_df, b_name_df)

    # boxplot_stage_time(extract_conditional_df(total_df, {"MI": [], "non_MI": [], "history": []}, ["early", "half", "late"]), ["early", "half", "late"], ['lightblue', 'lightgreen', 'lightyellow'])
    # boxplot_three_groups_two_each(within_group_test(ueq_df, "mean_hedonic"), ["Interacted", "Not interacted"], ['dimgrey', 'silver'], "Mean hedonic score")
    # boxplot_two_groups_two_each(within_group_test(ueq_df, "mean_hedonic"), ["Interacted", "Not interacted"], ['dimgrey', 'silver'], "Mean hedonic score")
    #
    # boxplot_three_groups_two_each(within_group_test(ueq_df, "mean_pragmatic"), ["Not interacted", "History"], ['dimgrey', 'silver'], "Mean pragmatic score")
    # MI_dict = between_groups_test(ueq_df, "mean_pragmatic")
    # boxplot_three_groups_three_each(MI_dict, ["MI", "Non-MI", "History"], ['dimgrey', 'silver', 'whitesmoke'], "Mean pragmatic score")

    # boxplot_two_groups_two_each(within_group_test(task_load_df, "mean_task_load"), ["Interacted", "Not interacted"], ['dimgrey', 'silver'], "Mean task load")
    # boxplot_three_groups_two_each(within_group_test(task_load_df, "mean_task_load"), ["Non-MI-not-interacted", "History"], ['dimgrey', 'silver'], "Mean task load")
    # boxplot_three_groups_three_each(between_groups_test(task_load_df, "spent_time"), ["Early", "Half", "Late"], ['dimgrey', 'silver', 'whitesmoke'], "Execution time (seconds)")