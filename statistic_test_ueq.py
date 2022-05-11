from scipy import stats
from scipy.stats import levene
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from descriptive_statistics import plain_three_groups_boxplot



pd.set_option("display.max_columns", None)



def between_groups_history(ueq_df):
    for i in ("Q2_1", "Q2_2", "Q2_3", "Q2_4", "Q2_5", "Q2_6", "Q2_7", "Q2_8"):
        ueq_df[i] = ueq_df[i].astype(int)
    #check if the groups between early, half, and late conditions differ from each other
    ueq_df["mean_pragmatic"] = (ueq_df[["Q2_1", "Q2_2", "Q2_3", "Q2_4"]].sum(axis=1))/4
    history_early = ueq_df[(ueq_df["stage"] == "main_history_early")]["mean_pragmatic"]
    history_half = ueq_df[(ueq_df["stage"] == "main_history_half")]["mean_pragmatic"]
    history_late = ueq_df[(ueq_df["stage"] == "main_history_late")]["mean_pragmatic"]

    interacted_early = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_early") | (ueq_df["stage"]=="main_non_MI_early"))].drop_duplicates()["mean_pragmatic"]
    interacted_half = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_half") | (ueq_df["stage"]=="main_non_MI_half"))].drop_duplicates()["mean_pragmatic"]
    interacted_late = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_late") | (ueq_df["stage"]=="main_non_MI_late"))].drop_duplicates()["mean_pragmatic"]

    non_interacted_early = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_early") | (ueq_df["stage"]=="main_non_MI_early"))].drop_duplicates()["mean_pragmatic"]
    non_interacted_half = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_half") | (ueq_df["stage"]=="main_non_MI_half"))].drop_duplicates()["mean_pragmatic"]
    non_interacted_late = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_late") | (ueq_df["stage"]=="main_non_MI_late"))].drop_duplicates()["mean_pragmatic"]
    print(stats.kruskal(interacted_early, non_interacted_early))


def comparison_ueq():
    condition_dict = {"MI_early":[], "MI_half":[], "MI_late":[], "non_MI_early":[], "non_MI_half":[], "non_MI_late":[],
                      "history_early":[], "history_half":[], "history_late":[]}


def normality_test(condition_dict):
    '''
    check the normality of each list, use Shapiro-Wilk Test
    :param condition_dict:
    :return:
    '''
    for key, value in condition_dict.items():
        statistics, pvalue = stats.shapiro(value)
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)

        if pvalue < 0.05:
            print("group %s is not normally distributed."%key)


def non_para_multiple(condition_dict):
    '''
    use non-parametric test: Kruskal-Wallis to check if the medians among multiple groups are equal
    :param condition_dict:
    :return:
    '''

    if len(list(condition_dict.keys())) == 3:
        # append values to a new dict key with three groups
        new_dict = {"three_groups":[]}
        for value in condition_dict.values():
            new_dict["three_groups"].append(value)

        statistics, pvalue = stats.kruskal(new_dict["three_groups"][0], new_dict["three_groups"][1], new_dict["three_groups"][2])
        print("The Kruskal-Wallis statistic, pvalue of group %s: "%(condition_dict.keys()), statistics, pvalue)

        if pvalue < 0.05:
            print("group is significantly different")


def between_groups_test(total_df, dependent_var):

    '''
    statistical test between conditional groups, either across MI conditions or entry points
    :param ueq_df: the df with score of ueq
    :param condition_dict: which groups to compare
    :return:
    '''
    condition_dict = {"MI_early":[], "MI_half":[], "MI_late":[], "non_MI_early":[], "non_MI_half":[], "non_MI_late":[], "history_early":[], "history_half":[], "history_late":[]}
    # extract nine conditional variables and create list for each group
    for key in condition_dict.keys():
        # for within task type and interacted types
        condition_l = list(total_df.loc[(total_df["stage"]=="main_"+key)].drop_duplicates(subset=["id"])[dependent_var].values)
        condition_dict[key].extend(condition_l)

    # # compare MI, non_MI, history conditions
    # MI_dict = {"MI": [], "non_MI":[], "history":[]}
    # for key in MI_dict.keys():
    #     for entry in ["early", "half", "late"]:
    #         MI_dict[key].extend(condition_dict[key+"_"+entry])
    #
    # # visualize the three groups with boxplot
    # plain_three_groups_boxplot(MI_dict, "mean pragmatic score", ["MI", "non_MI", "history"])
    #
    # # check the normality of MI, non_MI, history conditions, finding that none of them is normally distributed
    # normality_test(MI_dict)
    # # use kruskal-wallis
    # non_para_multiple(MI_dict)

    # compare early, half, late conditions
    entry_dict = {"early": [], "half":[], "late":[]}
    for key in entry_dict.keys():
        for entry in ["MI", "non_MI", "history"]:
            entry_dict[key].extend(condition_dict[entry+"_"+key])

    # visualize the three groups with boxplot
    plain_three_groups_boxplot(entry_dict, "mean pragmatic score", ["early", "half", "late"])

    # check the normality of MI, non_MI, history conditions, finding that none of them is normally distributed
    normality_test(entry_dict)
    # use kruskal-wallis
    non_para_multiple(entry_dict)

    return None

def between_groups_task_load(total_df):
    condition_dict = {"MI_early":[], "MI_half":[], "MI_late":[], "non_MI_early":[], "non_MI_half":[], "non_MI_late":[],
                      "history_early":[], "history_half":[], "history_late":[]}

    # extract nine conditional variables and create list for each group
    for key in condition_dict.keys():
        # for within task type and interacted types
        condition_l = list(total_df.loc[(total_df["stage"]=="main_"+key)].drop_duplicates(subset=["id"])["mean_task_load"].values)
        condition_dict[key].extend(condition_l)

    # check the normality of each list, use Shapiro-Wilk Test
    for key, value in condition_dict.items():
        statistics, pvalue = stats.shapiro(value)
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)





    # use non-parametric test: Kruskal-Wallis to check if the medians among groups are equal
    for condition1 in ["MI", "non_MI", "history"]:
        statistics, pvalue = stats.kruskal(condition_dict[condition1+"_"+"early"], condition_dict[condition1+"_"+"half"], condition_dict[condition1+"_"+"late"])
        print("The Kruskal-Wallis statistic, pvalue of group %s: "%condition1, statistics, pvalue)

    return condition_dict

def within_group_test(total_df, dependent_var):
    condition_dict = {"MI_early":[], "MI_half":[], "MI_late":[], "non_MI_early":[], "non_MI_half":[], "non_MI_late":[]}
    condition_list = ["Yes, I did.", "No, I did not."]
    # extract nine conditional variables and create list for each group
    for key in condition_dict.keys():
        for condition in condition_list:
        # for within task type and interacted types
            condition_l = list(total_df.loc[(total_df["stage"]=="main_"+key) & (total_df["interacted"]==condition)].drop_duplicates(subset=["id"])[dependent_var].values)
            condition_dict[key].append(condition_l)


    merge_dict = {"early":[], "half":[], "late":[]}
    for key in list(merge_dict.keys()):
        merge_dict[key].append(condition_dict["MI"+"_"+key][0]+(condition_dict["non_MI"+"_"+key][0]))
    for key in list(merge_dict.keys()):
        merge_dict[key].append(condition_dict["MI"+"_"+key][1]+(condition_dict["non_MI"+"_"+key][1]))

    # merge_dict = {"MI":[], "non_MI":[]}
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict[key+"_early"][0]+(condition_dict[key+"_half"][0]) + (condition_dict[key+"_late"][0]))
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict[key+"_early"][1]+(condition_dict[key+"_half"][1])+(condition_dict[key+"_late"][1]))

    # check the normality of each list, use Shapiro-Wilk Test
    for key, value in merge_dict.items():
        statistics, pvalue = stats.shapiro(value[0])
        print("The Shapiro-Wilk statistic, p-value of interacted group %s: "%key, statistics, pvalue)

        statistics, pvalue = stats.shapiro(value[1])
        print("The Shapiro-Wilk statistic, p-value of not interacted group %s: "%key, statistics, pvalue)

        # check if the variances of two conditions within one group are equal
        print("Variance check:", stats.levene(value[0], value[1]))

        # then we use two tailed T-test for independent samples to check if the average values are equal
        print("T-test: ", stats.ttest_ind(value[0], value[1]))

        # what if we use Mann-Whitney test
        print("Mann-Whitney Test: ", stats.mannwhitneyu(x=value[0], y=value[1], alternative='two-sided'))
        print("\n\n")
    #
    # # use non-parametric test: Kruskal-Wallis to check if the medians among groups are equal
    # for condition1 in ["MI", "non_MI", "history"]:
    #     statistics, pvalue = stats.kruskal(condition_dict[condition1+"_"+"early"], condition_dict[condition1+"_"+"half"], condition_dict[condition1+"_"+"late"])
    #     print("The Kruskal-Wallis statistic, pvalue of group %s: "%condition1, statistics, pvalue)

    return merge_dict


if __name__ == "__main__":
    # total_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/total_df.csv", index_col=0)
    # total_df = task_cognitive_load(total_df)
    # within_group_task_load(total_df, "mean_task_load")

    ueq_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/ueq.csv", index_col=0)

    # between_groups_ueq(ueq_df, "mean_pragmatic")
    within_group_test(ueq_df, "mean_pragmatic")











