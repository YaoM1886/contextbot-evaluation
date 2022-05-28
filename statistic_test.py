from scipy import stats
from scipy.stats import levene
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scikit_posthocs as sp


pd.set_option("display.max_columns", None)


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


def post_hoc_dunn_test(dict_values):
    #perform Dunn's test using a Bonferonni correction for the p-values
    print(sp.posthoc_dunn(dict_values, p_adjust = 'bonferroni'))


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

    # # # compare MI, non_MI, history conditions
    MI_dict = {"MI": [], "non_MI":[], "history":[]}
    for key in MI_dict.keys():
        for entry in ["early", "half", "late"]:
            MI_dict[key].append(condition_dict[key+"_"+entry])


    # # compare early, half, late conditions
    # entry_dict = {"early": [], "half":[], "late":[]}
    # for key in entry_dict.keys():
    #     for entry in ["MI", "non_MI", "history"]:
    #         entry_dict[key].extend(condition_dict[entry+"_"+key])



    post_hoc_dunn_test(MI_dict["history"])
    # dunn_df = total_df.loc[total_df["stage"].isin(["main_MI_half", "main_non_MI_half", "main_history_half"])]
    # print(sp.posthoc_dunn(dunn_df,  "mean_task_load","stage", "bonferroni"))

    for key, value in MI_dict.items():
        statistics, pvalue = stats.shapiro(value[0])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)

        statistics, pvalue = stats.shapiro(value[1])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)

        statistics, pvalue = stats.shapiro(value[2])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)


        # check if the variances of two conditions within one group are equal
        print("Variance check:", stats.levene(value[0], value[1], value[2]))

        # what if we use Mann-Whitney test
        print("Kruskal-Wallis Test: ", stats.kruskal(value[0],value[1],value[2]))
        print("\n\n")




    # # check the normality of MI, non_MI, history conditions, finding that none of them is normally distributed
    # normality_test(entry_dict)
    # # use kruskal-wallis
    # non_para_multiple(entry_dict)

    return MI_dict


def within_group_test(total_df, dependent_var):
    condition_dict = {"MI_early":[], "MI_half":[], "MI_late":[], "non_MI_early":[], "non_MI_half":[], "non_MI_late":[]}
    condition_list = ["inter", "not_inter"]
    # extract nine conditional variables and create list for each group
    for key in condition_dict.keys():
        for condition in condition_list:
        # for within task type and interacted types
            condition_l = list(total_df["consis_"+key+"_"+condition].values)
            condition_dict[key].append(condition_l)

    condition_dict["history_early"]=[]
    condition_dict["history_half"]=[]
    condition_dict["history_late"]=[]

    for key in ["early", "half", "late"]:
        # for within task type and interacted types
        condition_h = list(total_df["consis_history_"+key].values)
        condition_dict["history_"+key].extend(condition_h)

    # merge_dict = {"early":[], "half":[], "late":[]}
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict["MI"+"_"+key][0])
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict["history_"+key])


    merge_dict = {"early":[], "half":[], "late":[]}
    for key in list(merge_dict.keys()):
        merge_dict[key].append((condition_dict["MI"+"_"+key][0]))
    for key in list(merge_dict.keys()):
        merge_dict[key].append((condition_dict["MI"+"_"+key][1]))
    for key in list(merge_dict.keys()):
        merge_dict[key].append(condition_dict["history_"+key])
    print(merge_dict)


    # merge_dict = {"MI":[], "non_MI":[]}
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict[key+"_early"][0]+(condition_dict[key+"_half"][0]) + (condition_dict[key+"_late"][0]))
    # for key in list(merge_dict.keys()):
    #     merge_dict[key].append(condition_dict[key+"_early"][1]+(condition_dict[key+"_half"][1])+(condition_dict[key+"_late"][1]))

    # check the normality of each list, use Shapiro-Wilk Test
    # for key, value in merge_dict.items():
    #     statistics, pvalue = stats.shapiro(value[1])
    #     print("The Shapiro-Wilk statistic, p-value of interacted group %s: "%key, statistics, pvalue)
    #
    #     statistics, pvalue = stats.shapiro(value[2])
    #     print("The Shapiro-Wilk statistic, p-value of not interacted group %s: "%key, statistics, pvalue)
    #
    #     # check if the variances of two conditions within one group are equal
    #     print("Variance check:", stats.levene(value[1], value[2]))
    #
    #     # then we use two tailed T-test for independent samples to check if the average values are equal
    #     print("T-test: ", stats.ttest_ind(value[1], value[2]))
    #
    #     # what if we use Mann-Whitney test
    #     print("Mann-Whitney Test: ", stats.mannwhitneyu(x=value[1], y=value[2], alternative='two-sided'))
    #     print("\n\n")

    for key, value in merge_dict.items():
        statistics, pvalue = stats.shapiro(value[0])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)

        statistics, pvalue = stats.shapiro(value[1])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)

        statistics, pvalue = stats.shapiro(value[2])
        print("The Shapiro-Wilk statistic, p-value of group %s: "%key, statistics, pvalue)


        # check if the variances of two conditions within one group are equal
        print("Variance check:", stats.levene(value[0], value[1], value[2]))

        # what if we use Mann-Whitney test
        print("Kruskal-Wallis Test: ", stats.kruskal(value[0],value[1],value[2]))
        print("\n\n")


    return merge_dict


if __name__ == "__main__":
    # ueq_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/ueq.csv", index_col=0)
    # task_load_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/task_load.csv", index_col=0)
    #
    # between_groups_test(task_load_df, "spent_time")
    print("Kruskal-Wallis Test: ", stats.f_oneway([4.0, 6.3, 1.3, 3.3, 3.3, 6.3, 3.3, 2.7, 5.7, 2.0, 4.3, 6.0,5.3, 3.7, 4.0,  2.0, 4.7, 5.3, 3.7, 2.3, 4.0, 5.7, 5.3, 4.7, 2.3], [3.7, 4.7, 1.7, 4.7, 5.7, 3.0, 2.0, 4.7, 5.0, 5.3, 5.3, 5.0, 5.3, 4.7, 4.0,2.0, 1.7, 3.7, 3.3, 3.3, 4.3, 3.7, 3.3, 4.3, 3.3, 3.0, 3.7, 1.3, 4.3, 4.3], [2.0, 3.7, 2.0, 1.0, 4.7, 4.3, 4.3, 3.7, 3.7, 4.7, 4.0, 2.0, 4.3, 3.7, 3.3]))














