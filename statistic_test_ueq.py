from scipy import stats
from scipy.stats import levene
import numpy as np
import matplotlib.pyplot as plt
from preprocess import preprocess_tables_surveys, eval_ueq
import scikit_posthocs as sp


total_df = preprocess_tables_surveys()
total_df["spent_time"] = total_df["spent_time"].dt.total_seconds()
ueq_df = eval_ueq(total_df)

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


    #first check if each group follows the normal distribution
    # print(stats.shapiro(non_interacted_half))






#      then check, if the normality does not fit





def two_way_anova(ueq_df, col_str):
    # test the effect of two categorical variables: interacted+entry points on quantitative dependent variables
    interacted_early = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_early") | (ueq_df["stage"]=="main_non_MI_early"))].drop_duplicates()[col_str].astype(int).values
    interacted_half = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_half") | (ueq_df["stage"]=="main_non_MI_half"))].drop_duplicates()[col_str].astype(int).values
    interacted_late = ueq_df[(ueq_df["interacted"]=="Yes, I did.") & ((ueq_df["stage"]=="main_MI_late") | (ueq_df["stage"]=="main_non_MI_late"))].drop_duplicates()[col_str].astype(int).values

    non_interacted_early = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_early") | (ueq_df["stage"]=="main_non_MI_early"))].drop_duplicates()[col_str].astype(int).values
    non_interacted_half = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_half") | (ueq_df["stage"]=="main_non_MI_half"))].drop_duplicates()[col_str].astype(int).values
    non_interacted_late = ueq_df[(ueq_df["interacted"]=="No, I did not.") & ((ueq_df["stage"]=="main_MI_late") | (ueq_df["stage"]=="main_non_MI_late"))].drop_duplicates()[col_str].astype(int).values
    #
    #
    # early = []
    # early.append(list(interacted_early))
    # early.append(list(non_interacted_early))
    #
    # half = []
    # half.append(list(interacted_half))
    # half.append(list(non_interacted_half))
    #
    # late = []
    # late.append(list(interacted_late))
    # late.append(list(non_interacted_late))
    #
    # labels = ["interacted", "not interacted"]
    #
    # bplot = plt.boxplot(early, labels=labels, positions=(1, 1.4), widths=0.3, patch_artist=True)
    # bplot2 = plt.boxplot(half, labels=labels, positions=(2.5, 2.9), widths=0.3, patch_artist=True)
    # bplot3 = plt.boxplot(late, labels=labels, positions=(4, 4.4), widths=0.3, patch_artist=True)
    #
    # colors = ['lightblue', 'lightgreen']
    # for bplot in (bplot, bplot2, bplot3):
    #     for patch, color in zip(bplot['boxes'], colors):
    #         patch.set_facecolor(color)
    #
    # x_positions = [1, 2.5, 4]
    # x_positions_fmt = ["early", "half", "late"]
    # plt.xticks([i+0.8/2 for i in x_positions], x_positions_fmt)
    #
    # plt.ylabel("satisfaction score")
    # plt.legend(bplot["boxes"], labels, loc="best")
    # plt.show()

    print(stats.kruskal(interacted_early, interacted_half, interacted_late, non_interacted_early, non_interacted_half, non_interacted_late))
    # Dunn's test
    print(sp.posthoc_dunn([interacted_early, interacted_half, interacted_late, non_interacted_early, non_interacted_half, non_interacted_late], p_adjust = 'bonferroni'))





# two_way_anova(ueq_df, "satis_score")
between_groups_history(ueq_df)








