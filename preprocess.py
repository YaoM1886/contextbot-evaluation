import pandas as pd
import numpy as np

# TODO:
#      3. calculate the cognitive load score based on the manual
#      4. task for the H1, toloka dynamic overlap
#      5. task for the H3, google form for the psychologists

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)


worker_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/worker.csv")
behavior_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/behavior.csv")
message_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/message.csv")
time_df = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/time.csv")
pre_survey = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/pre_survey.csv", usecols=["Q10_1", "Q7"])[2:]
post_survey = pd.read_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/post_survey.csv", usecols=["Q7", "Q9", "Q6_1", "Q6_2", "Q6_3", "QID1_1", "QID1_2", "QID1_3", "QID1_4", "QID1_5", "QID1_6", "QID1_7", "QID1_8",                                                                                                                           "Q10", "Q2_1", "Q2_2", "Q2_3", "Q2_4", "Q2_5", "Q2_6", "Q2_7", "Q2_8", "Q11", "Q3_1", "Q3_2", "Q3_3", "Q3_4", "Q3_5", "Q3_6","Q4_1", "Q5", "Q12", "Q12", "TASK_TYPE"])[2:]


# rename columns of the survey
pre_survey.rename(columns={"Q10_1": "mood", "Q7": "prolific_id"}, inplace=True)
post_survey.rename(columns={"Q7":"prolific_id", "Q9": "interacted", "Q6_1":"familiar_tech", "Q6_2":"use_freq", "Q6_3":"get_help", "QID1_1":"explain_cxt", "QID1_2":"cxt_flow", "QID1_3": "reply_btn", "QID1_4": "feel_control", "QID1_5":"ling_cxt", "QID1_6": "seman_cxt", "QID1_7": "attent_check1", "QID1_8": "cogn_cxt", "Q10": "comment_contextbot", "Q11": "attent_check2", "Q4_1":"satis_score", "Q5":"comment_system", "Q12":"preprogrammed"}, inplace=True)



def link_surveys_table(table, pre_survey, post_survey):
    pre_survey_table = pd.merge(table, pre_survey, on="prolific_id", how="left")
    pre_post_survey_table = pd.merge(pre_survey_table, post_survey, on="prolific_id", how="left")

    return pre_post_survey_table


def link_tables():
    time_df.drop(["id"], axis=1, inplace=True)
    message_df.drop("id", axis=1, inplace=True)
    behavior_df.drop("id", axis=1, inplace=True)
    worker_time = pd.merge(worker_df, time_df, left_on="id", right_on="w_id", how="outer").drop(["w_id", "time_stamp"], axis=1)
    worker_time_message = pd.merge(worker_time, message_df, left_on="id", right_on="w_id", how="outer").drop("w_id", axis=1)
    worker_time_message.rename(columns={"time_stamp": "message_time"}, inplace=True)

    worker_time_message_behavior = pd.merge(worker_time_message, behavior_df, left_on="id", right_on="w_id", how="outer").drop("w_id", axis=1)
    worker_time_message_behavior.rename(columns={"time_stamp": "behavior_time"}, inplace=True)

    return worker_time_message_behavior


def reject_rows(total_df):
    # reject rows: either workers returned or timed out, no data in both db and post-survey
    rejected_rows = total_df["worker_utterance"].isna() & total_df["TASK_TYPE"].isna() & total_df["stage"].isna()
    # rejected_id = total_df[rejected_rows]["prolific_id"].unique()

    return rejected_rows


def empty_msg(total_df):
    # no messages were recorded in the db, but the survey was finished
    empty_msg_rows = total_df["worker_utterance"].isna()
    print("Total number of empty messages: ", len(total_df[empty_msg_rows]["id"].unique()))
    return empty_msg_rows

def count_conditions(total_df):
    group_cond = total_df.groupby(["stage"])["id"].nunique()
    print("\nTotal number of unique participants in each condition: ", group_cond)

    interacted_group_cond = total_df.groupby(["stage", "interacted"])["id"].nunique()
    print("\nTotal number of unique participants in each condition and the interacted status: ", interacted_group_cond)


def check_behavior_response(total_df):
    failed_behavior_response_series = total_df[total_df["interacted"]=="No, I did not."].groupby("id")["b_name"].count()
    failed_behavior_response_id = list(failed_behavior_response_series[failed_behavior_response_series!=0].index)

    print("\nTotal number of liars: ", len(failed_behavior_response_id))
    print("Suspect participants who failed to be honest about the interaction: ", worker_df[worker_df["id"].isin(failed_behavior_response_id)])

    # reject the wrong completion id
    wrong_completion_id = total_df[(total_df["stage"]=="main_history_late") & (total_df["b_name"] == "Clicked the ContextBot Icon")].loc[1574, ["id", "prolific_id"]]
    print("\nReject participants who gave the wrong completion code: ", wrong_completion_id)


    failed_behavior_response_id = [37, 108, 204, 266, 283, 340]
    return failed_behavior_response_id



def check_attention_q(total_df):

    # create the df with necessary info for checking the quality
    utterance_df = total_df.loc[:, ["id", "prolific_id", "stage", "worker_utterance", "msg_status", "start_time", "interacted", "attent_check1", "attent_check2"]].drop_duplicates().sort_values(by="start_time")

    # check the attention question 1 under conditions of MI and non-MI, and interacted with ContextBot
    attent_check1 = utterance_df[~utterance_df["stage"].isin(["main_history_early", "main_history_half", "main_history_late"])]
    attent_check1 = attent_check1[attent_check1["interacted"] == "Yes, I did."]

    # number of participants who failed attention check1
    id_failed_check1 = attent_check1[attent_check1["attent_check1"]!="Strongly disagree"]["id"].unique()
    num_failed_check1 = len(id_failed_check1)
    print("\nTotal number of participants who failed attention check 1: ", num_failed_check1)

    # number of participants who failed attention check2
    id_failed_check2 = utterance_df[utterance_df["attent_check2"] != "I give honest answers to this survey."]["id"].unique()
    num_failed_check2 = len(id_failed_check2)
    print("Total number of participants who failed attention check 2: ", num_failed_check2)

    # prolific_id for those who failed either check1 or check2
    rejected_id_attent_checks = list(id_failed_check2)+list(id_failed_check1)
    print("\nReject participants who failed any of the attention checks: ", worker_df[worker_df["id"].isin(rejected_id_attent_checks)])

    #check the utterance quality
    utterance_df = utterance_df[~utterance_df["id"].isin(rejected_id_attent_checks)]
    print("\nTotal number of qualified participants who passed attention checks: ", len(utterance_df["id"].unique()))
    # utterance_df.to_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/data/check_utterance_quality.csv")

    total_df = total_df[~total_df["id"].isin(rejected_id_attent_checks)]
    return total_df

def preprocess_tables_surveys():
    def ghost_rows(total_df):
        #  ghost rows are those rows where data was not properly stored in our db, but in the post-survey
        ghost_rows = total_df.loc[total_df["stage"] != total_df["TASK_TYPE"]]
        # also include those whose behavior data was not correctly recorded in the db
        failed_non_behavior_response_df = total_df[(total_df["interacted"] == "Yes, I did.")]
        ghost_rows = ghost_rows.append(failed_non_behavior_response_df.loc[failed_non_behavior_response_df['b_name'].isnull(), :])
        print("Total number of ghost rows (counted by ID): ", len(ghost_rows["id"].unique()))
        return ghost_rows["id"].unique()

    total_df = link_surveys_table(link_tables(), pre_survey, post_survey).sort_values(by=["stage"])

    # remove rejected rows
    total_df = total_df[~reject_rows(total_df)]

    # remove ghost rows
    ghost_rows = ghost_rows(total_df)
    total_df = total_df[~total_df["id"].isin(ghost_rows)]

    # remove empty messages
    total_df = total_df[~empty_msg(total_df)]

    # remove participants who failed attention checks
    total_df = check_attention_q(total_df)

    # remove participants who lied about the interaction
    total_df = total_df[~total_df["id"].isin(check_behavior_response(total_df))]

    # add spent time as a column to the total_df
    total_df["spent_time"] = pd.to_datetime(total_df["end_time"]) - pd.to_datetime(total_df["start_time"])

    # remove a record with repeated end_time
    total_df = total_df.sort_values('spent_time').drop_duplicates(['id', "start_time", "msg_status", "worker_utterance"], keep='last')

    # no NaN values exist in the time spent table
    print("\nTotal number of final participants: ", len(total_df["id"].unique()))
    # participants in each condition
    count_conditions(total_df)

    return total_df

def eval_ueq(total_df):
    ueq_df = total_df.loc[:, ["id", "prolific_id", "stage", "spent_time", "interacted", "preprogrammed", "Q2_1", "Q2_2", "Q2_3", "Q2_4", "Q2_5", "Q2_6", "Q2_7", "Q2_8"]].drop_duplicates()

    # ueq_df.to_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/ueq.csv")

def check_utterance(total_df):
    utterance_df = total_df.loc[:, ["id", "prolific_id", "worker_utterance", "msg_status"]]

    utterance_df.to_csv("/Users/sylvia/Documents/Netherlands/Course/MasterThesis/Experiments/final_data/check_utterance.csv")

if __name__ == "__main__":

    # preprocess the table and surveys, get the final data
    total_df = preprocess_tables_surveys()
    print(total_df.head())

    # calculate the user experience score, keep the id, prolific_id, stage, spent_time, interacted, Q2_1 to Q2_8, preprogrammed
    eval_ueq(total_df)





































