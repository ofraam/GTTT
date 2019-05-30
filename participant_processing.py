import pandas as pd
import seaborn as sns


def compute_bonus(turk_file, results_file):
    turk_df = pd.read_csv(turk_file)
    results_df = pd.read_csv(results_file)
    turk_df = turk_df[['userid','AssignmentId','WorkerId']]
    merged_df = pd.merge(results_df, turk_df, on='userid')
    print(merged_df['validated_correct'].head())
    merged_df['bonus'] = merged_df['validated_correct'].apply(lambda x: 1.0 if x==True else 0)
    print(merged_df['bonus'].head())
    merged_df.to_csv('experiments_files/merged_30052019.csv')
    bonus_df = merged_df.loc[merged_df['bonus'] == 1.0]
    bonus_df = bonus_df[['AssignmentId','WorkerId','bonus']]
    bonus_df['bonus_reason'] = 'Bonus for tictactoe task. Thank you for participating in our study!'
    bonus_df.to_csv('experiments_files/bonus_ttt_30052019.csv')
# print('hello')

if __name__ == "__main__":
    # turk1 = pd.read_csv('experiments_files/turk1.csv')
    # turk2 = pd.read_csv('experiments_files/turk2.csv')
    # turk3 = pd.read_csv('experiments_files/turk3.csv')
    # frames = [turk1, turk2, turk3]
    # turk_df = pd.concat(frames)
    # turk_df.to_csv('experiments_files/turk_all_11042019.csv')
    # compute_bonus('experiments_files/turk_all_11042019.csv', 'experiments_files/tttResultsApr11_2019.csv')
    compute_bonus('experiments_files/300519_turk.csv', 'experiments_files/tttResults300519.csv')