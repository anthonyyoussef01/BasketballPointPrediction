import pandas as pd
import datetime
from sklearn.model_selection import train_test_split

class DataParser():

    # remove unnecessary columns from the data frame
    removal_columns = ['StatID', 'Scope', 'TeamID', 'PlayerID', 'SeasonType', 'Season', 'Name',
       'Team', 'Position', 'Updated', 'GameID', 'OpponentID', 'Games',
       'OpponentStatID', 'DateTime', 'IsGameOver', 'Started', 'LineupStatus', 'LineupConfirmed',
       'Seconds', 'OffensiveRebounds', 'DefensiveRebounds', 'DoubleDoubles', 'TripleDoubles',
       'Assists', 'Steals', 'BlockedShots', 'Turnovers', 'PersonalFouls',
       'PlusMinus', 'BlocksAgainst',	'FantasyDataSalary',
       'PlayerEfficiencyRating', 'OffensiveReboundsPercentage',
       'DefensiveReboundsPercentage', 'TotalReboundsPercentage',
       'AssistsPercentage', 'StealsPercentage', 'BlocksPercentage',
       'TurnOversPercentage', 'Possessions', 'InjuryBodyPart', 'InjuryNotes', 'InjuryStartDate',
       'DivisionWins', 'DivisionLosses', 'ConferenceWins', 
       'ConferenceLosses', 'Wins', 'Losses', 'FanDuelPosition', 'DraftKingsPosition', 'YahooPosition', 'FantasyDraftSalary',
       'FieldGoalsMade', 'FieldGoalsAttempted', 'ThreePointersMade', 'ThreePointersAttempted', 'FreeThrowsMade', 'FreeThrowsAttempted',
       'OpponentPosition', 'FantasyDraftPosition', 'Unnamed: 78', 'Closed', 'Unnamed: 80', 'OpponentPositionRank']

    columns_to_average = ['FantasyPoints', 'FanDuelSalary',
       'DraftKingsSalary', 'YahooSalary', 'Minutes', 'UsageRatePercentage',
       'FantasyPointsFanDuel', 'FantasyPointsDraftKings',
       'FantasyPointsYahoo',
       'FantasyPointsFantasyDraft', 'Points']

    def __init__(self, csv_file):

        self.data = pd.read_csv("julius_randle_career_stats_by_game.csv", error_bad_lines=False)

        data_cpy = self.data.copy()

        # creates the points category
        data_cpy["Points"] = (data_cpy["ThreePointersMade"] * 3) + ((data_cpy["FieldGoalsMade"] - data_cpy["ThreePointersMade"]) * 2) + data_cpy["FreeThrowsMade"]

        self.data_cpy = data_cpy

    def clean_data(self):
        
        data_good = self.data_cpy.drop(labels=self.removal_columns, axis=1)

        data_good['InjuryStatus'].fillna(value="", inplace=True)

        data_good2 = data_good.copy()

        # change Home or Away to 0s and 1s
        data_good2['HomeOrAway'] = data_good['HomeOrAway'].map({'HOME': 1, 'AWAY': 0})

        # injury: non-null `InjuryStatus` paired with 0 `Minutes`
        injured_count = 0
        days_since_injury = [] ## will add as a `DaysSinceInjury` column (data point)
        for row in range(len(data_good)):
            if data_good['InjuryStatus'][row] != "" and data_good['Minutes'][row] == 0:
                injured_count = 0
            else:
                if row == 0:
                    injured_count = 0
                else:
                    injured_count += (datetime.datetime.strptime(data_good['Day'][row],  '%m/%d/%y') - datetime.datetime.strptime(data_good['Day'][row - 1],  '%m/%d/%y')).days
            days_since_injury.append(injured_count)

        # add the new column to the dataframe
        data_good2["DaysSinceInjury"] = days_since_injury

        # some of the data has only been collected after 2016, so remove all entries with null values
        data_good2 = data_good2[data_good2['OpponentRank'].notnull()]
        data_good2 = data_good2[data_good2['YahooSalary'].notnull()]
        data_good2 = data_good2[data_good2["DraftKingsSalary"].notnull()]

        # reset index
        data_good2.reset_index(inplace=True, drop=True)

        self.data_final = data_good2.copy()

    def average_last_n(self, df, window_size, reference_column, output_column):
        """
        Finds the average of the previous window_size number of values for a column,
        and creates a new column to represent these values.
        """
        lst = []

        for i in range(len(df)):
            if(i < window_size):
                lst.append(df[reference_column].iloc[0:i].mean())
            else:
                lst.append(df[reference_column].iloc[i-window_size:i].mean())

        lst[0] = 0
        df[output_column] = lst
        return


    def average_data(self):

        # iteratively update all the columns to be average of last 10
        for col in self.columns_to_average:
            new_name = col + "Avg"
            self.average_last_n(self.data_final, 10, col, new_name)

            if col != "Points":
                self.data_final.drop(labels=col, inplace=True, axis=1)

        # perform one-hot encoding with the opponents
        self.data_final = pd.concat([self.data_final, pd.get_dummies(self.data_final.Opponent, prefix='Opponent')], axis=1)
        self.data_final = self.data_final.drop(labels=["Day", "Opponent"], axis=1)
        self.data_final.drop(index=self.data_final.index[0], axis=0, inplace=True)


        self.data_final = self.data_final[self.data_final["DaysSinceInjury"] != 0]
        self.data_final.drop(columns="InjuryStatus", inplace=True)

    def create_features_and_target_data(self):

        self.features = self.data_final.drop(columns="Points")

        self.target = self.data_final["Points"]

    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.target, random_state=1000)
        return X_train, X_test, y_train, y_test