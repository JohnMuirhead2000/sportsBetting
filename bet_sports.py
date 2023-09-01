import math
import os
import random
from prettytable import PrettyTable
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import store_data

finalResults = PrettyTable()
finalResults.field_names = ["Dataset #", "Classifier Model", "RMSE"]


def main():
    print("lets bet SPORTS")

    # first build a list of teams
    teams = []
    filename = os.getcwd() + '/archive-2/team_info.csv'
    data = pd.read_csv(filename)
    target = data.iloc[:, 0].reset_index().values.tolist()
    for team in target:
        a_team = store_data.team(team[1])
        teams.append(a_team)

    filename = os.getcwd() + '/archive-2/game.csv'
    data = pd.read_csv(filename)

    dates = data.iloc[:, 1].reset_index().values.tolist()
    dates = [i[1] for i in dates]
    processed_dates = []
    for date in dates:
        if date not in processed_dates:
            processed_dates.append(date)
            # each season needs a team, which needs a number of games

    # processed_dates is a list of years:
    # teams is a list of team IDs'

    filled_teams = []
    for team in teams:
        for year in processed_dates:
            season = store_data.season(year, team.ID)
            team.add_season(season)
        filled_teams.append(team)

    # for team in filled_teams:
    #     print(str(team.ID))

    # we now have a list of teams, filled_teams, which stores each team, id and seasons. Now just
    # loop through the games and add each game appropriate

    games = data.iloc[:, :].reset_index().values.tolist()
    for game in games:
        # Only account for regular season
        if game[3] == "R":
            date = game[2]
            away_team = game[5]
            home_team = game[6]
            away_goals = game[7]
            home_goals = game[8]
            time = game[4]

            # game from home teams perspective:
            homes_game = store_data.game(date, home_team, away_team, home_goals, away_goals, time)

            # game from away teams perspective:
            aways_game = store_data.game(date, home_team, away_team, home_goals, away_goals, time)

            # for each team, if the ID in the game matches, for each of their seasons if the seasons matches,
            # then add that game to the seasons
            for team in filled_teams:
                if team.ID == int(home_team):
                    for season in team.get_seasons():
                        if season.year == date:
                            season.add_game(homes_game)

                if team.ID == int(away_team):
                    for season in team.get_seasons():
                        if season.year == date:
                            season.add_game(aways_game)
    # test_season = filled_teams[0].get_seasons()[0]
    # test_date = '2016-10-20T23:10:00Z'
    # print("test_date = " + str(test_date))
    # print("test_season.year = " + str(test_season.year))
    # print(test_season.get_recent(test_date, test_season.last_for, 2))
    # print(test_season.last_for)

    # The data will contain the following information:
    # for each game which exists get: their season average scored, season average scored on for both
    # teams
    # the average for both those figured for the last 3 games and 5
    data_set1 = []
    data_set2 = []
    data_set3 = []
    data_set4 = []
    data_set5 = []
    data_set6 = []
    y_data = []
    random.shuffle(games)
    for game in games:
        if game[3] == 'R':
            game_time = game[4]

            team_1 = get_team(filled_teams, game[5])
            team1_season = team_1.get_season(game[2])

            # the first output is the number of games. These values should be the same in the following 2 lines
            games_1, team1_average_for = team1_season.get_total_average(game_time, team1_season.last_for)
            games_1, team1_average_agi = team1_season.get_total_average(game_time, team1_season.last_agienst)

            games_1, team1_med_for = team1_season.get_total_medien(game_time, team1_season.last_for)
            games_1, team1_med_agi = team1_season.get_total_medien(game_time, team1_season.last_agienst)

            team1_3_average_for = team1_season.get_average(game_time, team1_season.last_for, 3)
            team1_3_average_agi = team1_season.get_average(game_time, team1_season.last_agienst, 3)

            team1_3_med_for = team1_season.get_medien(game_time, team1_season.last_for, 3)
            team1_3_med_agi = team1_season.get_medien(game_time, team1_season.last_agienst, 3)

            team1_5_average_for = team1_season.get_average(game_time, team1_season.last_for, 5)
            team1_5_average_agi = team1_season.get_average(game_time, team1_season.last_agienst, 5)

            team1_5_med_for = team1_season.get_medien(game_time, team1_season.last_for, 5)
            team1_5_med_agi = team1_season.get_medien(game_time, team1_season.last_agienst, 5)

            team_2 = get_team(filled_teams, game[6])
            team2_season = team_2.get_season(game[2])

            games_2, team2_average_for = team2_season.get_total_average(game_time, team2_season.last_for)
            games_2, team2_average_agi = team2_season.get_total_average(game_time, team2_season.last_agienst)

            games_2, team2_med_for = team2_season.get_total_medien(game_time, team2_season.last_for)
            games_2, team2_med_agi = team2_season.get_total_medien(game_time, team2_season.last_agienst)

            team2_3_average_for = team2_season.get_average(game_time, team2_season.last_for, 3)
            team2_3_average_agi = team2_season.get_average(game_time, team2_season.last_agienst, 3)

            team2_3_med_for = team2_season.get_medien(game_time, team2_season.last_for, 3)
            team2_3_med_agi = team2_season.get_medien(game_time, team2_season.last_agienst, 3)

            team2_5_average_for = team2_season.get_average(game_time, team2_season.last_for, 5)
            team2_5_average_agi = team2_season.get_average(game_time, team2_season.last_agienst, 5)

            team2_5_med_for = team2_season.get_medien(game_time, team2_season.last_for, 5)
            team2_5_med_agi = team2_season.get_medien(game_time, team2_season.last_agienst, 5)

            # make sure all the data exists to add this game
            if team1_3_average_for is not None and team1_3_average_agi is not None and \
                    team1_5_average_for is not None and team1_5_average_agi is not None and \
                    team2_3_average_for is not None and team2_3_average_agi is not None and \
                    team2_5_average_for is not None and team2_5_average_agi is not None and \
                    team1_average_for is not None and team1_average_agi is not None and \
                    team1_med_for is not None and team1_med_agi is not None and \
                    team2_med_for is not None and team2_med_agi is not None and \
                    team1_3_med_for is not None and team1_3_med_agi is not None and \
                    team1_5_med_for is not None and team1_5_med_agi is not None and \
                    team2_3_med_for is not None and team2_3_med_agi is not None and \
                    team2_5_med_for is not None and team2_5_med_agi is not None and \
                    team2_average_for is not None and team2_average_agi is not None:
                # We will use this data to create 6  different sets of arguments
                # 1 - Each teams average numer of goals scored for and against for the season
                # 2 - Each teams average numer of goals scored for and against for the season, past 3 and past 5 games
                # 3 - Each teams median numer of goals scored for and against for the season
                # 4 - Each teams median numer of goals scored for and against for the season, past 3 and past 5 games
                # 5 - 1 and 3 combined
                # 6 - 2 and 4 combined
                # note, additionally, all data sets will have the total number of games
                data_1 = [games_1,
                          team1_average_for, team1_average_agi,
                          games_2,
                          team2_average_for, team2_average_agi,
                          ]
                data_2 = [games_1,
                          team1_average_for, team1_average_agi,
                          team1_3_average_for, team1_3_average_agi,
                          team1_5_average_for, team1_5_average_agi,
                          games_2,
                          team2_average_for, team2_average_agi,
                          team2_3_average_for, team2_3_average_agi,
                          team2_5_average_for, team2_5_average_agi,
                          ]
                data_3 = [games_1,
                          team1_med_for, team1_med_agi,
                          games_2,
                          team2_med_for, team2_med_agi,
                          ]
                data_4 = [games_1,
                          team1_med_for, team1_med_agi,
                          team1_3_med_for, team1_3_med_agi,
                          team1_5_med_for, team1_5_med_agi,
                          games_2,
                          team2_med_for, team2_med_agi,
                          team2_3_med_for, team2_3_med_agi,
                          team2_5_med_for, team2_5_med_agi
                          ]
                data_5 = [games_1,
                          team1_average_for, team1_average_agi,
                          team1_med_for, team1_med_agi,
                          games_2,
                          team2_average_for, team2_average_agi,
                          team2_med_for, team2_med_agi,
                          ]
                data_6 = [games_1,
                          team1_average_for, team1_average_agi,
                          team1_med_for, team1_med_agi,
                          team1_3_average_for, team1_3_average_agi,
                          team1_3_med_for, team1_3_med_agi,
                          team1_5_average_for, team1_5_average_agi,
                          team1_5_med_for, team1_5_med_agi,
                          games_2,
                          team2_med_for, team2_med_agi,
                          team2_average_for, team2_average_agi,
                          team2_3_average_for, team2_3_average_agi,
                          team2_3_med_for, team2_3_med_agi,
                          team2_5_average_for, team2_5_average_agi,
                          team2_5_med_for, team2_5_med_agi]
                # y data is how many goals were scored in the same
                y_addition = game[7] + game[8]

                data_set1.append(data_1)
                data_set2.append(data_2)
                data_set3.append(data_3)
                data_set4.append(data_4)
                data_set5.append(data_5)
                data_set6.append(data_6)
                y_data.append(y_addition)

    # x_data and y_data should be filled
    # The data contains the following information about a game:
    # the first teams number of games that seasons,
    # the first teams average goals per game
    # #the first teams average goals scored on per game
    # the first teams average goals per game for the last 3 games
    # #the first teams average goals scored on per game for the last 3 games
    # the first teams average goals per game for the last 5 games
    # #the first teams average goals scored on per game for the last 5 games
    # all the same data for team2

    # the target is how many total goals were scored in the game:

    # we then normalize the data
    data_set1 = preprocessing.normalize(data_set1)
    data_set2 = preprocessing.normalize(data_set2)
    data_set3 = preprocessing.normalize(data_set3)
    data_set4 = preprocessing.normalize(data_set4)
    data_set5 = preprocessing.normalize(data_set5)
    data_set6 = preprocessing.normalize(data_set6)

    x_train_1, x_test_1, y_train1, y_test1 = train_test_split(data_set1, y_data, test_size=0.3, shuffle=False)
    x_train_2, x_test_2, y_train2, y_test2 = train_test_split(data_set2, y_data, test_size=0.3, shuffle=False)
    x_train_3, x_test_3, y_train3, y_test3 = train_test_split(data_set3, y_data, test_size=0.3, shuffle=False)
    x_train_4, x_test_4, y_train4, y_test4 = train_test_split(data_set4, y_data, test_size=0.3, shuffle=False)
    x_train_5, x_test_5, y_train5, y_test5 = train_test_split(data_set5, y_data, test_size=0.3, shuffle=False)
    x_train_6, x_test_6, y_train6, y_test6 = train_test_split(data_set6, y_data, test_size=0.3, shuffle=False)
    print("NOW TRAINING THE AI")
    RF_1 = RandomForestClassifier()
    RF_2 = RandomForestClassifier()
    RF_3 = RandomForestClassifier()
    RF_4 = RandomForestClassifier()
    RF_5 = RandomForestClassifier()
    RF_6 = RandomForestClassifier()

    BC_1 = BaggingClassifier()
    BC_2 = BaggingClassifier()
    BC_3 = BaggingClassifier()
    BC_4 = BaggingClassifier()
    BC_5 = BaggingClassifier()
    BC_6 = BaggingClassifier()

    LR_1 = LogisticRegression(max_iter=1000000)
    LR_2 = LogisticRegression(max_iter=1000000)
    LR_3 = LogisticRegression(max_iter=1000000)
    LR_4 = LogisticRegression(max_iter=1000000)
    LR_5 = LogisticRegression(max_iter=1000000)
    LR_6 = LogisticRegression(max_iter=1000000)

    print("TRAINING DATA 1")
    print("_____________________________________________________________________")
    # train our data
    RF_1.fit(x_train_1, y_train1)
    RF_1_predictions = RF_1.predict(x_test_1)
    # RF_1_predictions holds the predictions, y_test1 holds reality
    print("just trained random forest on data set 1")

    # train our data
    BC_1.fit(x_train_1, y_train1)
    BC_1_predictions = BC_1.predict(x_test_1)
    # BC_1_predictions holds the predictions, y_test1 holds reality
    print("just trained bagging classifier on data set 1")

    # train our data
    LR_1.fit(x_train_1, y_train1)
    LR_1_predictions = LR_1.predict(x_test_1)
    # LR_1_predictions holds the predictions, y_test1 holds reality
    print("just trained logistic reasoning on data set 1\n")

    print("TRAINING DATA 2")
    print("_____________________________________________________________________")
    # train our data
    RF_2.fit(x_train_2, y_train2)
    RF_2_predictions = RF_2.predict(x_test_2)
    # RF_2_predictions holds the predictions, y_test2 holds reality
    print("just trained random forest on data set 2")

    # train our data
    BC_2.fit(x_train_2, y_train2)
    BC_2_predictions = BC_2.predict(x_test_2)
    # BC_2_predictions holds the predictions, y_test1 holds reality
    print("just trained bagging classifier on data set 2")

    # train our data
    LR_2.fit(x_train_2, y_train2)
    LR_2_predictions = LR_2.predict(x_test_2)
    # LR_2_predictions holds the predictions, y_test2 holds reality
    print("just trained logistic reasoning on data set 2\n")

    print("TRAINING DATA 3")
    print("_____________________________________________________________________")
    # train our data
    RF_3.fit(x_train_3, y_train3)
    RF_3_predictions = RF_3.predict(x_test_3)
    # RF_3_predictions holds the predictions, y_test3 holds reality
    print("just trained random forest on data set 3")

    # train our data
    BC_3.fit(x_train_3, y_train3)
    BC_3_predictions = BC_3.predict(x_test_3)
    # BC_2_predictions holds the predictions, y_test1 holds reality
    print("just trained bagging classifier on data set 3")

    # train our data
    LR_3.fit(x_train_3, y_train3)
    LR_3_predictions = LR_3.predict(x_test_3)
    # LR_3_predictions holds the predictions, y_test3 holds reality
    print("just trained logistic reasoning on data set 3\n")

    print("TRAINING DATA 4")
    print("_____________________________________________________________________")
    # train our data
    RF_4.fit(x_train_4, y_train4)
    RF_4_predictions = RF_4.predict(x_test_4)
    # RF_4_predictions holds the predictions, y_test4 holds reality
    print("just trained random forest on data set 4")

    # train our data
    BC_4.fit(x_train_4, y_train4)
    BC_4_predictions = BC_4.predict(x_test_4)
    # BC_4_predictions holds the predictions, y_test4 holds reality
    print("just trained bagging classifier on data set 4")

    # train our data
    LR_4.fit(x_train_4, y_train4)
    LR_4_predictions = LR_4.predict(x_test_4)
    # LR_4_predictions holds the predictions, y_test4 holds reality
    print("just trained logistic reasoning on data set 4\n")

    print("TRAINING DATA 5")
    print("_____________________________________________________________________")
    # train our data
    RF_5.fit(x_train_5, y_train5)
    RF_5_predictions = RF_5.predict(x_test_5)
    # RF_5_predictions holds the predictions, y_test5 holds reality
    print("just trained random forest on data set 5")

    # train our data
    BC_5.fit(x_train_5, y_train5)
    BC_5_predictions = BC_5.predict(x_test_5)
    # BC_5_predictions holds the predictions, y_test5 holds reality
    print("just trained bagging classifier on data set 5")

    # train our data
    LR_5.fit(x_train_5, y_train5)
    LR_5_predictions = LR_5.predict(x_test_5)
    # LR_5_predictions holds the predictions, y_test5 holds reality
    print("just trained logistic reasoning on data set 5\n")

    print("TRAINING DATA 6")
    print("_____________________________________________________________________")
    # train our data
    RF_6.fit(x_train_6, y_train6)
    RF_6_predictions = RF_6.predict(x_test_6)
    # RF_6_predictions holds the predictions, y_test6 holds reality
    print("just trained random forest on data set 6")

    # train our data
    BC_6.fit(x_train_6, y_train6)
    BC_6_predictions = BC_6.predict(x_test_6)
    # BC_6_predictions holds the predictions, y_test6 holds reality
    print("just trained bagging classifier on data set 6")

    # train our data
    LR_6.fit(x_train_6, y_train6)
    LR_6_predictions = LR_6.predict(x_test_6)
    # LR_6_predictions holds the predictions, y_test6 holds reality
    print("just trained logistic reasoning on data set 6\n")

    print("now we find which model-data combination provides the most accurate results")

    result_RF_1 = RMS_error(RF_1_predictions, y_test1)
    result_BC_1 = RMS_error(BC_1_predictions, y_test1)
    result_LR_1 = RMS_error(LR_1_predictions, y_test1)

    result_RF_2 = RMS_error(RF_2_predictions, y_test2)
    result_BC_2 = RMS_error(BC_2_predictions, y_test2)
    result_LR_2 = RMS_error(LR_2_predictions, y_test2)

    result_RF_3 = RMS_error(RF_3_predictions, y_test3)
    result_BC_3 = RMS_error(BC_3_predictions, y_test3)
    result_LR_3 = RMS_error(LR_3_predictions, y_test3)

    result_RF_4 = RMS_error(RF_4_predictions, y_test4)
    result_BC_4 = RMS_error(BC_4_predictions, y_test4)
    result_LR_4 = RMS_error(LR_4_predictions, y_test4)

    result_RF_5 = RMS_error(RF_5_predictions, y_test5)
    result_BC_5 = RMS_error(BC_5_predictions, y_test5)
    result_LR_5 = RMS_error(LR_5_predictions, y_test5)

    result_RF_6 = RMS_error(RF_6_predictions, y_test6)
    result_BC_6 = RMS_error(BC_6_predictions, y_test6)
    result_LR_6 = RMS_error(LR_6_predictions, y_test6)

    print("RESULTS...")
    print("_____________________________________________________________________")

    # note using y_test1 is arbitrary, any of the y_test's should virtually the same result
    base_error = RMS_error([7] * len(y_test1), y_test1)
    # print("In a model with minimal intelligence (always guesses the rounded mean goal total of 7 goals"
    #       " the error is " + str(base_error))

    finalResults.add_rows(
        [
            ["/", "Simply Guessing 7", base_error],
            ["Data 1", "Random Forest", result_RF_1],
            ["Data 1", "Bagging", result_BC_1],
            ["Data 1", "Logistic Regression", result_LR_1],

            ["", "", ""],

            ["Data 2", "Random Forest", result_RF_2],
            ["Data 2", "Bagging", result_BC_2],
            ["Data 2", "Logistic Regression", result_LR_2],

            ["", "", ""],

            ["Data 3", "Random Forest", result_RF_3],
            ["Data 3", "Bagging", result_BC_3],
            ["Data 3", "Logistic Regression", result_LR_3],

            ["", "", ""],

            ["Data 4", "Random Forest", result_RF_4],
            ["Data 4", "Bagging", result_BC_4],
            ["Data 4", "Logistic Regression", result_LR_4],

            ["", "", ""],

            ["Data 5", "Random Forest", result_RF_5],
            ["Data 5", "Bagging", result_BC_5],
            ["Data 5", "Logistic Regression", result_LR_5],

            ["", "", ""],

            ["Data 6", "Random Forest", result_RF_6],
            ["Data 6", "Bagging", result_BC_6],
            ["Data 6", "Logistic Regression", result_LR_6]
        ]
    )

    print(finalResults)


# calculates the RMS of two sets. Note it is assumed that each set has the same amount of data
def RMS_error(set_1, set_2):
    square_total = 0
    for i in range(len(set_1)):
        current_square = math.pow((set_1[i] - set_2[i]), 2)
        square_total = current_square + square_total
    return math.sqrt(square_total / len(set_1))


def get_team(teams, ID):
    for team in teams:
        if team.ID == ID:
            return team


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
