# a team will have a list of seasons
import statistics


class team:

    def __init__(self, ID):
        self.ID = ID
        self.seasons = []

    def add_season(self, a_season):
        self.seasons.append(a_season)

    def get_seasons(self):
        return self.seasons

    # gets the season by the year
    def get_season(self, year):
        for a_season in self.seasons:
            if a_season.year == year:
                return a_season
        return None


# a season will have a list of games
class season:

    def __init__(self, year, ID):
        self.year = year
        self.ID = ID

        self.total_goals = 0
        self.total_agienst = 0
        self.total_games = 0

        # library holding date and corresponding goals gotten
        self.last_for = []
        self.last_agienst = []

    def get_game_total(self):
        return self.total_games

    def get_total_for(self):
        return self.total_goals

    # given a time, a list of [score, time] and an amount of data to look at, returns the medien
    def get_medien(self, time, data, amount):
        games = self.get_recent(time, data, amount)
        if games is not None:
            return statistics.median(games)
        else:
            return None

    def get_total_medien(self, time, data):
        amount, games = self.get_all_recent(time, data)
        if games is not None:
            return amount, statistics.median(games)
        else:
            return None, None

    def get_average(self, time, data, amount):
        games = self.get_recent(time, data, amount)
        if games is not None:
            return statistics.mean(games)
        else:
            return None

    # given a time, a list of [score, time] and an amount of data to look at, returns the average
    def get_total_average(self, time, data):
        amount, games = self.get_all_recent(time, data)
        if games is not None:
            return amount, statistics.mean(games)
        else:
            return None, None

    def get_total_agi(self):
        return self.total_agienst

    # updates relevant instance variables
    def add_game(self, a_game):
        self.total_games = self.total_goals + 1
        self.total_goals = self.total_goals + a_game.goals_scored
        self.total_agienst = self.total_agienst + a_game.opponent_goals
        self.modify_recent_games(a_game)

    def modify_recent_games(self, a_game):
        self.last_for = self.update_list(a_game.date, a_game.goals_scored, self.last_for)
        self.last_agienst = self.update_list(a_game.date, a_game.opponent_goals, self.last_agienst)

    # MIGHT NEED TO REVERSE THIS
    def update_list(self, date, data, a_list):
        a_list.append([date, data])
        a_list.sort()
        return a_list

    # given a date, this function gets the last amount value before the date, in the list
    def get_recent(self, date, a_list, amount):
        final_list = []
        for i in range(len(a_list)):
            if a_list[i][0] >= date:
                if i >= amount:
                    for j in range(1, amount + 1):
                        final_list.append(a_list[i - j][1])
                    return final_list
                else:
                    return None
        return None

    # slight modification, get the average of all games before it. if its th first game is returns none
    def get_all_recent(self, date, a_list):
        final_list = []
        total = 0
        for i in range(len(a_list)):
            if a_list[i][0] >= date:
                if total != 0:
                    return i, final_list
                else:
                    return None, None
            else:
                total = total + 1
                final_list.append(a_list[i][1])
        return None, None


class game:

    def __init__(self, the_season, ID, opp_ID, goals_scored, opponent_goals, date):
        self.the_season = the_season
        self.ID = ID
        self.opp_ID = opp_ID
        self.goals_scored = goals_scored
        self.opponent_goals = opponent_goals
        self.date = date
