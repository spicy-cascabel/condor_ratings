import math
from typing import Dict, List, Tuple
import mysql.connector
import unittest
import time
import sys
from collections import defaultdict

mysql_db_host = 'condor.live'
mysql_db_user = 'necrobot-read'
mysql_db_passwd = 'necrobot-read'
mysql_db_name = 'season_x'

PRIOR_STDEV = 600.0
PRIOR_ELOS_FILENAME = 'data/ratings_w0.csv'
ELO_RESULTS_FILENAME = 'data/ratings_cad'
RECORDS_FILENAME = 'data/records_cad'
LEAGUE_DATABASE_NAME = 'condor_x'
LOG_FILENAME = 'data/elorate_log.txt'
LEAGUE_TAG = 'cad'
AUTOMATCH_DEADLINE = '2020-06-28 08:00:00'

ignored_racers = [
]

custom_matches = [
    ('slimo', 'incnone', 3, 0),
]

# The biggielevs model ------------------------------


def biggielevs_phi(x: float) -> float:
    alpha = 0.42
    return (1 + alpha*math.erf(x*math.sqrt(math.pi)/2) + (1-alpha)*(x / (1 + abs(x))))/2


def biggielevs_dphi(x: float) -> float:
    alpha = 0.42
    return (alpha*math.exp(-x*x*math.pi/4) + (1-alpha)*(1/(1+abs(x))**2))/2


def biggielevs_dlogphi(x: float) -> float:
    return biggielevs_dphi(x)/biggielevs_phi(x)


# Unit conversions-----------------------------------
"""
The conversion here is at the elo=150 mark (the biggie-levs and bradley-terry model agree on winrates at 0.52 <-> 150)
"""


def elo_to_r(elo: float) -> float:
    return elo*(0.52/150)


def r_to_elo(r: float) -> float:
    return r*(150/0.52)


# Matchup, Player classes----------------------------

class Matchup(object):
    def __init__(self):
        self.wins = 0
        self.losses = 0

    @property
    def games(self):
        return self.wins + self.losses


class Player(object):
    def __init__(self, name, prior_elo=0.0, prior_stdev=400.0):
        self._name = name
        self._prior_r = elo_to_r(prior_elo)
        self._prior_var = elo_to_r(prior_stdev)**2
        self._last_r = self._prior_r
        self._r = self._prior_r
        self._last_step_size = 10

        self._matchups = dict()  # type: Dict[Player, Matchup]
        self._total_wins = 0
        self._test_matchups = dict()
        self._total_test_wins = 0

        self._name_lower = name.lower()
        self._hash = hash(self._name)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return self._name_lower == other._name_lower

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def r(self):
        return self._r

    @property
    def name(self):
        return self._name

    @property
    def elo(self):
        return r_to_elo(self._r)

    @property
    def last_r(self):
        return self._last_r

    def log_likelihood(self, r):
        evidence_ll = 0
        for player, matchup in self._matchups.items():
            a = biggielevs_phi(r - player.r)
            evidence_ll += matchup.wins*math.log(a) + matchup.losses*math.log(1-a)
        prior_ll = -((r - self._prior_r)/(self._prior_var))**2 - math.log(math.sqrt(2*math.pi)*self._prior_var)
        return evidence_ll + prior_ll

    @property
    def grad_component(self):
        evidence_cmp = 0
        for player, matchup in self._matchups.items():
            a = biggielevs_dphi(self.r - player.r)
            b = biggielevs_phi(self.r - player.r)
            evidence_cmp += a*(matchup.wins - matchup.games*b)/(b-b**2)
        priors_cmp = (self._prior_r - self._r)/self._prior_var
        return evidence_cmp + priors_cmp

    @property
    def matchup_info(self) -> str:
        format_str = '{name}: [{wins} - {losses}] -- '
        matchup_format = 'vs {opp}: [{wins} - {losses}]; '
        total_wins = 0
        total_losses = 0
        for _, matchup in self._matchups.items():
            total_wins += matchup.wins
            total_losses += matchup.losses
        outstr = format_str.format(name=self.name, wins=total_wins, losses=total_losses)
        for player, matchup in self._matchups.items():
            outstr += matchup_format.format(opp=player.name, wins=matchup.wins, losses=matchup.losses)
        return outstr[:-2]

    def add_games(self, opponent, wins: int, losses: int):
        if opponent in self._matchups:
            matchup = self._matchups[opponent]
        else:
            matchup = Matchup()
            self._matchups[opponent] = matchup

        matchup.wins += wins
        self._total_wins += wins
        matchup.losses += losses

    def add_test_games(self, opponent, wins, losses):
        if opponent in self._test_matchups:
            matchup = self._test_matchups[opponent]
        else:
            matchup = Matchup()
            self._test_matchups[opponent] = matchup
        matchup.wins += wins
        self._total_test_wins += wins
        matchup.losses += losses

    def run_mm_iteration(self):
        self._last_r = self._r
        grad_cpt = self.grad_component
        if grad_cpt == 0:
            return

        # Get step size via backtracking line search w/ Armijo-Goldstein
        # c = 0.5
        # tau = 0.5
        # alpha = 10 # self._last_step_size
        # sign = 1 if grad_cpt > 0 else -1
        # t = c*grad_cpt*grad_cpt
        # current_ll = self.log_likelihood(self._last_r)
        # while self.log_likelihood(self._last_r + alpha*sign) < current_ll + alpha*t:
        #     alpha = alpha*tau
        #
        # self._r = self._last_r + alpha*sign
        # self._last_step_size = 2*alpha

        self._r += 0.01*grad_cpt


# Module methods---------------------------------

def iterate_elos(player_list, max_repeats=200000, max_sec=1200, verbose=False):
    format_str = 'Iteration: {iter:>7} Time: {sec:10.10f}s Elo L2: {elo:20.20f} Grad L2: {grad:20.20f}'
    begin = time.monotonic()

    l2_elo_diff = 0.0
    l2_grad = 0.0
    sec = begin

    with open(LOG_FILENAME, 'w') as iter_log:
        for idx in range(max_repeats):
            l2_r_diff = 0.0
            l2_grad = 0.0

            # run one iteration for each player
            for player in player_list:
                player.run_mm_iteration()

            for player in player_list:
                l2_r_diff += (player.r - player.last_r)**2
                l2_grad += player.grad_component**2

            l2_elo_diff = r_to_elo(math.sqrt(l2_r_diff))
            l2_grad = math.sqrt(l2_grad)
            sec = time.monotonic() - begin

            if verbose and (idx % 500 == 0):
                print(format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad))
                iter_log.write(format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad) + '\n')

            if l2_elo_diff < 0.0000000000001 and l2_grad < 0.0000000000003:
                iter_log.write('Finished in {iter} iterations.\n'.format(iter=idx))
                return
            elif sec > max_sec:
                iter_log.write(
                    'Timed out: '
                    + format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad)
                    + '\n'
                )
                return

        iter_log.write(
            'Completed maximum number of iterations: '
            + format_str.format(iter=max_repeats, sec=sec, elo=l2_elo_diff, grad=l2_grad)
            + '\n'
        )


def normalize_elos(player_list, avg_elo):
    list_tot = 0
    for player in player_list:
        list_tot += player.elo
    list_avg = list_tot / len(player_list)

    for player in player_list:
        player.add_to_elo(avg_elo - list_avg)


def write_csv(player_list, filename='results.csv'):
    _do_write(player_list, filename, '{name},{elo}')


def write_formatted(player_list, filename='results.txt'):
    _do_write(player_list, filename, '{name:>20}: {elo:4.0f}')


def _do_write(player_list, filename, format_str):
    with open(filename, 'w') as file:
        for player in player_list:
            file.write(format_str.format(name=player.name, elo=player.elo) + '\n')


def make_player_dict(
        prior_elos: List[Tuple[str, float]],
        gametuples: List[Tuple[str, str, int, int]],
        stdev: float,
        test_gametuples: List[Tuple[str, str, int, int]] = None
) -> Dict[str, Player]:
    if test_gametuples is None:
        test_gametuples = []
    
    player_dict = dict()  # type: Dict[str, Player]
    for player_name, elo in prior_elos:
        player_dict[player_name.lower()] = Player(name=player_name, prior_elo=elo, prior_stdev=stdev)

    for p1name, p2name, p1wins, p2wins in gametuples:
        if p1name not in player_dict:
            print('Error: {name} has no prior elo.'.format(name=p1name))
            player_dict[p1name] = Player(name=p1name, prior_elo=1500, prior_stdev=stdev)
            # continue
        if p2name not in player_dict:
            print('Error: {name} has no prior elo.'.format(name=p2name))
            player_dict[p2name] = Player(name=p2name, prior_elo=1500, prior_stdev=stdev)
            # continue

        p1 = player_dict[p1name]
        p2 = player_dict[p2name]
        p1.add_games(opponent=p2, wins=p1wins, losses=p2wins)
        p2.add_games(opponent=p1, wins=p2wins, losses=p1wins)

    for p1name, p2name, p1wins, p2wins in test_gametuples:
        p1 = player_dict[p1name]
        p2 = player_dict[p2name]
        p1.add_test_games(opponent=p2, wins=p1wins, losses=p2wins)
        p2.add_test_games(opponent=p1, wins=p2wins, losses=p1wins)

    return player_dict


def write_records(player_list: List[Player], filename: str):
    with open(filename, 'w') as outfile:
        for player in sorted(player_list, key=lambda player: player.name.lower()):
            outfile.write(player.matchup_info + '\n')


# Necrobot specifics----------------------------------

def get_elos(
        prior_filename: str,
        games_database_name: str,
        max_repeats: int = 200000,
        max_sec: int = 1200,
        verbose: bool = False
):
    prior_stdev = PRIOR_STDEV
    prior_elos = read_priors(prior_filename)
    gametuples = get_gametuples_from_database(games_database_name)
    for game in custom_matches:
        gametuples.append(game)
    # for game in gametuples:
    #     print(game)

    player_dict = make_player_dict(
        prior_elos=prior_elos,
        gametuples=gametuples,
        stdev=prior_stdev
    )
    # print(player_dict)

    iterate_elos(
        player_list=player_dict.values(),
        max_repeats=max_repeats,
        max_sec=max_sec,
        verbose=verbose
    )

    sorted_players = sorted(player_dict.values(), key=lambda p: p.r, reverse=True)
    write_csv(sorted_players, '{}.csv'.format(ELO_RESULTS_FILENAME))
    write_formatted(sorted_players, '{}.txt'.format(ELO_RESULTS_FILENAME))
    write_records(sorted_players, '{}.txt'.format(RECORDS_FILENAME))


def read_priors(filename: str) -> List[Tuple[str, float]]:
    prior_elos = list()  # type: List[Tuple[str, float]]
    with open(filename) as file:
        for line in file:
            args = line.split(',')
            prior_elos.append((args[0], float(args[1]),))
    return prior_elos


def get_gametuples_from_database(database_name):
    db_conn = mysql.connector.connect(
        user=mysql_db_user,
        password=mysql_db_passwd,
        host=mysql_db_host,
        database=database_name
    )

    try:
        cursor = db_conn.cursor()

        cursor.execute(
            """
            SELECT 
                udw.twitch_name AS winner_name,
                udl.twitch_name AS loser_name
            FROM 
                race_summary
            JOIN
                matches ON matches.match_id = race_summary.match_id
            JOIN
                necrobot.users udw ON udw.user_id = race_summary.winner_id
            JOIN
                necrobot.users udl ON udl.user_id = race_summary.loser_id
            WHERE
                matches.league_tag = '{}' AND matches.finish_time < '{}'
            ORDER BY matches.finish_time ASC
            """.format(LEAGUE_TAG, AUTOMATCH_DEADLINE)
        )

        matchup_tuples = defaultdict(lambda: [0, 0])
        for row in cursor:
            winner = row[0].lower()
            loser = row[1].lower()

            if winner in ignored_racers or loser in ignored_racers:
                print("Ignored game {r1}-{r2}".format(r1=winner, r2=loser))

            if winner < loser:
                match = matchup_tuples[(winner, loser)]
                if match[0] + match[1] < 3:
                    match[0] += 1
            else:
                match = matchup_tuples[(loser, winner)]
                if match[0] + match[1] < 3:
                    match[1] += 1

        return list((k[0], k[1], v[0], v[1]) for k, v in matchup_tuples.items())

        # gametuples = []
        # for row in cursor:
        #     racer_1 = row[0].lower()
        #     racer_2 = row[1].lower()
        #
        #     if racer_1 not in ignored_racers and racer_2 not in ignored_racers:
        #         gametuples.append((racer_1, racer_2, 1, 0,))
        #     else:
        #         print("Ignored game {r1}-{r2}".format(r1=racer_1, r2=racer_2))
        # return gametuples
    finally:
        db_conn.close()


# Main --------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        priors_filename = sys.argv[1]
    else:
        priors_filename = PRIOR_ELOS_FILENAME

    print('Getting Elo priors from file {}...'.format(priors_filename))
    get_elos(priors_filename, LEAGUE_DATABASE_NAME, verbose=True)
    print('Elos written to file {}.csv'.format(ELO_RESULTS_FILENAME))


# Tests -------------------------------------

class TestRatings(unittest.TestCase):
    def setUp(self):
        pass

    @unittest.skip
    def test_ratings_1(self):
        stdev = 140.0
        self.players = dict()   # type: Dict[str, Player]
        games, test_games = self.read_season5_db()
        self.players = make_player_dict(
            prior_elos=self.get_prior_elos(),
            gametuples=games,
            stdev=stdev,
            test_gametuples=test_games
        )

        iterate_elos(self.players.values(), max_repeats=2000, max_sec=12, verbose=False)
        normalize_elos(self.players.values(), 0)
        sorted_players = sorted(self.players.values(), key=lambda p: p.gamma, reverse=True)

        write_csv(player_list=sorted_players)
        write_formatted(player_list=sorted_players)

    def test_ratings(self):
        get_elos(PRIOR_ELOS_FILENAME, 'season_6', max_repeats=1000, max_sec=3, verbose=False)

    @unittest.skip
    def test_stdev(self):
        steps = 10
        min_stdev = 100
        max_stdev = 2000
        format_str = 'Stdev: {stdev:6.6f} -- LL: {ll:20.20f}\n'
        games, test_games = self.read_season5_db()
        prior_elos = self.get_prior_elos()
        with open('inc_stdev_test.txt', 'w') as outfile:
            for i in range(steps):
                lm = float(i)/float(steps-1)
                test_stdev = math.exp(lm*math.log(max_stdev) + (1.0 - lm)*math.log(min_stdev))
                ll = self.get_ll_for_stdev(test_stdev, prior_elos, games, test_games)
                outfile.write(format_str.format(stdev=test_stdev, ll=ll))

    def get_ll_for_stdev(self, stdev, prior_elos, games, test_games):
        self.players = make_player_dict(
            prior_elos=prior_elos,
            gametuples=games,
            stdev=stdev,
            test_gametuples=test_games
        )

        iterate_elos(self.players.values())
        log_likelihood = 0

        for player in self.players.values():
            log_likelihood += player.test_log_likelihood
        return log_likelihood

    def read_season5_db(self) -> Tuple[List[Tuple[str, str, int, int]], List[Tuple[str, str, int, int]]]:
        class CondorMatch(object):
            def __init__(self, match_week, player_1, player_2, p1_wins, p2_wins):
                self.racer_1 = player_1
                self.racer_2 = player_2
                self.week = match_week
                self.races = []
                self.r1_wins = p1_wins
                self.r2_wins = p2_wins
                while p1_wins > 0:
                    self.races.append(CondorRace(player_1, player_2))
                    p1_wins -= 1
                while p2_wins > 0:
                    self.races.append(CondorRace(player_2, player_1))
                    p2_wins -= 1

        class CondorRace(object):
            def __init__(self, winner, loser):
                self.winner = winner.lower()
                self.loser = loser.lower()

        showcase_matches = [
            # Week 1 Fri
            CondorMatch(1, 'asherrose', 'the_crystal_clod', 2, 0),
            CondorMatch(1, 'ARTQ', 'yuka', 2, 1),
            CondorMatch(1, 'Echaen', 'DisgruntledGoof', 3, 0),
            CondorMatch(1, 'squega', 'moyuma', 2, 0),
            CondorMatch(1, 'JackOfGames', 'paperdoopliss', 2, 0),
            CondorMatch(1, 'incnone', 'mayantics', 2, 1),
            # Week 1 Sat
            CondorMatch(1, 'Fraxtil', 'mantasMBL', 2, 0),
            CondorMatch(1, 'naymin', 'revalize', 1, 2),
            CondorMatch(1, 'staekk', 'spootybiscuit', 1, 2),
            CondorMatch(1, 'mudjoe2', 'oblivion111', 2, 0),
            CondorMatch(1, 'spootybiscuit', 'incnone', 2, 1),
            CondorMatch(1, 'jackofgames', 'mudjoe2', 2, 0),
            # Week 2 Fri
            CondorMatch(2, 'axem', 'yuuchan', 2, 0),
            CondorMatch(2, 'bacing', 'ARTQ', 2, 0),
            CondorMatch(2, 'yuka', 'hypershock', 2, 0),
            CondorMatch(2, 'pillarmonkey', 'cyber_1', 1, 2),
            CondorMatch(2, 'spacecow2455', 'amak11', 2, 0),
            CondorMatch(2, 'paperdoopliss', 'squega', 2, 0),
            CondorMatch(2, 'staekk', 'mayantics', 1, 2),
            CondorMatch(2, 'incnone', 'paperdoopliss', 2, 0),
            # Week 2 Sat
            CondorMatch(2, 'Echaen', 'mudjoe2', 1, 2),
            CondorMatch(2, 'mayantics', 'mudjoe2', 1, 2),
            CondorMatch(2, 'naymin', 'CS1', 2, 0),
            CondorMatch(2, 'sirwuffles', 'yjalexis', 1, 2),
            CondorMatch(2, 'wonderj13', 'kingcaptain27', 2, 1),
            CondorMatch(2, 'tufwfo', 'moyuma', 1, 2),
            # Week 3 Fri
            CondorMatch(3, 'Squega', 'Echaen', 2, 1),
            CondorMatch(3, 'Midna', 'CheesiestPotato', 1, 2),
            CondorMatch(3, 'tetel', 'zellyff', 2, 0),
            CondorMatch(3, 'thalen', 'boredmai', 2, 1),
            CondorMatch(3, 'oblivion111', 'moyuma', 2, 0),
            CondorMatch(3, 'roncli', 'revalize', 2, 1),
            CondorMatch(3, 'staekk', 'paperdoopliss', 2, 0),
            # Week 3 Sat
            CondorMatch(3, 'pillarmonkey', 'fraxtil', 1, 2),
            CondorMatch(3, 'kingcaptain27', 'muffin', 2, 0),
            CondorMatch(3, 'spacecow2455', 'sponskapatrick', 1, 2),
            CondorMatch(3, 'tictacfoe', 'ARTQ', 2, 0),
            CondorMatch(3, 'slackaholicus', 'mantasmbl', 2, 0),
            CondorMatch(3, 'mayantics', 'cyber_1', 2, 0),
            CondorMatch(3, 'heather', 'pibonacci', 1, 2),
            CondorMatch(3, 'oblivion111', 'squega', 2, 0),
            CondorMatch(3, 'staekk', 'mayantics', 1, 2),
            # Week 4 Fri
            CondorMatch(4, 'greenyoshi', 'odoko_noko', 2, 0),
            CondorMatch(4, 'skullgirls', 'boredmai', 1, 2),
            CondorMatch(4, 'tufwfo', 'paratroopa1', 2, 1),
            CondorMatch(4, 'kingtorture', 'medvezhonok', 1, 2),
            CondorMatch(4, 'staekk', 'moyuma', 2, 0),
            CondorMatch(4, 'abuyazan', 'disgruntledgoof', 1, 2),
            CondorMatch(4, 'squega', 'roncli', 2, 0),
            CondorMatch(4, 'squega', 'staekk', 1, 2),
            # Week 4 Sat
            CondorMatch(4, 'tictacfoe', 'naymin', 1, 2),
            CondorMatch(4, 'yuuchan', 'teraka', 1, 2),
            CondorMatch(4, 'hypershock', 'plectro', 2, 1),
            CondorMatch(4, 'gunlovers', 'raviolinguini', 2, 0),
            CondorMatch(4, 'sirwuffles', 'asherrose', 2, 0),
            CondorMatch(4, 'pibonacci', 'pancelor', 1, 2),
            CondorMatch(4, 'artq', 'thouther', 2, 1),
            CondorMatch(4, 'echaen', 'paperdoopliss', 2, 1),
            CondorMatch(4, 'echaen', 'cyber_1', 2, 0),
            # Week 5 Fri
            CondorMatch(5, 'flygluffet', 'gfitty', 2, 1),
            CondorMatch(5, 'arboretic', 'raviolinguini', 1, 2),
            CondorMatch(5, 'disgruntledgoof', 'fraxtil', 2, 0),
            CondorMatch(5, 'pancelor', 'squega', 2, 1),
            CondorMatch(5, 'wow_tomato', 'madoka', 0, 2),
            CondorMatch(5, 'disgruntledgoof', 'pancelor', 2, 1),
            CondorMatch(5, 'kika', 'the_crystal_clod', 0, 2),
            CondorMatch(5, 'gunlovers', 'medvezhonok', 2, 0),
            # Week 5 Sat
            CondorMatch(5, 'abuyazan', 'thedarkfreaack', 0, 2),
            CondorMatch(5, 'grimy42', 'thouther', 2, 0),
            CondorMatch(5, 'plectro', 'amak11', 2, 1),
            CondorMatch(5, 'naymin', 'moyuma', 2, 1),
            CondorMatch(5, 'cyber_1', 'tufwfo', 2, 0),
            CondorMatch(5, 'naymin', 'cyber_1', 0, 2),
            CondorMatch(5, 'yuuchan', 'famslayer', 0, 2),
            CondorMatch(5, 'tictacfoe', 'necrorebel', 1, 2),
            CondorMatch(5, 'teraka', 'yuka', 0, 2),
            CondorMatch(5, 'yjalexis', 'revalize', 1, 2),
            # Play-in R1
            CondorMatch(6, 'moyuma', 'medvezhonok', 2, 0),
            CondorMatch(6, 'revalize', 'abuyazan', 2, 0),
            CondorMatch(6, 'pancelor', 'thedarkfreaack', 0, 2),
            CondorMatch(6, 'tufwfo', 'roncli', 2, 1),
            CondorMatch(6, 'tictacfoe', 'yjalexis', 1, 2),
            CondorMatch(6, 'fraxtil', 'necrorebel', 0, 2),
            CondorMatch(6, 'squega', 'gunlovers', 2, 0),
            CondorMatch(6, 'naymin', 'pibonacci', 2, 0),
            # Play-in R2
            CondorMatch(6, 'thedarkfreaack', 'tufwfo', 0, 2),
            CondorMatch(6, 'yjalexis', 'necrorebel', 2, 0),
            CondorMatch(6, 'moyuma', 'revalize', 0, 2),
            CondorMatch(6, 'squega', 'naymin', 0, 2),
            # Play-in R3
            CondorMatch(6, 'revalize', 'tufwfo', 3, 2),
            CondorMatch(6, 'yjalexis', 'naymin', 2, 3),
            # Playoff Day 1
            CondorMatch(7, 'cyber_1', 'echaen', 3, 2),
            CondorMatch(7, 'naymin', 'mayantics', 3, 2),
            CondorMatch(7, 'staekk', 'disgruntledgoof', 3, 2),
            CondorMatch(7, 'oblivion111', 'revalize', 2, 3),
            CondorMatch(7, 'mudjoe2', 'staekk', 3, 1),
            CondorMatch(7, 'staekk', 'oblivion111', 0, 2),
            # Playoff Day 2
            CondorMatch(7, 'jackofgames', 'naymin', 1, 3),
            CondorMatch(7, 'incnone', 'revalize', 3, 0),
            CondorMatch(7, 'echaen', 'jackofgames', 1, 2),
            CondorMatch(7, 'spootybiscuit', 'cyber_1', 3, 1),
            CondorMatch(7, 'oblivion111', 'echaen', 2, 0),
            CondorMatch(7, 'mayantics', 'cyber_1', 0, 2),
            CondorMatch(7, 'disgruntledgoof', 'revalize', 2, 0),
            CondorMatch(7, 'spootybiscuit', 'incnone', 0, 3),
            CondorMatch(7, 'cyber_1', 'disgruntledgoof', 2, 1),
            # Playoff Day 3
            CondorMatch(7, 'mudjoe2', 'naymin', 3, 2),
            CondorMatch(7, 'naymin', 'cyber_1', 0, 2),
            CondorMatch(7, 'incnone', 'mudjoe2', 1, 3),
            CondorMatch(7, 'spootybiscuit', 'oblivion111', 2, 0),
            CondorMatch(7, 'spootybiscuit', 'cyber_1', 2, 0),
            CondorMatch(7, 'incnone', 'spootybiscuit', 3, 2),
            CondorMatch(7, 'incnone', 'mudjoe2', 3, 1),
            CondorMatch(7, 'incnone', 'mudjoe2', 3, 0),
        ]

        db_conn = mysql.connector.connect(
            user=mysql_db_user,
            password=mysql_db_passwd,
            host=mysql_db_host,
            database='condor_s5'
        )

        try:
            cursor = db_conn.cursor()

            cursor.execute(
                """
                SELECT 
                    ud1.rtmp_name AS racer_1,
                    ud2.rtmp_name AS racer_2,
                    week_number,
                    racer_1_wins,
                    racer_2_wins
                FROM 
                    match_data
                JOIN
                    user_data ud1 ON ud1.racer_id = match_data.racer_1_id
                JOIN
                    user_data ud2 ON ud2.racer_id = match_data.racer_2_id
                WHERE
                    week_number BETWEEN 1 AND 5
                """
            )

            test_gametuples = []
            gametuples = []

            for row in cursor:
                racer_1 = row[0].lower()
                racer_2 = row[1].lower()
                racer_1_wins = int(row[3])
                racer_2_wins = int(row[4])

                gametuples.append((racer_1, racer_2, racer_1_wins, racer_2_wins,))
        finally:
            db_conn.close()

        for match in showcase_matches:
            # if match.racer_1.lower() in playoffs_players and match.racer_2.lower() in playoffs_players:
            #     test_gametuples.append(
            #         (match.racer_1.lower(), match.racer_2.lower(), match.r1_wins, match.r2_wins,)
            #     )
            test_gametuples.append(
                (match.racer_1.lower(), match.racer_2.lower(), match.r1_wins, match.r2_wins,)
            )

        return gametuples, test_gametuples

    @staticmethod
    def get_prior_elos() -> List[Tuple[str, int]]:
        elo_str = """
       1 spootybiscuit        678   91   91    49   65%   565    0% 
       2 incnone              655   84   84    60   63%   552    0% 
       3 mudjoe2              620   82   82    60   60%   545    0% 
       4 jackofgames          608  103  103    38   61%   531    0% 
       5 oblivion111          549   91   91    47   53%   526    0% 
       6 cyber_1              490   85   85    59   61%   397    0% 
       7 mayantics            464   89   89    51   39%   545    0% 
       8 naymin               456   82   82    68   65%   328    0% 
       9 staekk               452   84   84    57   44%   501    0% 
      10 paperdoopliss        445  104  104    35   46%   476    0% 
      11 revalize             416   92   92    52   62%   325    0% 
      12 pillarmonkey         406  129  129    24   63%   310    0% 
      13 tufwfo               404   91   91    48   58%   340    0% 
      14 disgruntledgoof      400   91   91    51   61%   314    0% 
      15 echaen               399   89   89    54   43%   455    0% 
      16 squega               390   86   86    51   47%   413    0% 
      17 moyuma               373   92   92    46   50%   370    0% 
      18 fraxtil              368  119  119    27   59%   302    0% 
      19 pancelor             367   98   98    41   59%   305    0% 
      20 tictacfoe            335  126  126    29   72%   160    0% 
      21 necrorebel           315  103  103    37   54%   283    0% 
      22 thedarkfreaack       303  102  102    36   50%   305    0% 
      23 roncli               301  105  105    38   47%   311    0% 
      24 yjalexis             299   95   95    46   52%   274    0% 
      25 paratroopa1          286  108  108    33   45%   320    0% 
      26 abuyazan             262  102  102    37   46%   293    0% 
      27 gunlovers            206  106  106    36   67%    90    0% 
      28 pibonacci            204  142  142    20   60%   137    0% 
      29 invertttt            197  143  143    18   44%   234    0% 
      30 kingtorture          159  108  108    33   58%   112    0% 
      31 slackaholicus        141  113  113    32   50%   145    0% 
      32 bacing               133  127  127    26   42%   185    0% 
      33 grimy42              125  116  116    32   63%    36    0% 
      34 sponskapatrick       124  142  142    21   43%   180    0% 
      35 flygluffet           118  119  119    27   63%    29    0% 
      36 artq                 117  104  104    40   65%     2    0% 
      37 thouther             112  140  140    17   59%    56    0% 
      38 bastet                93  135  135    21   57%    40    0% 
      39 mantasmbl             91  113  113    34   50%    88    0% 
      40 heather               91  128  128    21   52%    76    0% 
      41 progus91              87  113  113    30   50%    91    0% 
      42 sirwuffles            72  107  107    35   54%    39    0% 
      43 gfitty                66  108  108    33   55%    40    0% 
      44 arboretic             65  111  111    33   58%     8    0% 
      45 hordeoftribbles       57  114  114    30   47%    79    0% 
      46 ratata                57  129  129    30   23%   272    0% 
      47 medvezhonok           55  110  110    37   41%   129    0% 
      48 madhyena              22  142  142    24   17%   268    0% 
      49 cs1                   22  110  110    32   47%    48    0% 
      50 yuka                  -9  115  115    37   68%  -149    0% 
      51 emuemu               -18  128  128    24   42%    36    0% 
      52 raviolinguini        -27  104  104    35   46%     1    0% 
      53 spacecow2455         -28  122  122    35   66%  -171    0% 
      54 sailormint           -36  164  164    12   33%    62    0% 
      55 tetel                -45  161  161    20   80%  -271    0% 
      56 teraka               -51  145  145    23   78%  -290    0% 
      57 wonderj13            -58  119  119    27   48%   -47    0% 
      58 flamehaze0           -95  260  260    12    0%   338    0% 
      59 asherrose           -112  116  116    34   47%   -85    0% 
      60 crazyeightsfan69    -127  116  116    30   40%   -53    0% 
      61 axem                -134  114  114    32   50%  -134    0% 
      62 kingcaptain27       -159  119  119    29   66%  -287    0% 
      63 saakas0206          -172  124  124    30   23%    25    0% 
      64 missingno           -183  134  134    30   20%    62    0% 
      65 cheesiestpotato     -188  114  114    33   55%  -223    0% 
      66 boredmai            -236  108  108    36   58%  -296    0% 
      67 ekimekim            -264  132  132    30   20%    -9    0% 
      68 plectro             -275  142  142    18   67%  -388    0% 
      69 kika                -286  110  110    32   53%  -306    0% 
      70 famslayer           -288  122  122    26   54%  -324    0% 
      71 thalen              -292  125  125    27   37%  -192    0% 
      72 hypershock          -296  121  121    29   45%  -248    0% 
      73 yuuchan             -298  115  115    31   61%  -381    0% 
      74 skullgirls          -300  117  117    27   56%  -336    0% 
      75 madoka              -321  131  131    20   55%  -354    0% 
      76 muffin              -329  105  105    32   56%  -368    0% 
      77 zellyff             -332  132  132    26   38%  -243    0% 
      78 greenyoshi          -345  110  110    32   41%  -273    0% 
      79 midna               -347  108  108    33   39%  -272    0% 
      80 odoko_noko          -380  118  118    32   50%  -387    0% 
      81 the_crystal_clod    -385  113  113    34   50%  -386    0% 
      82 amak11              -397  114  114    32   34%  -272    0% 
      83 gauche              -415  122  122    30   40%  -341    0% 
      84 sillypears          -416  116  116    30   47%  -390    0% 
      85 wow_tomato          -445  111  111    32   56%  -494    0% 
      86 definitely_not_him  -462  129  129    24   33%  -341    0% 
      87 zetto               -501  281  281     6    0%  -169    0% 
      88 gemmi               -519  169  169    12   33%  -416    0% 
      89 paperlaur           -526  121  121    30   33%  -405    0% 
      90 uselessgamer        -575  228  228     6   17%  -400    0% 
      91 lismati             -576  139  139    24   21%  -347    0% 
      92 janegland           -632  265  265     6    0%  -346    0% 
      93 cyberman            -742  203  203    24    0%  -292    0% 
      94 tome123             -833  218  218    18    0%  -428    0% 
        """

        the_elos = list()  # type: List[Tuple[str, int]]
        for line in elo_str.split('\n'):
            args = line.split()
            if args:
                # straightforward way
                the_elos.append((args[1], int(args[2]),))

                # weird clamping
                # raw_elo = int(args[2])
                # if 600 <= raw_elo:
                #     raw_elo = 600
                # elif 300 <= raw_elo < 600:
                #     raw_elo = 300
                # elif 0 <= raw_elo < 300:
                #     raw_elo = 0
                # elif -300 <= raw_elo < 0:
                #     raw_elo = -300
                # elif raw_elo < -200:
                #     raw_elo = -600
                #
                # the_elos[args[1]] = raw_elo

                # just 0
                # the_elos[args[1]] = 0.0

        return the_elos


if __name__ == "__main__":
    pass
