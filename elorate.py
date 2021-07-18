import math
from typing import Dict, List, Tuple
import mysql.connector
import unittest
import time
import sys
from collections import defaultdict

# Modify these parameters per call
LEAGUE = 'coh'
SEASON = 'sxii'
WEEK = 1
PRIOR_STDEV_BY_WEEK = {
    1: 300.0,
    2: 450.0,
    3: 600.0,
    4: 600.0,
}

AUTOMATCH_DEADLINE_BY_WEEK = {
    1: '2021-07-18 18:00:00',
    2: '2021-07-25 18:00:00',
    3: '2021-07-01 18:00:00',
    4: '2021-07-08 09:00:00',
}

OVERTURN_RESULTS = {
}

PRIOR_STDEV = PRIOR_STDEV_BY_WEEK[WEEK]
AUTOMATCH_DEADLINE = AUTOMATCH_DEADLINE_BY_WEEK[WEEK]

mysql_db_host = 'condor.live'
mysql_db_user = 'necrobot-read'
mysql_db_passwd = 'necrobot-read'
mysql_db_name = 'condor_xii'

# Don't modify these
FOLDER = 'data_{league}'.format(league=LEAGUE)
PRIOR_ELOS_FILENAME = '{f}/ratings_{s}_{lg}_wk{w}.csv'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=0)
ELO_RESULTS_FILENAME = '{f}/ratings_{s}_{lg}_wk{w}'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=WEEK)
RECORDS_FILENAME = '{f}/records_{s}_{lg}_wk{w}'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=WEEK)
LOG_FILENAME = '{f}/elorate_log.txt'.format(f=FOLDER)

# Don't use these racers in the computation
ignored_racers = [
]

# Add these extra matches into the computation
custom_matches = [
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
Linear conversion at the elo=150 mark (the biggie-levs and bradley-terry model agree on winrates at 0.52 <-> 150)
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
        self._next_r = self._prior_r
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
        prior_ll = -((r - self._prior_r)/self._prior_var)**2 - math.log(math.sqrt(2*math.pi)*self._prior_var)
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

    def prep_mm_iteration(self):
        self._next_r = self._r + 0.01*self.grad_component

    def finalize_mm_iteration(self):
        self._last_r = self._r
        self._r = self._next_r


# Module methods---------------------------------

def iterate_elos(player_list, max_repeats=200000, max_sec=1200, verbose=False):
    format_str = 'Iteration: {iter:>7} Time: {sec:10.10f}s Elo L2: {elo:20.20f} Grad L2: {grad:20.20f} LL: {ll:20.20f}'
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
                player.prep_mm_iteration()

            for player in player_list:
                player.finalize_mm_iteration()
                l2_r_diff += (player.r - player.last_r)**2
                l2_grad += player.grad_component**2

            l2_elo_diff = r_to_elo(math.sqrt(l2_r_diff))
            l2_grad = math.sqrt(l2_grad)
            sec = time.monotonic() - begin

            if verbose and (idx % 500 == 0):
                log_prob = 0
                for player in player_list:
                    log_prob += player.log_likelihood(player.r)
                print(format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad, ll=log_prob))
                iter_log.write(format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad, ll=log_prob) + '\n')

            if l2_elo_diff < 0.000000000000001 or l2_grad < 0.00000000000001:
                iter_log.write('Finished in {iter} iterations.\n'.format(iter=idx))
                return
            elif sec > max_sec:
                log_prob = 0
                for player in player_list:
                    log_prob += player.log_likelihood(player.r)
                iter_log.write(
                    'Timed out: '
                    + format_str.format(iter=idx, sec=sec, elo=l2_elo_diff, grad=l2_grad, ll=log_prob)
                    + '\n'
                )
                return

        log_prob = 0
        for player in player_list:
            log_prob += player.log_likelihood(player.r)
        iter_log.write(
            'Completed maximum number of iterations: '
            + format_str.format(iter=max_repeats, sec=sec, elo=l2_elo_diff, grad=l2_grad, ll=log_prob)
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
        for player in sorted(player_list, key=lambda p: p.name.lower()):
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

    total_games = 0
    for p1, p2, w1, w2 in gametuples:
        total_games += w1 + w2
    print("Total number of games counted: {}".format(total_games))

    player_dict = make_player_dict(
        prior_elos=prior_elos,
        gametuples=gametuples,
        stdev=prior_stdev
    )

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
            """.format(LEAGUE, AUTOMATCH_DEADLINE)
        )

        matchup_tuples = defaultdict(lambda: [0, 0])
        for row in cursor:
            winner = row[0].lower()
            loser = row[1].lower()

            if winner in ignored_racers or loser in ignored_racers:
                print("Ignored game {r1}-{r2}".format(r1=winner, r2=loser))

            if (loser, winner) in OVERTURN_RESULTS:
                OVERTURN_RESULTS.remove((loser, winner))
                winner, loser = loser, winner
                print("Overturned game {r1}-{r2}".format(r1=loser, r2=winner))

            if winner < loser:
                match = matchup_tuples[(winner, loser)]
                match[0] += 1
            else:
                match = matchup_tuples[(loser, winner)]
                match[1] += 1

        return list((k[0], k[1], v[0], v[1]) for k, v in matchup_tuples.items())

    finally:
        db_conn.close()


# Main --------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        priors_filename = sys.argv[1]
    else:
        priors_filename = PRIOR_ELOS_FILENAME

    print('Getting Elo priors from file {}...'.format(priors_filename))
    get_elos(priors_filename, mysql_db_name, verbose=True)
    print('Elos written to file {}.csv'.format(ELO_RESULTS_FILENAME))


# Tests -------------------------------------

class TestRatings(unittest.TestCase):
    def setUp(self):
        pass

    def test_ratings(self):
        # TODO
        pass
