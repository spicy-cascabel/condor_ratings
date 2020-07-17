import math
import pulp
import random
import unittest
from typing import Dict, List, Set, Tuple
import mysql.connector
import sys


mysql_db_host = 'condor.live'
mysql_db_user = 'necrobot-read'
mysql_db_passwd = 'necrobot-read'
mysql_db_name = 'condor_x'

LEAGUE = 'noc'
SEASON = 'sx'
WEEK = 3    # Matchups for week AFTER this week
NUM_AUTOGENS = 2 if LEAGUE == 'cad' else 1
MINIMUM_CYCLE_SIZE = 7 if LEAGUE == 'cad' else 1

if LEAGUE == 'coh':
    SPECIAL_NUM_AUTOGENS = {'d_tea': 2}
elif LEAGUE == 'noc':
    SPECIAL_NUM_AUTOGENS = {'abu__yazan': 2}
else:
    SPECIAL_NUM_AUTOGENS = {}

FOLDER = 'data_{league}'.format(league=LEAGUE)
RIDER = '{s}_{lg}_wk{w}'.format(s=SEASON, lg=LEAGUE, w=WEEK)

INPUT_FILENAME = '{f}/ratings_{s}_{lg}_wk{w}.csv'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=WEEK)
MATCHUP_FILENAME = '{f}/matchups_{s}_{lg}_wk{w}.csv'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=WEEK+1)
MATCHUP_SUMMARY_FILENAME = '{f}/matchcycles_{s}_{lg}_wk{w}.txt'.format(f=FOLDER, s=SEASON, lg=LEAGUE, w=WEEK+1)
BANNED_MACHUPS_FILENAME = '{f}/bannedmatches_{s}.txt'.format(f=FOLDER, s=SEASON)
DROPPED_RACERS_FILENAME = '{f}/drops_{lg}.txt'.format(f=FOLDER, lg=LEAGUE)

rand = random.Random()
rand.seed()


class Matchup(object):
    def __init__(self, player_1: str, player_2: str):
        player_1 = player_1.lower()
        player_2 = player_2.lower()
        self.player_1 = player_1 if player_1 < player_2 else player_2
        self.player_2 = player_2 if player_1 < player_2 else player_1

    def __hash__(self):
        return hash((self.player_1, self.player_2,))

    def __eq__(self, other):
        return self.player_1 == other.player_1 and self.player_2 == other.player_2

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Matchup {} - {}'.format(self.player_1, self.player_2)


def get_entropy(p1_elo: float, p2_elo: float) -> float:
    prob_p1_wins = 1 / (1 + pow(10, (p2_elo - p1_elo) / 400))
    return -(prob_p1_wins * math.log2(prob_p1_wins) + (1 - prob_p1_wins) * math.log2(1 - prob_p1_wins))


def get_utility(p1_elo: float, p2_elo: float) -> float:
    return math.sqrt(get_entropy(p1_elo, p2_elo))


def get_matchups(
        elos: Dict[str, float],
        banned_matches: Set[Matchup],
        num_matches: int = NUM_AUTOGENS
) -> Set[Matchup]:
    ilp_prob = pulp.LpProblem('matchup_problem', pulp.LpMaximize)

    # Make variables
    all_players = list(elos.keys())
    matchups = dict()   # type: Dict[str, Dict[str, pulp.LpVariable]]
    for p_idx in range(len(all_players)):
        p1_name = all_players[p_idx]
        matchups[p1_name] = dict()
        for q_idx in range(p_idx + 1, len(all_players)):
            p2_name = all_players[q_idx]
            if Matchup(p1_name, p2_name) not in banned_matches:
                matchups[p1_name][p2_name] = pulp.LpVariable(
                    "matchup_{0}_{1}".format(p1_name, p2_name), 0, 1, pulp.LpInteger
                )

    # Add entropy value of each matchup
    matchup_utility = dict()
    for player in matchups:
        for opp in matchups[player]:
            matchup_utility[matchups[player][opp]] = get_utility(elos[player], elos[opp])

    # Set optimization objective
    ilp_prob.setObjective(pulp.LpAffineExpression(matchup_utility, name="matchup_utility"))

    # Make constraints
    for player in matchups:
        edges_from_player = [matchups[player][opp] for opp in matchups[player]]
        for otherplayer in matchups:
            if player in matchups[otherplayer]:
                edges_from_player.append(matchups[otherplayer][player])

        if player in SPECIAL_NUM_AUTOGENS:
            ilp_prob += pulp.lpSum(edges_from_player) == SPECIAL_NUM_AUTOGENS[player], ""
        else:
            ilp_prob += pulp.lpSum(edges_from_player) == num_matches, ""

    # Solve problem
    ilp_prob.solve(pulp.PULP_CBC_CMD(maxSeconds=20, msg=0, fracGap=0.001))
    print("Status:", pulp.LpStatus[ilp_prob.status])
    created_matches = set()
    for player in matchups:
        for opp in matchups[player]:
            if pulp.value(matchups[player][opp]) == 1:
                created_matches.add(Matchup(player, opp,))
    return created_matches


def get_matches_by_player(matches: Set[Matchup]) -> Dict[str, List[str]]:
    matches_by_player = dict()  # type: Dict[str, List[str]]
    for match in matches:
        if match.player_1 not in matches_by_player:
            matches_by_player[match.player_1] = list()
        if match.player_2 not in matches_by_player:
            matches_by_player[match.player_2] = list()
        matches_by_player[match.player_1].append(match.player_2)
        matches_by_player[match.player_2].append(match.player_1)
    return matches_by_player


def get_matchup_cycles(matches_by_player: Dict[str, List[str]]) -> List[List[str]]:
    matchup_cycles = list()
    checked_players = set()
    for player_name in matches_by_player.keys():
        assert player_name == player_name.lower()
        if player_name in checked_players:
            continue
        checked_players.add(player_name)

        new_cycle = [player_name]
        while True:
            wrote_any = False
            for opp in matches_by_player[player_name]:
                if opp not in checked_players:
                    new_cycle.append(opp)
                    checked_players.add(opp)
                    wrote_any = True
                    player_name = opp
                    break
            if not wrote_any:
                break
        matchup_cycles.append(new_cycle)
    return matchup_cycles


def enforce_min_cycle_size(
        matches_by_player: Dict[str, List[str]],
        elos: Dict[str, float],
        banned_matches: Set[Matchup],
        min_cycle_size: int
) -> Set[Matchup]:
    """This function assumes each player has exactly two matchups, otherwise it will break!"""
    for matchup_list in matches_by_player.values():
        assert len(matchup_list) == 2

    matchup_cycles = get_matchup_cycles(matches_by_player)  # type: List[List[str]]

    def exchange_cost(match_1: Tuple[str, str], match_2: Tuple[str, str]) -> float:
        return get_utility(elos[match_1[0]], elos[match_1[1]]) + get_utility(elos[match_2[0]], elos[match_2[1]]) \
                - get_utility(elos[match_1[0]], elos[match_2[0]]) - get_utility(elos[match_1[1]], elos[match_2[1]])

    while True:
        # Find the smallest cycle
        min_cycle = min(matchup_cycles, key=lambda l: len(l))

        # If the minimum cycle is long enough, we are done
        if len(min_cycle) >= min_cycle_size:
            break

        # Find the minimum cost exchange of something in this cycle and something in another cycle
        min_exchange_cost = float('inf')
        min_exchange_pair = None
        cycle_with_other_exchange = None

        pairs_in_min_cycle = [(min_cycle[0], min_cycle[-1],), (min_cycle[-1], min_cycle[0],)]
        for p1, p2 in zip(min_cycle, min_cycle[1:]):
            pairs_in_min_cycle.append((p1, p2,))
            pairs_in_min_cycle.append((p2, p1,))

        for cycle in matchup_cycles:
            if cycle is min_cycle:
                continue

            for p1, p2 in pairs_in_min_cycle:
                for q1, q2 in zip(cycle, cycle[1:] + [cycle[0]]):
                    if Matchup(p1, q1) not in banned_matches and Matchup(p2, q2) not in banned_matches:
                        exch_cost = exchange_cost((p1, p2), (q1, q2))
                        if exch_cost < min_exchange_cost:
                            min_exchange_cost = exch_cost
                            min_exchange_pair = [(p1, p2), (q1, q2)]
                            cycle_with_other_exchange = cycle

        if min_exchange_pair is None:
            print("Couldn't break a small cycle.")
            break

        # Exchange p1 and q1
        matchup_cycles.remove(min_cycle)
        matchup_cycles.remove(cycle_with_other_exchange)
        p1, p2 = min_exchange_pair[0]
        q1, q2 = min_exchange_pair[1]
        p1_idx = min_cycle.index(p1)
        p2_idx = min_cycle.index(p2)
        step_size = -1 if p1_idx < p2_idx else 1

        q2_idx = cycle_with_other_exchange.index(q2)
        matchup_cycles.append(
            cycle_with_other_exchange[:q2_idx] + min_cycle[p1_idx::step_size]
            + min_cycle[:p1_idx:step_size] + cycle_with_other_exchange[q2_idx:]
        )

    # Create the matchup set
    matchup_set = set()
    for cycle in matchup_cycles:
        if len(cycle) < 2:
            continue
        matchup_set.add(Matchup(cycle[0], cycle[-1]))
        for p1, p2 in zip(cycle, cycle[1:]):
            matchup_set.add(Matchup(p1, p2))
    return matchup_set


def write_matchup_csv_from_elo_csv(csv_filename: str, matchup_filename: str, summary_filename: str, drops_filename: str):
    the_elos = read_elos_from_csv(csv_filename)
    the_elos_dict = dict()
    name_cap_dict = dict()

    for player_name, _ in the_elos:
        name_cap_dict[player_name.lower()] = player_name

    for player_name, player_elo in the_elos:
        the_elos_dict[player_name.lower()] = player_elo

    banned_matchups = get_banned_matchups()

    with open(drops_filename, 'r') as drops_file:
        for line in drops_file:
            name = line.rstrip('\n').lower()
            if name in the_elos_dict:
                del the_elos_dict[name]

    matches = get_matchups(elos=the_elos_dict, banned_matches=banned_matchups, num_matches=NUM_AUTOGENS)

    if MINIMUM_CYCLE_SIZE >= 3 and NUM_AUTOGENS == 2:
        matches = enforce_min_cycle_size(
            matches_by_player=get_matches_by_player(matches),
            elos=the_elos_dict,
            banned_matches=banned_matchups,
            min_cycle_size=MINIMUM_CYCLE_SIZE
        )

    matches_by_player = get_matches_by_player(matches)   # type: Dict[str, List[str]]

    with open(matchup_filename, 'w') as outfile:
        format_str = '{player_1},{player_2}\n'
        for match in matches:
            outfile.write(
                format_str.format(
                    player_1=name_cap_dict[match.player_1],
                    player_2=name_cap_dict[match.player_2]
                )
            )

    matchup_rating_diffs = dict()
    for matchup in matches:
        matchup_rating_diffs[matchup] = abs(the_elos_dict[matchup.player_1] - the_elos_dict[matchup.player_2])
    matchup_rating_diffs = sorted(matchup_rating_diffs.items(), key=lambda p: p[1], reverse=True)

    player_rankings = sorted(the_elos_dict.keys(), key=lambda p: the_elos_dict[p], reverse=True)
    matchup_ranking_diffs = dict()
    for matchup in matches:
        matchup_ranking_diffs[matchup] = abs(player_rankings.index(matchup.player_1) - player_rankings.index(matchup.player_2))
    matchup_ranking_diffs = sorted(matchup_ranking_diffs.items(), key=lambda p: p[1], reverse=True)

    matchup_cycles = get_matchup_cycles(matches_by_player)
    with open(summary_filename, 'w') as outfile:
        outfile.write('--Matchup cycles------------------------------\n')
        for matchup_cycle in matchup_cycles:
            line = '  '
            for player_name in matchup_cycle:
                line += player_name + ', '
            outfile.write(line[:-1] + '\n')

        outfile.write('\n--Most imbalanced matchups (by rating difference)---------\n')
        for matchup, rating in matchup_rating_diffs[:20]:
            outfile.write('  {r1} - {r2}: {d}\n'.format(r1=matchup.player_1, r2=matchup.player_2, d=rating))

        outfile.write('\n--Most imbalanced matchups (by ranking difference)---------\n')
        for matchup, rating in matchup_ranking_diffs[:20]:
            outfile.write('  {r1} - {r2}: {d}\n'.format(r1=matchup.player_1, r2=matchup.player_2, d=rating))


def read_elos_from_csv(csv_filename: str) -> List[Tuple[str, float]]:
    elos = list()   # type: List[Tuple[str, float]]
    with open(csv_filename, 'r') as file:
        for line in file:
            vals = line.split(',')
            elos.append((vals[0].lower(), float(vals[1]),))
    return elos


def get_extra_banned_matchups() -> Set[Matchup]:
    matchups = set()    # type: Set[Matchup]
    with open(BANNED_MACHUPS_FILENAME) as file:
        for line in file:
            players = line.split(',')
            matchups.add(Matchup(players[0].lower(), players[1].lower().rstrip('\n')))
    return matchups


def get_banned_matchups() -> Set[Matchup]:
    db_conn = mysql.connector.connect(
        user=mysql_db_user,
        password=mysql_db_passwd,
        host=mysql_db_host,
        database=mysql_db_name
    )

    try:
        cursor = db_conn.cursor()
        cursor.execute(
            """
            SELECT 
                ud1.twitch_name AS racer_1,
                ud2.twitch_name AS racer_2,
                league_tag AS league
            FROM 
                matches
            JOIN
                necrobot.users ud1 ON ud1.user_id = matches.racer_1_id
            JOIN
                necrobot.users ud2 ON ud2.user_id = matches.racer_2_id
            WHERE
                league_tag = '{league}'
            """.format(league=LEAGUE)
        )

        banned_matchups = get_extra_banned_matchups()    # type: Set[Matchup]

        for row in cursor:
            if row[0] is None or row[1] is None:
                raise RuntimeError('Racer without twitch name in match database.')

            racer_1 = row[0].lower()
            racer_2 = row[1].lower()
            banned_matchups.add(Matchup(racer_1, racer_2))

        print('Banned matchups:', banned_matchups)

        return banned_matchups
    finally:
        db_conn.close()


def main():
    if len(sys.argv) > 1:
        elo_csv = sys.argv[1]
    else:
        elo_csv = INPUT_FILENAME

    matchup_output = MATCHUP_FILENAME
    summary_output = MATCHUP_SUMMARY_FILENAME
    dropped_racers = DROPPED_RACERS_FILENAME

    print('Making matchups from Elo file {}...'.format(elo_csv))
    write_matchup_csv_from_elo_csv(elo_csv, matchup_output, summary_output, dropped_racers)
    print('Matchups created.')


class TestAutomatch(unittest.TestCase):
    def test_enforce_min_cycles(self):
        matches = {
            Matchup('a', 'b'),
            Matchup('b', 'c'),
            Matchup('c', 'd'),
            Matchup('d', 'e'),
            Matchup('e', 'a'),
            Matchup('f', 'g'),
            Matchup('g', 'h'),
            Matchup('h', 'f'),
        }
        matches_by_player = get_matches_by_player(matches)
        elos = {
            'a': 0,
            'b': 1,
            'c': 3,
            'd': 6,
            'e': 10,
            'f': 15,
            'g': 21,
            'h': 28,
        }
        banned_matches = set()
        new_match_set = enforce_min_cycle_size(
                            matches_by_player=matches_by_player,
                            elos=elos,
                            banned_matches=banned_matches,
                            min_cycle_size=4
                        )

        new_match_cycles = get_matchup_cycles(get_matches_by_player(new_match_set))
        self.assertEqual(len(new_match_cycles), 1)

        banned_matches = {Matchup('d', 'h'), Matchup('d', 'g'), Matchup('d', 'f')}
        new_match_set = enforce_min_cycle_size(
            matches_by_player=matches_by_player,
            elos=elos,
            banned_matches=banned_matches,
            min_cycle_size=4
        )
        new_match_cycles = get_matchup_cycles(get_matches_by_player(new_match_set))
        self.assertEqual(len(new_match_cycles), 1)
        self.assertFalse(banned_matches.intersection(new_match_set))

    def test_enforce_min_cycles_2(self):
        matches = {
            Matchup('a', 'b'),
            Matchup('b', 'c'),
            Matchup('c', 'd'),
            Matchup('d', 'e'),
            Matchup('e', 'a'),
            Matchup('f', 'g'),
            Matchup('g', 'h'),
            Matchup('h', 'f'),
            Matchup('i', 'j'),
            Matchup('j', 'k'),
            Matchup('k', 'i'),
        }
        matches_by_player = get_matches_by_player(matches)
        elos = {
            'a': 1000,
            'b': 900,
            'c': 800,
            'd': 700,
            'e': 600,
            'f': 500,
            'g': 400,
            'h': 300,
            'i': 200,
            'j': 100,
            'k': 0,
        }
        banned_matches = set()
        new_match_set = enforce_min_cycle_size(
                            matches_by_player=matches_by_player,
                            elos=elos,
                            banned_matches=banned_matches,
                            min_cycle_size=6
                        )
        desired_matchups = {
            Matchup('a', 'b'),
            Matchup('b', 'c'),
            Matchup('c', 'd'),
            Matchup('d', 'f'),
            Matchup('f', 'h'),
            Matchup('h', 'j'),
            Matchup('j', 'k'),
            Matchup('k', 'i'),
            Matchup('i', 'g'),
            Matchup('g', 'e'),
            Matchup('e', 'a'),
        }
        self.assertEqual(new_match_set, desired_matchups)


if __name__ == "__main__":
    # unittest.main()
    main()
