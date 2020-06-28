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
WEEK = 0
RIDER = '{s}_{lg}_wk{w}'.format(s=SEASON, lg=LEAGUE, w=WEEK)
NUM_AUTOGENS = 1

INPUT_FILENAME = 'data/ratings_{r}.csv'.format(r=RIDER)
MATCHUP_FILENAME = 'data/matchups_{r}.csv'.format(r=RIDER)
MATCHUP_SUMMARY_FILENAME = 'data/matchcycles_{r}.csv'.format(r=RIDER)
BANNED_MACHUPS_FILENAME = 'data/bannedmatches_{s}.txt'.format(s=SEASON)


rand = random.Random()
rand.seed()


class Matchup(object):
    def __init__(self, player_1: str, player_2: str):
        self.player_1 = player_1 if player_1 < player_2 else player_2
        self.player_2 = player_2 if player_1 < player_2 else player_1

    def __hash__(self):
        return hash((self.player_1.lower(), self.player_2.lower(),))

    def __eq__(self, other):
        return self.player_1.lower() == other.player_1.lower() and self.player_2.lower() == other.player_2.lower()

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
    ilp_prob = pulp.LpProblem('Matchup problem', pulp.LpMaximize)

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
                    "Matchup_{0}_{1}".format(p1_name, p2_name), 0, 1, pulp.LpInteger
                )

    # Add entropy value of each matchup
    matchup_utility = dict()
    for player in matchups:
        for opp in matchups[player]:
            matchup_utility[matchups[player][opp]] = get_utility(elos[player], elos[opp])

    # Set optimization objective
    ilp_prob.setObjective(pulp.LpAffineExpression(matchup_utility, name="Matchup utility"))

    # Make constraints
    for player in matchups:
        edges_from_player = [matchups[player][opp] for opp in matchups[player]]
        for otherplayer in matchups:
            if player in matchups[otherplayer]:
                edges_from_player.append(matchups[otherplayer][player])

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


def write_matchup_csv_from_elo_csv(csv_filename: str, matchup_filename: str, summary_filename: str):
    the_elos = read_elos_from_csv(csv_filename)
    the_elos_dict = dict()
    name_cap_dict = dict()

    for player_name, _ in the_elos:
        name_cap_dict[player_name.lower()] = player_name

    for player_name, player_elo in the_elos:
        the_elos_dict[player_name.lower()] = player_elo
    matches = get_matchups(elos=the_elos_dict, banned_matches=get_banned_matchups(), num_matches=NUM_AUTOGENS)

    matches_by_player = dict()  # type: Dict[str, List[str]]
    for match in matches:
        if match.player_1 not in matches_by_player:
            matches_by_player[match.player_1] = list()
        if match.player_2 not in matches_by_player:
            matches_by_player[match.player_2] = list()
        matches_by_player[match.player_1].append(match.player_2)
        matches_by_player[match.player_2].append(match.player_1)

    with open(matchup_filename, 'w') as outfile:
        format_str = '{player_1},{player_2}\n'
        for match in matches:
            outfile.write(
                format_str.format(
                    player_1=name_cap_dict[match.player_1],
                    player_2=name_cap_dict[match.player_2]
                )
            )

    with open(summary_filename, 'w') as outfile:
        checked_players = set()
        for player_name, _ in the_elos:
            player_name = player_name.lower()
            if player_name in checked_players:
                continue
            checked_players.add(player_name)

            # Write the cycle starting with player_name
            line = player_name + ','
            while True:
                if player_name not in matches_by_player:
                    break # while
                match_list = matches_by_player[player_name]

                wrote_any = False
                for opp in match_list:
                    opp = opp.lower()
                    if opp not in checked_players:
                        wrote_any = True
                        line += opp + ','
                        player_name = opp
                        checked_players.add(opp)
                        break  # for

                if not wrote_any:
                    break # while

            outfile.write(line[:-1] + '\n')


def read_elos_from_csv(csv_filename: str) -> List[Tuple[str, float]]:
    elos = list()   # type: List[Tuple[str, float]]
    with open(csv_filename, 'r') as file:
        for line in file:
            vals = line.split(',')
            elos.append((vals[0], float(vals[1]),))
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
                ud2.twitch_name AS racer_2
            FROM 
                matches
            JOIN
                necrobot.users ud1 ON ud1.user_id = matches.racer_1_id
            JOIN
                necrobot.users ud2 ON ud2.user_id = matches.racer_2_id
            """
        )

        banned_matchups = get_extra_banned_matchups()    # type: Set[Matchup]

        for row in cursor:
            if row[0] is None or row[1] is None:
                raise RuntimeError('Racer without twitch name in match database.')

            racer_1 = row[0].lower()
            racer_2 = row[1].lower()
            banned_matchups.add(Matchup(racer_1, racer_2))

        print(banned_matchups)

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

    print('Making matchups from Elo file {}...'.format(elo_csv))
    write_matchup_csv_from_elo_csv(elo_csv, matchup_output, summary_output)
    print('Matchups created.')


if __name__ == "__main__":
    # unittest.main()
    main()


class TestAutomatch(unittest.TestCase):
    MAX_ELO = 678
    MIN_ELO = -833

    def test_automatch(self):
        elo_csv = 'data/ratings_test.csv'
        matchup_output = 'data/matchups.csv'
        matchpairs_output = 'data/matchpairs.csv'
        write_matchup_csv_from_elo_csv(elo_csv, matchup_output, matchpairs_output)

    def test_season_automatch(self):
        num_weeks = 5
        the_elos = self.get_elos()
        banned_matches = set()  # type: Set[Matchup]
        matches_by_week = []    # type: List[Set[Matchup]]
        for _ in range(num_weeks):
            week_matches = get_matchups(
                elos=the_elos,
                banned_matches=banned_matches
            )
            matches_by_week.append(week_matches)
            for matchup in week_matches:
                banned_matches.add(matchup)

        matches_by_player = dict()  # type: Dict[str, Dict[int, List[str]]]
        for the_player in the_elos.keys():
            matches_by_player[the_player] = dict()
            for week_num in range(1, num_weeks + 1):
                matches_by_player[the_player][week_num] = list()

        for week_num in range(1, num_weeks + 1):
            for matchup in matches_by_week[week_num - 1]:
                matches_by_player[matchup.player_1][week_num].append(matchup.player_2)
                matches_by_player[matchup.player_2][week_num].append(matchup.player_1)

        self.print_dict_formatted(matches_by_player)

    @staticmethod
    def print_dict_as_csv(matchups_by_player: Dict[str, Dict[int, List[str]]]) -> None:
        outfile = open('ilp_test.csv', 'w')
        for plr, week_dict in matchups_by_player.items():
            the_line = plr + ','
            for week, opps in week_dict.items():
                for opp in opps:
                    the_line += opp + ','

            the_line = the_line[:-1] + '\n'
            outfile.write(the_line)

    @staticmethod
    def print_dict_formatted(matchups_by_player: Dict[str, Dict[int, List[str]]]) -> None:
        outfile = open('ilp_test.txt', 'w')
        for plr, week_dict in matchups_by_player.items():
            the_line = '{plr:>20}: '.format(plr=plr)
            for week, opps in week_dict.items():
                for opp in opps:
                    the_line += opp + ', '
                the_line = the_line[:-2] + ' -- '

            the_line = the_line[:-4] + '\n'
            outfile.write(the_line)

    @classmethod
    def interpolate_for_elo(cls, a: float, b: float, eloval: float) -> float:
        elo_range = cls.MAX_ELO - cls.MIN_ELO
        return (a*(cls.MAX_ELO - eloval) + b*(eloval - cls.MIN_ELO))/elo_range

    @staticmethod
    def get_winner(elo_1: float, elo_2: float):
        pwin = 1.0 / (1 + pow(10, float(elo_2 - elo_1)/400.0))
        if rand.random() < pwin:
            return 1
        else:
            return 2

    @staticmethod
    def get_elos() -> Dict[str, float]:
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

        the_elos = {}
        for line in elo_str.split('\n'):
            args = line.split()
            if args:
                the_elos[args[1]] = float(args[2])

        return the_elos
