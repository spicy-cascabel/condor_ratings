from collections import defaultdict

LEAGUE = 'coh'
SEASON = 'sx'
WEEK = 0
RIDER = '{s}_{lg}_wk{w}'.format(s=SEASON, lg=LEAGUE, w=WEEK)

MATCHUP_FILENAME = 'data/matchups_{r}.csv'.format(r=RIDER)
SUMMARY_FILENAME = 'data/matchups_check_{r}.csv'.format(r=RIDER)

if __name__ == "__main__":
    matchups = defaultdict(lambda: set())
    with open(MATCHUP_FILENAME, 'r') as file:
        for line in file:
            racers = line.split(',')
            r1 = racers[0].lower()
            r2 = racers[1].lower().rstrip('\n')
            matchups[r1].add(r2)
            matchups[r2].add(r1)

    with open(SUMMARY_FILENAME, 'w') as outfile:
        checked_players = set()
        for player_name in matchups.keys():
            player_name = player_name.lower()
            if player_name in checked_players:
                continue
            checked_players.add(player_name)

            # Write the cycle starting with player_name
            line = player_name + ','
            while True:
                match_list = matchups[player_name.lower()]
                if not match_list:
                    break  # while

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

