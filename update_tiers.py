#!/usr/bin/python3

import re
import itertools
import operator
import csv
import argparse
import pdb
import mysql.connector
import racer_lookups
import sys
import pprint

parser = argparse.ArgumentParser(description='Look up PBs from necrolab')
parser.add_argument('--mode', default='update', help='create|update')
parser.add_argument('--elo-csv', default=None,
                    help='csv file with name,elo rows and no header')
parser.add_argument('--side-event-signups-csv', default=None,
                    help='[for create mode] csv file with name,1,0,1,0 rows indicating signups, and header "name,<league>,<league>,..."')
parser.add_argument('--tier-sizes', default='8,8,8,8,*',
                    help='comma-delimited list of tier sizes. * indicates "the rest"')
parser.add_argument('--tier-names', default='Crystal,Obsidian,Titanium,Gold,Blood',
                    help='comma-delimited list of tier names.')
parser.add_argument('--week', default=None)
parser.add_argument('--season', default=None)
parser.add_argument('--season-db', default=None,
                    help='Defaults to condor<season>')
parser.add_argument('--dry-run', default=True, action=argparse.BooleanOptionalAction,
                    help='Only output SQL commands that would be run; set to false to write to DB.')
parser.add_argument('--db-readwrite-user', default=None)
parser.add_argument('--db-readwrite-password', default=None,
                    help='Never ever put this in code!!!')
parser.add_argument('--sql_outfile', default=None)
args = parser.parse_args()


def get_db(user='necrobot-read', password='necrobot-read', host='condor.live', database_name=None):
    return mysql.connector.connect(
        user=user,
        password=password,
        host=host,
        database=database_name
    )


class RacerStats:
    def __init__(self, necrobot_racer):
        self.necrobot_racer = necrobot_racer
        self.side_events = None
        self.elo = None
        self.tier = None

    def __repr__(self):
        return '{:10s} {:4s} {:20s} {}'.format(
            self.tier if self.tier else 'n/a',
            str(self.elo) if self.elo else 'n/a',
            str(self.side_events) if self.side_events else 'n/a',
            self.necrobot_racer)


def read_elo_csv(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        # this is hopefully already sorted, but make sure!
        return sorted([[r[0], int(r[1])] for r in reader], key=lambda x: -x[1])


def read_side_event_signups_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames[0] != 'name':
            print('Expected header "name,<league>,..." but got {}'.format(
                ',', join(reader.fieldnames)), file=sys.stderr)
            sys.exit(1)
        return (reader.fieldnames[1:], list(reader))


def parse_tier_size(t):
    if t.isnumeric() and int(t) > 0:
        return int(t)
    if t == '*':
        return t
    print(f'bad tier size: "{t}"', file=sys.stderr)
    sys.exit(1)


def assign_tiers(elos, tier_size_spec, tier_name_spec):
    tier_sizes = [parse_tier_size(t) for t in tier_size_spec.split(',')]
    tier_names = tier_name_spec.split(',')
    if tier_sizes.count('*') > 1:
        print('only one tier may have unbounded size (*)', file=sys.stderr)
        sys.exit(1)
    if len(tier_sizes) != len(tier_names):
        print('length mismatch between tier names and tier sizes', file=sys.stderr)
        sys.exit(1)

    star_pos = tier_sizes.index('*') if '*' in tier_sizes else None
    if not star_pos and len(elos) != sum(tier_sizes):
        print(
            f'have {len(elos)} racers, but tier sizes {tier_size_spec} add up to ' +
            f'{sum(tier_sizes)}', file=sys.stderr)
        sys.exit(1)

    if star_pos:
        nonstar_sum = sum([t for t in tier_sizes if t != '*'])
        if nonstar_sum >= len(elos):
            print(
                f'have {len(elos)} racers, but non-* tier sizes in {tier_size_spec} ' +
                f'add up to {nonstar_sum}', file=sys.stderr)
            sys.exit(1)

    # Accumulate tier sizes into slice cutoffs, starting from the front and
    # back, leaving the possible middle for the '*'.
    split_pos = star_pos if star_pos else len(tier_sizes)
    top_tier_cutoffs = [
        0] + list(itertools.accumulate(tier_sizes[:split_pos], operator.add))
    bottom_tier_cutoffs = list(reversed(list(itertools.accumulate(tier_sizes[(
        split_pos+1):], operator.add)))) if split_pos < len(tier_sizes) - 1 else []
    # Negative slices would also work but this hurt my brain less and was easier to validate.
    bottom_tier_cutoffs = [len(elos) - t for t in (bottom_tier_cutoffs + [0])]
    cutoffs = top_tier_cutoffs + bottom_tier_cutoffs
    tier_map = {}
    for (tier_index, (lower, upper)) in enumerate(zip(cutoffs[:-1], cutoffs[1:])):
        for r in elos[lower:upper]:
            tier_map[r[0]] = tier_names[tier_index]
    return tier_map


def print_summary(racer_stats, tier_names):
    main_event_racers = len([r for r in racer_stats.values() if r.tier])
    side_event_racers = len([r for r in racer_stats.values() if r.side_events])
    tier_counts = {tier: len(ids)
                   for tier, ids in tier_to_id_map(racer_stats).items()}
    tier_summary = ', '.join(
        [f'{name}: {tier_counts[name]}' for name in tier_names.split(',')])
    print('Summary:')
    print(f'  main event entrants: {main_event_racers}', file=sys.stderr)
    print(f'    {tier_summary}')
    print(f'  side event entrants: {side_event_racers}', file=sys.stderr)
    print(f'  total racers: {len(racer_stats)}')


# This is horrible please don't be mad.
# Names are the *only* thing here that can run afoul of sql escaping issues.
# It would take me a bit to rewrite this to still support dry runs printing the
# full commands while also using execute(cmd, params) like I'm supposed to.
# (why isn't there a variation of execute that gives you the string??)
# so... just strip characters idk
# this *is* the racer name that gets displayed on the website, so ideally fix this :/
_SAFE_NAME_REGEX = re.compile('[^A-Za-z0-9_ ]')


def safe_name(necrobot_racer):
    # if necrobot_racer.discord_name:
        # return _SAFE_NAME_REGEX.sub('', necrobot_racer.discord_name)
    # else:
        # return _SAFE_NAME_REGEX.sub('', necrobot_racer.twitch_name)
    return necrobot_racer.twitch_name


def create_entrants_table_cmds(racer_stats, side_event_leagues):
    side_league_columns = ', '.join(
        [f'{l} SMALLINT(1)' for l in side_event_leagues])
    create_cmd = f'''
    CREATE TABLE entrant_info (
        user_id SMALLINT(5) NOT NULL PRIMARY KEY,
        racer_name TINYTEXT NOT NULL,
        current_tier VARCHAR(10),
        w1_tier VARCHAR(10),
        w2_tier VARCHAR(10),
        w3_tier VARCHAR(10),
        w4_tier VARCHAR(10),
        {side_league_columns}
        );
    '''
    racer_entries = ',\n'.join(
        [f'({r.necrobot_racer.user_id}, "{safe_name(r.necrobot_racer)}")' for r in racer_stats.values()])
    # TODO more carefully support racers already being present
    # that said, at this point it's possible to just drop the table and recreate
    # off of each week's ELOs if absolutely necessary
    insert_racers_cmd = f'''
    INSERT IGNORE INTO entrant_info (user_id, racer_name) VALUES
    {racer_entries};'''
    print('Table creation:')
    print(create_cmd)
    print('\n')
    print('Racer entry insertion:')
    print(insert_racers_cmd)
    print('\n')
    return [create_cmd, insert_racers_cmd]


def build_update_side_event_cmd(racer, side_event_leagues):
    values = ', '.join([f'{l} = {entered}' for l, entered in zip(
        side_event_leagues, racer.side_events)])
    return f'''
    UPDATE entrant_info SET
    {values}
    WHERE user_id = {racer.necrobot_racer.user_id};
    '''


def update_side_event_signups_cmds(racer_stats, side_event_leagues):
    side_event_cmd = ''.join([build_update_side_event_cmd(
        r, side_event_leagues) for r in racer_stats.values() if r.side_events])
    print('Update side events:')
    print(side_event_cmd)
    return [side_event_cmd]


def build_update_tier_cmd(tier, racer_ids, week):
    ids = ', '.join([str(i) for i in racer_ids])
    return f'''
    UPDATE entrant_info SET
      current_tier = "{tier}", w{week}_tier = "{tier}"
      WHERE user_id in ({ids});
    '''


def tier_to_id_map(racer_stats):
    tier_map = {}
    for r in racer_stats.values():
        if not r.tier:
            continue
        if r.tier not in tier_map:
            tier_map[r.tier] = []
        tier_map[r.tier].append(r.necrobot_racer.user_id)
    return tier_map


def update_tiers_cmds(racer_stats, week):
    tier_map = tier_to_id_map(racer_stats)
    update_tiers_cmd = ''.join(
        [build_update_tier_cmd(tier, racers, week) for (tier, racers) in tier_map.items()])
    print('Update tiers:')
    print(update_tiers_cmd)
    print('')
    return [update_tiers_cmd]


def execute_write_commands(commands, user, password, season_db):
    readwrite_db = mysql.connector.connect(
        user=user,
        password=password,
        host='condor.live',
        database=season_db
    )
    readwrite_db.start_transaction()
    try:
        cursor = readwrite_db.cursor()
        for c in commands:
            print('==========================================')
            print('==========================================')
            print('==========================================')
            print("command: '{}'".format(c))
            results = cursor.execute(c)
            if results:
                for result in results:
                    if result.with_rows:
                        print("Rows produced by statement '{}':".format(
                            result.statement))
                        print(result.fetchall())
                    else:
                        print("Number of rows affected by statement '{}': {}".format(
                            result.statement, result.rowcount))
        print('==========================================')
        print('==========================================')
        print('==========================================')
        readwrite_db.commit()
    finally:
        cursor.close()
        readwrite_db.close()


def main():
    if args.mode not in ('create', 'update'):
        print(f'unknown mode: "{args.mode}"', file=sys.stderr)
        sys.exit(1)
    if bool(args.season) == bool(args.season_db):
        print(f'exactly one of --season or --season-db must be provided',
              file=sys.stderr)
        sys.exit(1)

    elos = read_elo_csv(args.elo_csv)

    # There may be side-event-only racers.
    all_racers = set([e[0]for e in elos])
    side_event_signups = None
    if args.side_event_signups_csv:
        (side_leagues, side_event_signups) = read_side_event_signups_csv(
            args.side_event_signups_csv)
        all_racers.update([s['name'] for s in side_event_signups])

    readonly_db = get_db()
    (racers, inexact_matches, missing) = racer_lookups.lookup_racers(
        readonly_db, all_racers)

    print('Generating per-racer data...')
    racer_stats = {r.user_id: RacerStats(r) for r in racers.values()}
    tiers = assign_tiers(elos, args.tier_sizes, args.tier_names)
    for elo_entry in elos:
        name = elo_entry[0]
        if name not in racers:
            print(f'skipping {name} (main event)', file=sys.stderr)
            continue
        user_id = racers[name].user_id
        racer_stats[user_id].elo = elo_entry[1]
        racer_stats[user_id].tier = tiers[name]
    if side_event_signups:
        for entry in side_event_signups:
            name = entry['name']
            if name not in racers:
                print(f'skipping {name} (side events)', file=sys.stderr)
                continue
            racer_stats[racers[name].user_id].side_events = [entry[l]
                                                             for l in side_leagues]
    print('')

    commands = []
    week = args.week
    if args.mode == 'create':
        week = 1
        commands.extend(create_entrants_table_cmds(
            racer_stats, side_event_leagues=side_leagues))
    if side_event_signups:
        commands.extend(update_side_event_signups_cmds(
            racer_stats, side_leagues))
    commands.extend(update_tiers_cmds(racer_stats, week=week))

    season_db = args.season_db if args.season_db else f'condor{args.season}'
    print(f'Will operate on season database: {season_db}')
    print_summary(racer_stats, args.tier_names)
    if args.dry_run:
        print('^^ CHECK THIS CAREFULLY ^^', file=sys.stderr)
        print('^^ MAKE SURE SEASON IS CORRECT ^^', file=sys.stderr)
        print('and then run with --no-dry-run', file=sys.stderr)
    if inexact_matches:
        print('Racer(s) with inexact matches found in necrobot user list:')
        for a, b in inexact_matches:
            print(f'  "{a}" -> "{b}"')
        print('Check that these are correct, and ideally correct them in input files!')
    if missing:
        print(
            f'Reminder: missing racers present: {missing}', file=sys.stderr)
        print('(see above SQL commands for possible matches by edit distance)')

    if args.sql_outfile:
        with open(args.sql_outfile, 'w') as f:
            f.write(f'USE {season_db};\n')
            for c in commands:
                f.write(c)

    if not args.dry_run:
        if not args.db_readwrite_user or not args.db_readwrite_password:
            print('To write to the DB, --db-readwrite-user and --db_readwrite-password are required.', file=sys.stderr)
            print('DID NOT WRITE!', file=sys.stderr)
            sys.exit(1)
        print('EXECUTION NOT YET SUPPORTED SORRY :(', file=sys.stderr)
        print('use --sql_outfile=path/to/sqlfile and execute manually with:', file=sys.stderr)
        print('mysql --host=condor.live --user=... --password=... < path/to/sqlfile', file=sys.stderr)
        sys.exit(1)
        execute_write_commands(commands, user=args.db_readwrite_user,
                               password=args.db_readwrite_password, season_db=season_db)


main()
