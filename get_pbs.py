#!/usr/bin/python3

# TODO:
# handle score runs (right now timedelta handling is hardcoded)
# enumerate valid strings
# handle fetch errors
# paginate :)

import argparse
import sys
import csv
import os
from datetime import datetime, timedelta
import time
import requests

CACHE_DIR = '.necrolab-cache'
CHARACTER = 'cadence'
MAX_AGE_DAYS = 1


class LeaderboardParams:

    def __init__(
        self,
        character='cadence',
        version='amplified',
        game_mode='normal',
        run_type='speed',
        percent='any-percent',
        seeding='unseeded',
    ):
        self.character = character
        self.version = version
        self.game_mode = game_mode
        self.run_type = run_type
        self.percent = percent
        self.seeding = seeding


class LeaderboardEntry:
    rank = None
    time = None
    name = None

    def __init__(self, rank, time, name):
        self.rank = rank
        self.time = time
        self.name = name
    
    def __repr__(self):
        return f'{self.rank}/{self.time}/{self.name}'


def slug(leaderboard_params):
    return "~".join([
        "all-time",
        leaderboard_params.percent,
        leaderboard_params.character,
        leaderboard_params.version,
        leaderboard_params.game_mode,
        leaderboard_params.run_type,
        leaderboard_params.seeding,
        "single-player",
        "ost",
    ])


def url(leaderboard_params):
    return f"https://www.necrolab.com/api/leaderboards/steam/{slug(leaderboard_params)}/entries?limit={args.request_limit}"


def cache_file(leaderboard_params):
    return os.path.join(CACHE_DIR, slug(leaderboard_params))


# TODO just cache the json itself, honestly
def retrieve(leaderboard_params):
    fetch_url = url(leaderboard_params)
    response = requests.get(fetch_url)
    if 'data' not in response.json():
        print(
            f'Unexpected json response (no data field):\n{response.json()}\n for:\n{fetch_url})', file=sys.stderr)
        sys.exit(1)
    if not response.json()['data']:
        print(
            f'Empty response from NecroLab API for:\n{fetch_url}', file=sys.stderr)
        print(f'Check that your parameters are spelled correctly.', file=sys.stderr)
        sys.exit(1)
    return [
        LeaderboardEntry(e["rank"], timedelta(seconds=float(e["metric"])),
                         e["username"]) for e in response.json()["data"]
    ]


def write_cache_file(filename, entries):
    if os.path.isfile(CACHE_DIR):
        print(f'CACHE_DIR "{CACHE_DIR}" is a file!', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
    with open(filename, "w", encoding="utf-8", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows([e.rank, e.time.total_seconds(), e.name]
                         for e in entries)


def leaderboard_entry_from_csv_line(r):
    return LeaderboardEntry(r[0], timedelta(seconds=float(r[1])), r[2])


def read_cache_file(filename):
    with open(filename, encoding="utf-8", newline="") as csvfile:
        reader = csv.reader(csvfile)
        # return [LeaderboardEntry(r[0], r[1], r[2]) for r in reader]
        return [leaderboard_entry_from_csv_line(r) for r in reader]


def cache_file_up_to_date(filename):
    if not os.path.isfile(filename):
        print(f"No cache at: {filename}", file=sys.stderr)
        return False
    age = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(
        os.path.getmtime(filename))
    max_age = timedelta(days=MAX_AGE_DAYS)
    print(f"Cache age: {age}", file=sys.stderr)
    return age <= max_age


def get_leaderboard(leaderboard_params):
    filename = cache_file(leaderboard_params)
    if cache_file_up_to_date(filename):
        return read_cache_file(filename)
    else:
        leaderboard = retrieve(leaderboard_params)
        write_cache_file(filename, leaderboard)
        return leaderboard


def leaderboard_by_name(leaderboard):
    return {e.name.lower(): e for e in leaderboard if e.name}


def find_pb(username, name_map):
    if username in name_map:
        return name_map[username]
    else:
        for name, entry in name_map.items():
            if username in name:
                print(
                    f'Warning: matching username "{username}" to steam name "{name}"',
                    file=sys.stderr,
                )
                return entry
        return LeaderboardEntry("?", "?", "?")

def best_time(times):
    actual_times = [t for t in times if isinstance(t, timedelta)]
    if len(actual_times) == 0:
        return None
    return min(actual_times)

def best_time_for_ranking(times):
    best = best_time(times)
    return best if best else timedelta(minutes=20)

def best_time_string(times):
    best = best_time(times)
    return best if best else "?"
    


def find_pbs(usernames, name_maps, elos=False):
    output_rows = []
    if not elos:
        output_rows.append(["name"] + list(name_maps.keys()) + ["best"])
    if elos:
        ranked_users = sorted([(u, best_time_for_ranking([find_pb(u, name_map).time for _, name_map in name_maps.items()]))
                               for u in usernames], key=lambda x: x[1])
        for (n, (u, t)) in enumerate(ranked_users):
            # interpolate from 1800 to 1200
            elo = 1800 - 600 * n / (len(ranked_users)-1)
            print(f'{u},{round(elo)}')
    else:
        for u in usernames:
            times = [
                find_pb(u, name_map).time
                for _, name_map in name_maps.items()
            ]
            output_rows.append([u] + [str(t) for t in times] +
                               [str(best_time(times))])
    for r in output_rows:
        print(args.output_separator.join(r))


def leaderboard_params_from_args(args, **kwargs):
    return LeaderboardParams(
        character=args.character,
        game_mode=args.game_mode,
        seeding=args.seeding,
        **kwargs,
    )


parser = argparse.ArgumentParser(description='Look up PBs from necrolab')
parser.add_argument('--cache_dir', default=CACHE_DIR)
parser.add_argument('--character', default=CHARACTER)
parser.add_argument('--game_mode', default='normal')
parser.add_argument('--seeding', default='unseeded')
parser.add_argument('--output_separator', default='\t')
parser.add_argument('--users_file', default=None)
parser.add_argument('--elos', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--request_limit', default=1000)
parser.add_argument('users', metavar='user', nargs='*')

args = parser.parse_args()

leaderboards = {
    'synchrony':
    get_leaderboard(
        leaderboard_params_from_args(args, version='synchrony-amplified')),
}
if args.character not in ('ensemble', 'chaunter', 'klarinetta', 'suzu'):
    leaderboards['amplified'] = get_leaderboard(leaderboard_params_from_args(args, version='amplified'))

name_maps = {k: leaderboard_by_name(v) for k, v in leaderboards.items()}

users = args.users.copy()
if args.users_file:
    with open(args.users_file) as f:
        for l in f:
            users.append(l.strip())
print("users: ", users, file=sys.stderr)
find_pbs([x.lower() for x in users], name_maps, args.elos)
