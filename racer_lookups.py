from datetime import datetime, timedelta
import editdistance
import json
import os
import re
import sys
import time
import pdb


class NecrobotRacer:
    def __init__(self, user_id, discord_name, twitch_name):
        self.user_id = user_id
        self.discord_name = discord_name
        self.twitch_name = twitch_name

    def __str__(self):
        return f'{self.user_id}/{self.discord_name}/{self.twitch_name}/'

    def __repr__(self):
        return str(self)

    def from_dict(d):
        return NecrobotRacer(d['user_id'], d['discord_name'], d['twitch_name'])

    def to_dict(self):
        return {'user_id': self.user_id, 'discord_name': self.discord_name, 'twitch_name': self.twitch_name}


def get_all_racers_impl(db_conn):
    cursor = db_conn.cursor()
    cursor.execute("""
    select user_id, discord_name, twitch_name from necrobot.users
                    """)
    # We drop racers without twich *or* discord set. Apparently it's possible to
    # register with just rtmp_name??
    return [NecrobotRacer(row[0], row[1], row[2]) for row in cursor if row[1] or row[2]]


def cache_file_up_to_date(filename, max_cache_age_days=1):
    if not os.path.isfile(filename):
        print(f"No cache at: {filename}", file=sys.stderr)
        return False
    age = datetime.fromtimestamp(time.time()) - datetime.fromtimestamp(
        os.path.getmtime(filename))
    max_age = timedelta(days=max_cache_age_days)
    print(f"Cache age: {age}", file=sys.stderr)
    return age <= max_age


def read_cache_file(filename):
    with open(filename) as f:
        return json.load(f)


def write_cache_file(data, filename):
    cache_dir = os.path.dirname(filename)
    if os.path.isfile(cache_dir):
        print(f'Cache directory "{cache_dir}" is a file!', file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    with open(filename, 'w') as f:
        json.dump(data, f)


def racers_cache_file(cache_dir):
    return os.path.join(cache_dir, "necrobot-users")


def get_all_racers(db, cache_dir):
    cache_file = racers_cache_file(cache_dir)
    if cache_file_up_to_date(cache_file):
        return [NecrobotRacer.from_dict(d) for d in read_cache_file(cache_file)]
    else:
        racers = get_all_racers_impl(db)
        write_cache_file([r.to_dict() for r in racers], cache_file)
        return racers


_CANONICALIZATION_REMOVE_PATTERN = re.compile(r'[^a-z]')


def canonicalize_name(n):
    return _CANONICALIZATION_REMOVE_PATTERN.sub('', n.lower())


def distance(name, racer):
    distances = []
    if racer.twitch_name:
        distances.append(editdistance.eval(name, racer.twitch_name))
    if racer.discord_name:
        distances.append(editdistance.eval(name, racer.discord_name))
    if len(distances) == 0:
        print(racer)
    return min(distances)


def closest_matches(name, racers):
    return sorted([(r, distance(name, r)) for r in racers], key=lambda x: x[1])[:3]


# Racers are human, and therefore quite bad at getting their names correct.
# To that end, we make multiple passes, all of which attempt to match with both
# twitch and discord names:
# - exact case-insensitive match
# - "canonicalized" match
# - edit-distance on canonicalized names (debug output only)

def lookup_racers(db, racer_names, cache_dir='.necrobot-cache'):
    racers = get_all_racers(db, cache_dir)
    inexact_matches = []
    remaining_names = set(racer_names)
    result = {}
    twitch_map = {r.twitch_name.lower(): r for r in racers if r.twitch_name}
    discord_map = {r.discord_name.lower(): r for r in racers if r.discord_name}

    found_names = set()
    for n, c in [(n, n.lower()) for n in remaining_names]:
        if c in twitch_map:
            result[n] = twitch_map[c]
            found_names.add(n)
        elif c in discord_map:
            result[n] = discord_map[c]
            found_names.add(n)
    remaining_names = remaining_names.difference(found_names)
    found_names = set()

    twitch_map = {canonicalize_name(
        r.twitch_name): r for r in racers if r.twitch_name}
    discord_map = {canonicalize_name(
        r.discord_name): r for r in racers if r.discord_name}
    for n, c in [(n, canonicalize_name(n)) for n in remaining_names]:
        if c in twitch_map:
            result[n] = twitch_map[c]
            inexact_matches.append((n, c))
            found_names.add(n)
        elif c in discord_map:
            result[n] = discord_map[c]
            inexact_matches.append((n, c))
            found_names.add(n)
    remaining_names = remaining_names.difference(found_names)
    found_names = set()

    for n in remaining_names:
        print(f'Not found: \'{n}\'; closest matches: {closest_matches(n, racers)}', file=sys.stderr)
    return (result, inexact_matches, remaining_names)
