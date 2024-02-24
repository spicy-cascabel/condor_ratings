#!/bin/bash
# This is a kinda overkill method - we make fake ELOs using PBs, then run them
# through elorate (which really just outputs the same thing again, given that
# no matches have happened yet), then use automatch.
rootdir="$(dirname $0)"
function matchups() {
	league="$1"
	character="$2"
	$rootdir/get_pbs.py --elos --character=$character --users_file=data/racers_${league}.txt > data/ratings_15_${league}_wk0.csv &&
	python3 elorate.py --league=${league} &&
	python3 automatch.py --league=${league}
}

matchups noc nocturna
matchups dia diamond
matchups mel melody
matchups ens ensemble
