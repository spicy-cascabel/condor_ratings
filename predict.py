import json
import math


ratings = {
}


def elo_to_r(elo: float) -> float:
    return elo*(0.52/150)


def biggielevs_phi(x: float) -> float:
    alpha = 0.42
    return (1 + alpha*math.erf(x*math.sqrt(math.pi)/2) + (1-alpha)*(x / (1 + abs(x))))/2


def odds_of_winning(mmr_difference: float) -> float:
    return biggielevs_phi(elo_to_r(mmr_difference))


def odds_of_winning_bo3(mmr_difference: float) -> float:
    p = odds_of_winning(mmr_difference)
    return p**3 + 3*(p**2)*(1-p)


def odds_of_winning_bo5(mmr_difference: float) -> float:
    p = odds_of_winning(mmr_difference)
    q = 1-p
    return p**5 + 5*(p**4)*q + 10*(p**3)*(q**2)


def most_likely_outcome_bo3(mmr_difference: float) -> str:
    p = odds_of_winning(mmr_difference)
    q = 1-p
    odds_of_outcome = {
        '2-0': p**2,
        '2-1': 2*(p**2)*q,
        '1-2': 2*(q**2)*p,
        '0-2': q**2,
    }
    return max(odds_of_outcome.keys(), key=lambda x: odds_of_outcome[x])


def most_likely_outcome_bo5(mmr_difference: float) -> str:
    p = odds_of_winning(mmr_difference)
    q = 1-p
    odds_of_outcome = {
        '3-0': p**3,
        '3-1': 3*(p**3)*q,
        '3-2': 6*(p**3)*(q**2),
        '2-3': 6*(q**3)*(p**2),
        '1-3': 3*(q**3)*p,
        '0-3': q**3,
    }
    return max(odds_of_outcome.keys(), key=lambda x: odds_of_outcome[x])


def get_odds_table(odds_fn) -> str:
    bo3_string = " "*16
    for name_1 in ratings.keys():
        bo3_string += "{0:>16}".format(name_1)

    bo3_string += "\n"
    for name_1, rating_1 in ratings.items():
        bo3_string += "{0:>15} ".format(name_1)
        for rating_2 in ratings.values():
            bo3_string += "            {0:>04.1f}".format(100*odds_fn(mmr_difference=(rating_1 - rating_2)))
        bo3_string += "\n"
    return bo3_string


def get_outcome_table() -> str:
    outcome_table = " "*29 + "   2-0   2-1   1-2   0-2   |   3-0   3-1   3-2   2-3   1-3   0-3\n"
    format_str = "{n1:>14}-{n2:<14}:  {o1:>02.0f}%   {o2:>02.0f}%   {o3:>02.0f}%   {o4:>02.0f}%   |" \
                 "   {o5:>02.0f}%   {o6:>02.0f}%   {o7:>02.0f}%   {o8:>02.0f}%   {o9:>02.0f}%   {o10:>02.0f}%"
    for name_1, rating_1 in ratings.items():
        for name_2, rating_2 in ratings.items():
            if not name_1 < name_2:
                continue
            p = odds_of_winning(mmr_difference=(rating_1-rating_2))
            q = 1-p
            outcomes = [100*(p**2), 100*2*(p**2)*q, 100*2*(q**2)*p, 100*(q**2),
                        100*(p**3), 100*3*(p**3)*q, 100*6*(p**3)*(q**2),
                        100*6*(q**3)*(p**2), 100*3*(q**3)*p, 100*(q**3)]
            outcome_table += format_str.format(
                n1=name_1,
                n2=name_2,
                o1=outcomes[0],
                o2=outcomes[1],
                o3=outcomes[2],
                o4=outcomes[3],
                o5=outcomes[4],
                o6=outcomes[5],
                o7=outcomes[6],
                o8=outcomes[7],
                o9=outcomes[8],
                o10=outcomes[9],
            )
            outcome_table += "\n"
    return outcome_table


def get_outcomes_json():
    outcomes_json = dict()
    for name_1, rating_1 in ratings.items():
        predictions_dict = dict()
        for name_2, rating_2 in ratings.items():
            p = odds_of_winning(mmr_difference=(rating_1-rating_2))
            q = 1-p
            bo3_dict = {
                'win': int(100*(p**2 + 2*(p**2)*q)),
                '2-0': int(100*(p**2)),
                '2-1': int(100*2*(p**2)*q),
                '1-2': int(100*2*(q**2)*p),
                '0-2': int(100*(q**2)),
            }
            bo5_dict = {
                'win': int(100*(p**3 + 3*(p**3)*q + 6*(p**3)*(q**2))),
                '3-0': int(100*p**3),
                '3-1': int(100*3*(p**3)*q),
                '3-2': int(100*6*(p**3)*(q**2)),
                '2-3': int(100*6*(q**3)*(p**2)),
                '1-3': int(100*3*(q**3)*p),
                '0-3': int(100*(q**3)),
            }
            predictions_dict[name_2] = {'bo3': bo3_dict, 'bo5': bo5_dict}

        outcomes_json[name_1] = {'predictions': predictions_dict}
    return outcomes_json


if __name__ == "__main__":
    with open('data/sxii_finals_matchscore_predictions.json', 'w') as file:
        json.dump(get_outcomes_json(), file, indent=4)
    # with open('data/s8_finals_predictions.txt', 'w') as file:
    #     file.write('Best of 3 predictions:\n')
    #     file.write(get_odds_table(odds_of_winning_bo3))
    #     file.write('\nBest of 5 predictions:\n')
    #     file.write(get_odds_table(odds_of_winning_bo5))
    #     file.write('\nMost likely outcome in a bo3:\n')
    #     file.write(get_outcome_table(most_likely_outcome_bo3))
    #     file.write('\nMost likely outcome in a bo5:\n')
    #     file.write(get_outcome_table(most_likely_outcome_bo5))

