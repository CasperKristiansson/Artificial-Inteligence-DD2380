import time


def evaluate(state):
    score = 0

    score += state.player_scores[0] - state.player_scores[1]

    skip_distance_calculation = [False, False]

    for player in [0, 1]:
        if state.player_caught[player] != -1:  # Player has caught a fish
            skip_distance_calculation[player] = True
            fish_value = state.fish_scores[state.player_caught[player]]
            hook_y = state.hook_positions[player][1]
            if hook_y == 19:
                score += fish_value if player == 0 else -fish_value
            else:
                distance_to_score = 19 - hook_y
                score += (fish_value / distance_to_score) if player == 0 else -(fish_value / distance_to_score)

    for fish_index, fish_pos in state.fish_positions.items():
        if fish_index not in state.player_caught.values():
            for player in [0, 1]:
                if not skip_distance_calculation[player]:
                    hook_pos = state.hook_positions[player]
                    distance = abs(hook_pos[0] - fish_pos[0]) + abs(hook_pos[1] - fish_pos[1])

                    if distance == 0:
                        score_change = state.fish_scores[fish_index]
                    else:
                        score_change = state.fish_scores[fish_index] / distance

                    score += score_change if player == 0 else -score_change

    return score


def min_max(node, depth, alpha, beta, maximizing_player, start_time):
    if depth == 0 or time_is_up(start_time):
        return evaluate(node.state)

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.compute_and_get_children():
            eval = min_max(child, depth - 1, alpha, beta, False, start_time)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.compute_and_get_children():
            eval = min_max(child, depth - 1, alpha, beta, True, start_time)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval
    

def time_is_up(start_time, time_limit=0.06):
    return (time.time() - start_time) >= time_limit


def min_max_root(node):
    best_action = None
    best_score = float('-inf')
    depth = 5

    alpha = float('-inf')
    beta = float('inf')

    start_time = time.time()

    depth = 4
    while not time_is_up(start_time):
        alpha = float('-inf')
        beta = float('inf')

        for child in node.compute_and_get_children():
            score = min_max(child, depth - 1, alpha, beta, False, start_time)
            if score > best_score:
                best_score = score
                best_action = child.move

        depth += 1

    return best_action
