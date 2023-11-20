def evaluate(state):
    score = 0

    score += state.player_scores[0] - state.player_scores[1]
    skip_distance_calculation = [False, False]

    for player in [0, 1]:
        if state.player_caught[player] != -1:
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


def min_max(node, depth, alpha, beta, maximizing_player):
    if depth == 0:
        return evaluate(node.state)

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.compute_and_get_children():
            eval = min_max(child, depth - 1, alpha, beta, False)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.compute_and_get_children():
            eval = min_max(child, depth - 1, alpha, beta, True)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval


def min_max_root(node):
    best_action = None
    best_score = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    depth = 4

    for child in node.compute_and_get_children():
        score = min_max(child, depth - 1, alpha, beta, False)
        if score > best_score:
            best_score = score
            best_action = child.move

    return best_action
