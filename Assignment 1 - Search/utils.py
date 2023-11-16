def evaluate_hook_positions(hook_positions, fish_positions, fish_scores):
    def manhattan_distance(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    scores = {0: 0, 1: 0}

    for hook_index, hook_pos in hook_positions.items():
        min_distance = float('inf')
        closest_fish_score = 0

        for fish_index, fish_pos in fish_positions.items():
            distance = manhattan_distance(hook_pos, fish_pos)

            if distance < min_distance:
                min_distance = distance
                closest_fish_score = fish_scores[fish_index]

        scores[hook_index] = closest_fish_score - min_distance

    if hook_positions[0][0] == hook_positions[1][0] - 1 or hook_positions[0][0] == hook_positions[1][0] + 1:
        scores[0] += 5

    score_difference = scores[0] - scores[1]
    return score_difference


def min_max(depth, is_maximizing_player, hook_positions, fish_positions, fish_scores):
    if depth == 0:
        return evaluate_hook_positions(hook_positions, fish_positions, fish_scores)

    if is_maximizing_player:
        best_score = float('-inf')
        actions = generate_actions(hook_positions, 1)
        for action in actions:
            new_hook_positions = hook_positions.copy()
            new_hook_positions[0] = move_hook(hook_positions[0], action)
            score = min_max(depth - 1, False, new_hook_positions, fish_positions, fish_scores)
            best_score = max(best_score, score)
        return best_score
    else:
        worst_score = float('inf')
        actions = generate_actions(hook_positions, 0)
        for action in actions:
            new_hook_positions = hook_positions.copy()
            new_hook_positions[1] = move_hook(hook_positions[1], action)
            score = min_max(depth - 1, True, new_hook_positions, fish_positions, fish_scores)
            worst_score = min(worst_score, score)
        return worst_score


def choose_best_action(hook_positions, fish_positions, fish_scores):
    best_action = None
    best_score = float('-inf')

    actions = generate_actions(hook_positions, 1)
    for action in actions:
        new_hook_positions = hook_positions.copy()
        new_hook_positions[0] = move_hook(hook_positions[0], action)
        score = min_max(4, False, new_hook_positions, fish_positions, fish_scores)

        if score > best_score:
            best_score = score
            best_action = (0, action)

    return best_action


def generate_actions(hook_positions, opponent_hook_index):
    if opponent_hook_index == 1:
        opponent_x, _ = hook_positions[1]
        x, y = hook_positions[0]
    else:
        opponent_x, _ = hook_positions[0]
        x, y = hook_positions[1]

    actions = ['right', 'left', 'up', 'down']
    possible_actions = []

    for action in actions:
        if action == 'right' and x < 19 and not x == opponent_x + 1:
            possible_actions.append(action)
        elif action == 'left' and x > 0 and not x == opponent_x - 1:
            possible_actions.append(action)
        elif action == 'up' and y < 19:
            possible_actions.append(action)
        elif action == 'down' and y > 0:
            possible_actions.append(action)

    return possible_actions


def move_hook(position, action):
    x, y = position
    if action == 'right':
        return x + 1, y
    elif action == 'left':
        return x - 1, y
    elif action == 'up':
        return x, y + 1
    elif action == 'down':
        return x, y - 1


def move_fish(position, action):
    return move_hook(position, action)
