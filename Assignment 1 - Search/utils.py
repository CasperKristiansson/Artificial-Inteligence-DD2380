def evaluate_hook_scores(hook_positions, fish_positions, fish_scores):
    def distance(hook_pos, fish_pos):
        return abs(hook_pos[0] - fish_pos[0]) + abs(hook_pos[1] - fish_pos[1])

    hook_scores = {}
    for hook_id, hook_pos in hook_positions.items():
        score = 0
        for fish_id, fish_pos in fish_positions.items():
            dist = distance(hook_pos, fish_pos)
            # Check if any other hook is closer to this fish
            is_closest = True
            for other_hook_id, other_hook_pos in hook_positions.items():
                if other_hook_id != hook_id and distance(other_hook_pos, fish_pos) < dist:
                    is_closest = False
                    break

            if is_closest:
                score += fish_scores[fish_id] / (dist + 1)

        hook_scores[hook_id] = score

    return hook_scores


def min_max(depth, is_maximizing_player, hook_positions, fish_positions, fish_scores):
    if depth == 0:
        evaluated_scores = evaluate_hook_scores(hook_positions, fish_positions, fish_scores)
        # Return the score of hook 0
        return evaluated_scores[0]

    if is_maximizing_player:
        best_score = float('-inf')
        # Generate possible actions for hook 0
        actions = generate_actions(hook_positions[0])
        for action in actions:
            new_hook_positions = hook_positions.copy()
            new_hook_positions[0] = move_hook(hook_positions[0], action)
            # Recursively call min_max for the next depth
            score = min_max(depth - 1, False, new_hook_positions, fish_positions, fish_scores)
            best_score = max(best_score, score)
        return best_score
    else:
        worst_score = float('inf')
        # Consider the moves of all hooks except hook 0
        for hook_id, pos in hook_positions.items():
            if hook_id != 0:
                actions = generate_actions(pos)
                for action in actions:
                    new_hook_positions = hook_positions.copy()
                    new_hook_positions[hook_id] = move_hook(pos, action)
                    # Recursively call min_max for the next depth
                    score = min_max(depth - 1, True, new_hook_positions, fish_positions, fish_scores)
                    worst_score = min(worst_score, score)
        return worst_score


def choose_best_action(hook_positions, fish_positions, fish_scores):
    best_action = None
    best_score = float('-inf')

    actions = generate_actions(hook_positions[0])
    for action in actions:
        new_hook_positions = hook_positions.copy()
        new_hook_positions[0] = move_hook(hook_positions[0], action)
        score = min_max(4, False, new_hook_positions, fish_positions, fish_scores)

        if score > best_score:
            best_score = score
            best_action = (0, action)

    return best_action


def generate_actions(position):
    x, y = position
    actions = ['stay', 'right', 'left', 'up', 'down']
    possible_actions = []

    for action in actions:
        if action == 'stay':
            possible_actions.append(action)
        elif action == 'right' and x < 19:
            possible_actions.append(action)
        elif action == 'left' and x > 0:
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
    return position  # For 'stay' or any invalid action


def move_fish(position, action):
    # This function could be identical to move_hook, assuming fish move similarly.
    # If fish have different movement rules, adjust this function accordingly.
    return move_hook(position, action)
