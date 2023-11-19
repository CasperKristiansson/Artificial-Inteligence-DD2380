def minimax(node, depth, maximizing_player):
    # Base case
    if depth == 0:
        return node.state.player_scores[node.state.player]

    if maximizing_player:
        max_eval = float('-inf')
        for child in node.compute_and_get_children():
            eval = minimax(child, depth - 1, False)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.compute_and_get_children():
            eval = minimax(child, depth - 1, True)
            min_eval = min(min_eval, eval)
        return min_eval

def minimax_root(node):
    best_action = None
    best_score = float('-inf')
    depth = 5

    for child in node.compute_and_get_children():
        score = minimax(child, depth - 1, False)
        if score > best_score:
            best_score = score
            best_action = child.move

    return best_action
