#!/usr/bin/env python3
import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from utils import choose_best_action

with open("moves.txt", "w") as file:
    file.write("")


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        children = initial_tree_node.compute_and_get_children()
        print(children)

        hook_positions = initial_tree_node.state.hook_positions
        fish_positions = initial_tree_node.state.fish_positions
        fish_scores = initial_tree_node.state.fish_scores
        player_caught = initial_tree_node.state.player_caught


        # print("hook_positions: ", hook_positions)
        # print("fish_positions: ", fish_positions)
        # print("fish_scores: ", fish_scores)

        # print("player_caught: ", player_caught)

        # if player_caught[0] != -1:
        #     print(70*"*")
        #     return "up"

        # a = choose_best_action(hook_positions, fish_positions, fish_scores)

        action = choose_best_action(hook_positions, fish_positions, fish_scores,player_caught)

        with open("moves.txt", "a") as file:
            file.write(action + "\n")

        return action

        random_move = random.randrange(5)
        return ACTION_TO_STR[random_move]
