"""This module defines a random players baseline
"""

from poke_env.environment import AbstractBattle
from poke_env.player.battle_order import BattleOrder
from poke_env.player.player import Player


class RandomPlayer(Player):
    moves = {}
    moves_discovered = 0

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        tmp_moves = {}
        tmp_moves_discovered = 0
        for pok in battle.team.values():
            tmp_moves[pok.species] = set(move.id for move in pok.moves.values())
            tmp_moves_discovered = tmp_moves_discovered + len(pok.moves.values())
        
        if tmp_moves != self.moves:
            self.moves = tmp_moves
            self.moves_discovered = tmp_moves_discovered
            print("=== New moves discovered! ===")
            print(f"Total moves discovered: {self.moves_discovered}")
            for species, moves in self.moves.items():
                print(f"{species}: {', '.join(moves)}")

        action = self.choose_random_move(battle)
        print(f"RandomPlayer {action}")
        return action
