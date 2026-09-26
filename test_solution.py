"""Regression tests for the CrystalRover transition model.

Generative AI declaration: OpenAI Codex (GPT-5, 26 September 2026) helped
design and implement these tests. The author reviewed the output.
"""

import unittest

from game_env import GameEnv
from game_state import GameState
from solution import Solver


class TransitionModelTests(unittest.TestCase):
    def test_transition_probabilities_sum_to_one(self):
        for level in ("L1", "L2", "L3", "L4"):
            with self.subTest(level=level):
                solver = Solver(GameEnv(f"testcases/{level}.txt"))
                solver.vi_initialise()
                for outcomes in solver.transitions.values():
                    self.assertAlmostEqual(
                        sum(probability for probability, _, _ in outcomes),
                        1.0,
                        places=12,
                    )

    def test_invalid_walk_from_crater_is_ignored(self):
        env = GameEnv("testcases/L1.txt")
        solver = Solver(env)
        row, col = env.crater_positions[0]
        state = GameState(row, col, tuple(0 for _ in env.crystal_positions))

        next_state, reward, game_over = solver._apply_dynamics_with_distance(
            state, GameEnv.WALK_RIGHT, 1
        )

        self.assertEqual(next_state, state)
        self.assertEqual(reward, 0.0)
        self.assertFalse(game_over)

    def test_invalid_jump_from_ground_is_ignored(self):
        env = GameEnv("testcases/L1.txt")
        solver = Solver(env)
        state = env.get_init_state()

        next_state, reward, game_over = solver._apply_dynamics_with_distance(
            state, GameEnv.JUMP_RIGHT, 1
        )

        self.assertEqual(next_state, state)
        self.assertEqual(reward, 0.0)
        self.assertFalse(game_over)


if __name__ == "__main__":
    unittest.main()
