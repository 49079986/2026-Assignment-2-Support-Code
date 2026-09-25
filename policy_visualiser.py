"""Render CrystalRover values and policy as a grid image.

Generative AI declaration: OpenAI Codex (GPT-5, 25 September 2026) helped
design and implement this visualiser. The author reviewed the output.
"""

import sys

from PIL import Image, ImageDraw, ImageFont

from game_env import GameEnv
from solution import Solver


CELL = 92
MARGIN = 36
ARROWS = {
    "l": "←", "r": "→", "u": "↑", "d": "↓",
}


def _font(size, bold=False):
    names = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_policy(env, values, policy, output_path, crystal_status=None, title=None):
    """Render one crystal-status slice; callable after any VI/PI iteration."""
    if crystal_status is None:
        crystal_status = env.get_init_state().crystal_status

    width = env.n_cols * CELL + 2 * MARGIN
    height = env.n_rows * CELL + 2 * MARGIN + 64
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    heading = title or "CrystalRover value and policy visualiser"
    draw.text((MARGIN, 16), heading, fill="#111827", font=_font(26, True))

    matching = [v for s, v in values.items() if s.crystal_status == crystal_status]
    low = min(matching) if matching else 0.0
    high = max(matching) if matching else 1.0
    span = max(high - low, 1e-9)

    for row in range(env.n_rows):
        for col in range(env.n_cols):
            x0 = MARGIN + col * CELL
            y0 = MARGIN + 64 + row * CELL
            x1, y1 = x0 + CELL, y0 + CELL
            tile = env.grid_data[row][col]
            state = next((s for s in values if s.row == row and s.col == col and
                          s.crystal_status == crystal_status), None)

            if tile == GameEnv.ROCK_TILE:
                fill = "#374151"
            elif tile == GameEnv.LAVA_TILE:
                fill = "#ef4444"
            elif tile == GameEnv.CRATER_TILE:
                fill = "#9ca3af"
            elif state is not None:
                ratio = (values[state] - low) / span
                fill = (int(239 - 150 * ratio), int(246 - 35 * ratio),
                        int(255 - 90 * ratio))
            else:
                fill = "#f3f4f6"

            draw.rectangle((x0, y0, x1, y1), fill=fill, outline="#d1d5db", width=2)
            if (row, col) in env.launch_positions:
                draw.text((x0 + 7, y0 + 5), "EXIT", fill="#065f46", font=_font(13, True))
            if (row, col) in env.crystal_positions:
                draw.text((x1 - 21, y0 + 5), "C", fill="#0369a1", font=_font(15, True))
            if (row, col) == (env.init_row, env.init_col):
                draw.text((x0 + 7, y0 + 5), "START", fill="#7c2d12", font=_font(11, True))

            if state is not None:
                draw.text((x0 + 7, y0 + 29), f"{values[state]:.1f}",
                          fill="#111827", font=_font(15))
                action = policy.get(state)
                if action:
                    arrow = ARROWS[action[-1]]
                    draw.text((x1 - 34, y1 - 40), arrow, fill="#111827", font=_font(31, True))
                    draw.text((x0 + 7, y1 - 24), action, fill="#111827", font=_font(12, True))

    image.save(output_path)


def main(args):
    if len(args) != 2:
        raise SystemExit("Usage: python policy_visualiser.py TESTCASE OUTPUT.png")
    env = GameEnv(args[0])
    solver = Solver(env)
    solver.vi_plan_offline()
    render_policy(env, solver.vi_values, solver.vi_policy, args[1],
                  title="L1 converged VI values and policy")


if __name__ == "__main__":
    main(sys.argv[1:])
