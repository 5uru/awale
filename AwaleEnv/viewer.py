from IPython.display import SVG
from typing import List, Tuple


def generate_seed_positions(num_seeds: int) -> List[Tuple[int, int]]:
    """
    Generate positions for seeds based on count.
    """
    if num_seeds <= 4:
        return [(0, -10), (-10, 0), (10, 0), (0, 10)][:num_seeds]
    elif num_seeds <= 8:
        return [
            (0, -10),
            (-10, -10),
            (10, -10),
            (-10, 0),
            (10, 0),
            (-10, 10),
            (0, 10),
            (10, 10),
        ][:num_seeds]
    else:
        return [
            (0, -12),
            (-6, -10),
            (6, -10),
            (-10, -6),
            (10, -6),
            (-12, 0),
            (12, 0),
            (-10, 6),
            (10, 6),
            (-6, 10),
            (6, 10),
            (0, 12),
        ][:num_seeds]


def generate_pit_svg(cx: int, cy: int, seeds: int) -> str:
    """
    Generate SVG for a single pit and its seeds.
    """
    pit_svg = [
        f"""
    <g>
        <circle cx="{cx}" cy="{cy}" r="30" fill="#E5BA73" stroke="#7D5A50" stroke-width="3"/>"""
    ]

    if seeds > 0:
        positions = generate_seed_positions(seeds)
        seeds_svg = [
            f'        <circle cx="{cx + dx}" cy="{cy + dy}" r="6" fill="#8B4513"/>'
            for dx, dy in positions
        ]
        pit_svg.extend(seeds_svg)

    # Add seed count text
    pit_svg.append(
        f'        <text x="{cx}" y="{cy+5}" font-family="Arial" '
        f'font-size="16" fill="#7D5A50" text-anchor="middle">{seeds}</text>'
    )
    pit_svg.append("    </g>")

    return "\n".join(pit_svg)


def generate_svg_content(board: List[int], scores: List[int]) -> str:
    """
    Generate the SVG content as a string.
    """
    # Define pit positions
    pit_positions = [
        (500, 150),
        (420, 150),
        (340, 150),
        (260, 150),
        (180, 150),
        (100, 150),  # Top row
        (100, 250),
        (180, 250),
        (260, 250),
        (340, 250),
        (420, 250),
        (500, 250),  # Bottom row
    ]

    # Generate all pits
    pits_svg = [
        generate_pit_svg(cx, cy, seeds) for (cx, cy), seeds in zip(pit_positions, board)
    ]

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg viewBox="0 0 600 400" xmlns="http://www.w3.org/2000/svg">
    <!-- Background -->
    <rect width="600" height="400" fill="#F9F5EB" rx="20"/>
    
    <!-- Board -->
    <rect x="50" y="100" width="500" height="200" fill="#B4846C" rx="15"/>
    
    <!-- Pits and Seeds -->
    {''.join(pits_svg)}
    
    <!-- Player Labels and Scores -->
    <!-- Player 2 -->
    <rect x="220" y="50" width="160" height="35" fill="#7D5A50" rx="10"/>
    <text x="230" y="75" font-family="Arial" font-size="24" fill="#F9F5EB" font-weight="bold">Player 2</text>
    
    <!-- Player 2 Score -->
    <rect x="500" y="50" width="60" height="35" fill="#7D5A50" rx="10"/>
    <text x="515" y="75" font-family="Arial" font-size="24" fill="#F9F5EB" font-weight="bold">{scores[1]}</text>
    
    <!-- Player 1 -->
    <rect x="220" y="315" width="160" height="35" fill="#7D5A50" rx="10"/>
    <text x="230" y="340" font-family="Arial" font-size="24" fill="#F9F5EB" font-weight="bold">Player 1</text>
    
    <!-- Player 1 Score -->
    <rect x="500" y="315" width="60" height="35" fill="#7D5A50" rx="10"/>
    <text x="515" y="340" font-family="Arial" font-size="24" fill="#F9F5EB" font-weight="bold">{scores[0]}</text>
</svg>"""


def render_board(board: List[int], scores: List[int]) -> SVG:
    """
    Render the Awale board state as an SVG displayable in a Jupyter notebook.

    Args:
        board: List of 12 integers representing seeds in each pit
        scores: List of 2 integers representing player scores

    Returns:
        IPython.display.SVG: SVG object for notebook display
    """
    svg_content = generate_svg_content(board, scores)
    return SVG(data=svg_content)


def save_board_svg(board: List[int], scores: List[int], filename: str) -> None:
    """
    Save the board state as an SVG file.

    Args:
        board: List of 12 integers representing seeds in each pit
        scores: List of 2 integers representing player scores
        filename: Name of the file to save
    """
    svg_content = generate_svg_content(board, scores)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)
