# Gym-maze format:
"""
The data structure to represent the mazes is a list of lists (`list[list]`) that contains the encoding of the discrete cell positions `(i,j)` of the maze. Each list inside the main list represents a row `i` of the maze, while the elements of the row are the intersections with the column index `j`.

    The cell encoding can have 5 different values:
    * `1: int` - Indicates that there is a wall in this cell.
    * `0: int` - Indicates that this cell is free for the agent and goal.
    * `"g": str` - Indicates that this cell can contain a goal. There can be multiple goals in the same maze and one of them will be randomly selected when the environment is reset.
    * `"r": str` - Indicates cells in which the agent can be initialized in when the environment is reset.
    * `"c": str` - Stands for combined cell and indicates that this cell can be initialized as a goal or agent reset location.

    Note that if all the empty cells are given a value of `0` and there are no cells in the map representation with values `"g"`, `"r"`, or `"c"`, the initial goal and reset locations will be randomly chosen from the empty cells with value `0`.

    Also, the maze data structure is discrete. However the observations are continuous and variance is added to the goal and the agent's initial pose by adding a sammpled noise from a uniform distribution to the cell's `(x,y)` coordinates in the MuJoCo simulation.
"""

# get gym-maze format from string representation


def msw(string):
    # Multiline string wrapper: removes leading newline if present
    if string.startswith("\n"):
        return string[1:]
    return string


def parse_maze_from_string(layout_str):
    maze = []
    for line in layout_str.strip().splitlines():
        row = []
        for char in line:
            if char == "*":
                row.append(1)  # Wall
            elif char == " ":
                row.append(0)  # Free space
            else:
                raise ValueError(f"Unexpected character '{char}' in maze layout.")
        maze.append(row)
    return maze


def parse_string_from_maze(maze):
    layout_str = ""
    for row in maze:
        for cell in row:
            if cell == 1:
                layout_str += "*"
            elif cell == 0:
                layout_str += " "
            else:
                raise ValueError(f"Unexpected cell value '{cell}' in maze.")
        layout_str += "\n"
    return layout_str


if __name__ == "__main__":
    TEST_MAZE_LAYOUT = msw(
        """
*************
*   *       *
*   *   *****
*           *
**  **  **  *
*           *
*   *****   *
*   *****   *
*   *****   *
*   *****   *
*************
"""
    )
    maze_format = parse_maze_from_string(TEST_MAZE_LAYOUT)
    print(maze_format)
    print(parse_string_from_maze(maze_format))
