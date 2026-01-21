import labmaze
from utils import msw

# https://github.com/google-deepmind/labmaze/
# maze = labmaze.RandomMaze(
#     height=11, width=13, random_seed=42, spawns_per_room=1, objects_per_room=1
# )
# print(maze.entity_layer)


MAZE_LAYOUT = msw(
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
maze = labmaze.FixedMazeWithRandomGoals(
    entity_layer=MAZE_LAYOUT, num_spawns=1, num_objects=1
)
# print(maze.entity_layer)
