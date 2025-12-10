
import labmaze
maze = labmaze.RandomMaze(height=11, width=13, random_seed=42)
print(maze.entity_layer)


MAZE_LAYOUT = """
*********
*********
*********
***   ***
***   ***
***   ***
*********
"""[1:]
maze = labmaze.FixedMazeWithRandomGoals(entity_layer=MAZE_LAYOUT)
print(maze.entity_layer)