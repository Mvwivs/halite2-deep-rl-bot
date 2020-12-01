from math import floor, sqrt


class Grid:
    def __init__(self, num_tiles: int, height: float, width: float) -> None:
        self.num_tiles = num_tiles
        self.map_height = height
        self.map_width = width
        self.tile_height = height / floor(sqrt(self.num_tiles))
        self.tile_width = self.tile_height * 1.5
        self.grid_width = int(self.map_width / self.tile_width)
        self.grid_height = int(self.map_height / self.tile_height)

    def get_tile_id(self, x, y) -> int:
        tile_x, tile_y = self.to_grid_coord(x, y)
        return floor(tile_x + self.grid_height * tile_y)

    def from_grid_coords(self, x, y):
        return x * self.tile_width, y * self.tile_height

    def to_grid_coord(self, x, y) -> (int, int):
        tile_x = floor(x / self.tile_width)
        tile_y = floor(y / self.tile_height)
        return tile_x, tile_y

    def id_to_grid_coord(self, tile_id):
        y = floor(tile_id / self.grid_height)
        x = tile_id - y * self.grid_height
        return x, y

    def get_tile_center(self, g_x, g_y):
        return g_x * self.tile_width + self.tile_width / 2, g_y * self.tile_height + self.tile_height / 2,

    def get_tile_center_by_id(self, tile_id):
        x, y = self.id_to_grid_coord(tile_id)
        return self.get_tile_center(x, y)

    def __repr__(self):
        return f'{self.num_tiles=}\n{self.map_width=}\n{self.map_height=}\n{self.tile_width=}\n' \
               f'{self.tile_height=}\n{self.grid_height=}\n{self.grid_width=}'
