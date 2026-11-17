class Obstacle:

    def __init__(self, name, width, height):
        self.name = name
        self.width = width
        self.height = height

    def get_occupied_positions(self, x, y):
        positions = []

        for i in range(self.height):
            for j in range(self.width):
                positions.append((x + i, y + j))

        return positions