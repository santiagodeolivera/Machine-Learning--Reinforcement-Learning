from obstacles.obstacle import Obstacle


class Bed(Obstacle):

    def __init__(self):
        super().__init__("Cama", width=3, height=2)