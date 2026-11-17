from obstacles.obstacle import Obstacle


class Table(Obstacle):

    def __init__(self):
        super().__init__("Mesa", width=2, height=2)