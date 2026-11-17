class Robot:

    MAX_BATTERY = 100

    def __init__(self, position=(0, 0)):
        self.position = position
        self.battery = self.MAX_BATTERY

    def spend_battery(self, amount=1):
        # La batería nunca baja de 0
        self.battery = max(0, self.battery - amount)

    def move(self, new_position):
        self.position = new_position
        self.spend_battery()

    def clean(self):
        self.spend_battery()

    def recharge(self):
        self.battery = self.MAX_BATTERY
