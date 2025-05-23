import numpy as np


class Game:
    def __init__(self):
        self.current_number = 0

    def play(self, take_smaller_number):
        if take_smaller_number:
            self.current_number += 1
        else:
            self.current_number += 2
        return self.current_number


class TrainablePlayer:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        self.policy = np.ones(10) * 0.5

    def train(self):
        mygame = Game()
        state = mygame.current_number
        game_record = list()
        while state < 10:
            random_number = np.random.rand()
            if random_number < self.policy[state]:
                take_smaller_number = True
            else:
                take_smaller_number = False
            game_record.append((state, take_smaller_number))
            state = mygame.play(take_smaller_number)

            # if we won
            if state == 10:
                for move in game_record:
                    if move[1]:  # took smaller number
                        self.policy[move[0]] = min(1, self.policy[move[0]] + self.learning_rate)
                    else:
                        self.policy[move[0]] = max(0, self.policy[move[0]] - self.learning_rate)
                break

            # random playing opponent
            random_number = np.random.rand()
            if random_number < 0.5:
                take_smaller_number = True
            else:
                take_smaller_number = False
            state = mygame.play(take_smaller_number)

            # if we lost
            if state == 10:
                for move in game_record:
                    if move[1]:  # took smaller number
                        self.policy[move[0]] = min(1, self.policy[move[0]] - self.learning_rate)
                    else:
                        self.policy[move[0]] = max(0, self.policy[move[0]] + self.learning_rate)
                break


def main():
    myplayer = TrainablePlayer()
    for i in range(1000):
        myplayer.train()

    for i in range(len(myplayer.policy)):
        print(f"i: {i}, policy: {myplayer.policy[i]}")


if __name__ == "__main__":
    main()
