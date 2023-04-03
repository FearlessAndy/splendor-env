from typing import Dict, Optional, Tuple, Union

import gymnasium
import numpy as np


# class Cards():
#     spend_blue = 0
#     spend_red = 0
#     spend_green = 0
#     spend_black = 0
#     spend_white = 0
#     type = 0 # 0-blue, 1-red, 2-green, 3-black, 4-white
#     score = 0
#     def __init__(self, spend_blue, spend_red, spend_green, spend_black, spend_white, type, score):
#         self.spend_blue = spend_blue
#         self.spend_red = spend_red
#         self.spend_green = spend_green
#         self.spend_black = spend_black
#         self.spend_white = spend_white
#         self.type = type
#         self.score = score
#
# class Nobles():
#     spend_blue = 0
#     spend_red = 0
#     spend_green = 0
#     spend_black = 0
#     spend_white = 0
#     score = 0
#
#     def __init__(self, spend_blue, spend_red, spend_green, spend_black, spend_white, score):
#         self.spend_blue = spend_blue
#         self.spend_red = spend_red
#         self.spend_green = spend_green
#         self.spend_black = spend_black
#         self.spend_white = spend_white
#         self.score = score


class SplendorEnvSimple(gymnasium.Env):
    metadata = {"render_modes": ["human"], "name": "splendor_v0"}

    low_banquet = [[1, 0, 1, 1, 1, 1, 0],
                   [1, 2, 0, 2, 0, 2, 0],
                   [3, 0, 0, 0, 0, 4, 0],
                   [0, 3, 0, 0, 0, 2, 0],
                   [4, 0, 0, 0, 0, 3, 1],
                   [0, 0, 3, 0, 0, 3, 0],
                   [2, 1, 0, 0, 2, 3, 0],
                   [0, 3, 1, 1, 0, 3, 0],
                   [1, 1, 1, 1, 0, 4, 0],
                   [0, 0, 0, 0, 4, 1, 1],
                   [1, 1, 2, 1, 0, 4, 0],
                   [1, 0, 0, 1, 3, 4, 0],
                   [0, 1, 2, 0, 0, 3, 0],
                   [0, 1, 0, 3, 1, 1, 0],
                   [1, 1, 0, 2, 1, 2, 0],
                   [0, 0, 0, 3, 0, 0, 0],
                   [0, 0, 0, 4, 0, 2, 1],
                   [3, 0, 1, 0, 1, 2, 0],
                   [0, 2, 0, 1, 0, 4, 0],
                   [2, 0, 2, 1, 0, 4, 0],
                   [0, 0, 0, 0, 3, 1, 0],
                   [1, 1, 1, 0, 1, 3, 0],
                   [0, 0, 2, 2, 0, 0, 0],
                   [0, 0, 1, 2, 2, 1, 0],
                   [1, 1, 0, 1, 1, 2, 0],
                   [0, 4, 0, 0, 0, 0, 1],
                   [0, 2, 0, 0, 2, 1, 0],
                   [2, 0, 1, 0, 0, 1, 0],
                   [0, 0, 2, 0, 2, 3, 0],
                   [0, 0, 0, 2, 1, 0, 0],
                   [2, 1, 1, 0, 1, 3, 0],
                   [1, 0, 1, 1, 2, 1, 0],
                   [0, 1, 1, 1, 1, 0, 0],
                   [0, 2, 1, 1, 1, 0, 0],
                   [2, 0, 0, 2, 0, 4, 0],
                   [2, 2, 0, 0, 0, 2, 0],
                   [0, 0, 4, 0, 0, 4, 1],
                   [1, 0, 0, 0, 2, 2, 0],
                   [0, 2, 2, 0, 1, 0, 0],
                   [1, 1, 3, 0, 0, 0, 0]]
    low_secret = [0]

    middle_banquet = [[5, 0, 0, 0, 0, 0, 2],
                      [6, 0, 0, 0, 0, 0, 3],
                      [0, 0, 5, 0, 0, 2, 2],
                      [3, 0, 0, 2, 2, 2, 1],
                      [0, 0, 3, 2, 3, 3, 1],
                      [0, 2, 0, 3, 2, 1, 1],
                      [1, 2, 4, 0, 0, 3, 2],
                      [0, 0, 0, 0, 5, 3, 2],
                      [2, 0, 2, 0, 3, 3, 1],
                      [0, 5, 0, 3, 0, 4, 2],
                      [3, 3, 0, 0, 2, 4, 1],
                      [0, 3, 5, 0, 0, 3, 2],
                      [0, 0, 0, 5, 3, 1, 2],
                      [2, 3, 2, 0, 0, 0, 1],
                      [0, 6, 0, 0, 0, 1, 3],
                      [0, 1, 0, 4, 2, 0, 2],
                      [0, 0, 6, 0, 0, 2, 3],
                      [0, 0, 0, 5, 0, 1, 2],
                      [5, 0, 3, 0, 0, 2, 2],
                      [0, 0, 0, 6, 0, 3, 3],
                      [3, 2, 0, 3, 0, 1, 1],
                      [3, 0, 0, 0, 5, 0, 2],
                      [0, 3, 2, 0, 3, 2, 1],
                      [0, 2, 3, 2, 0, 4, 1],
                      [0, 5, 0, 0, 0, 4, 2],
                      [0, 0, 0, 0, 6, 4, 3],
                      [4, 0, 2, 0, 1, 1, 2],
                      [2, 0, 0, 1, 4, 2, 2],
                      [2, 0, 3, 3, 0, 0, 1],
                      [0, 4, 1, 2, 0, 4, 2]]
    middle_secret = [0]

    high_banquet = [[3, 3, 5, 0, 3, 3, 3],
                    [0, 0, 0, 0, 7, 0, 4],
                    [5, 0, 3, 3, 3, 1, 3],
                    [3, 5, 3, 3, 0, 4, 3],
                    [3, 3, 0, 3, 5, 2, 3],
                    [0, 3, 0, 6, 3, 4, 4],
                    [0, 3, 3, 5, 3, 0, 3],
                    [0, 6, 3, 3, 0, 3, 4],
                    [3, 3, 6, 0, 0, 1, 4],
                    [0, 7, 0, 0, 0, 3, 4],
                    [0, 0, 0, 7, 3, 4, 5],
                    [0, 7, 0, 3, 0, 3, 5],
                    [0, 0, 7, 0, 0, 1, 4],
                    [6, 0, 3, 0, 3, 2, 4],
                    [7, 0, 3, 0, 0, 2, 5],
                    [0, 3, 7, 0, 0, 1, 5],
                    [3, 0, 0, 3, 6, 0, 4],
                    [3, 0, 0, 0, 7, 0, 5],
                    [0, 0, 0, 7, 0, 4, 4],
                    [7, 0, 0, 0, 0, 2, 4]]
    high_secret = [0]

    noble_banquet = [[0, 4, 0, 4, 0, 3],
                     [3, 3, 3, 0, 0, 3],
                     [3, 0, 0, 3, 3, 3],
                     [0, 3, 3, 3, 0, 3],
                     [0, 3, 0, 3, 3, 3],
                     [0, 0, 0, 4, 4, 3],
                     [4, 0, 0, 0, 4, 3],
                     [3, 0, 3, 0, 3, 3],
                     [4, 0, 4, 0, 0, 3],
                     [0, 4, 4, 0, 0, 3]]
    noble_usable = [0]

    public_money = [5, 5, 5, 5, 5, 5]  # 0-blue, 1-red, 2-green, 3-black, 4-white, 5- gold
    agent_money = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    agent_card = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # 0-blue, 1-red, 2-green, 3-black, 4-white
    agent_book = [[1, 1, 1, 1, 0, 0, 0],
                  [1, 1, 1, 1, 0, 0, 0]]
    agent_score = [0, 0]
    agent_role = 0

    low_public = [[0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]]

    middle_public = [[0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0]]

    high_public = [[0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0, 0, 0]]

    def __init__(self) -> None:
        self.action_space = gymnasium.spaces.Discrete(41)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, shape=(129,), dtype=np.int32)

    def _get_observation(self):
        result = []
        for i in range(2):
            for j in range(6):
                result.append(self.agent_money[i][j])
            for j in range(5):
                result.append(self.agent_card[i][j])
            result.append(self.agent_score[i])
        for i in range(6):
            result.append(self.public_money[i])
        for i in range(2):
            for j in range(7):
                result.append(self.agent_book[i][j])
        for i in range(4):
            for j in range(7):
                result.append(self.low_public[i][j])
        for i in range(4):
            for j in range(7):
                result.append(self.middle_public[i][j])
        for i in range(4):
            for j in range(7):
                result.append(self.high_public[i][j])
        result.append(self.agent_role)

        return np.array(result)

    def step(self, action):
        action_mask = self.update_mask()
        assert action_mask[action] == 1
        reward, terminated = self.update_state(action)
        obs = self._get_observation()
        action_mask = self.update_mask()
        return obs, reward, terminated, action_mask

    def purchase(self, card) -> None:
        for i in range(5):
            total = self.agent_money[self.agent_role][i] + self.agent_card[self.agent_role][i]
            if total >= card[i]:
                self.agent_money[self.agent_role][i] = self.agent_money[self.agent_role][i] - max(0, card[i] -
                                                                                                  self.agent_card[
                                                                                                      self.agent_role][
                                                                                                      i])
                self.public_money[i] = self.public_money[i] + max(0, card[i] - self.agent_card[self.agent_role][i])
            else:
                self.public_money[i] += self.agent_money[self.agent_role][i]
                self.agent_money[self.agent_role][i] = 0
                self.agent_money[self.agent_role][5] = self.agent_money[self.agent_role][5] - (card[i] - total)
                self.public_money[5] = self.public_money[5] + card[i] - total
        self.agent_score[self.agent_role] += card[6]
        self.agent_card[self.agent_role][card[5]] += 1

    def refresh_low(self, pos) -> None:
        if len(self.low_secret) == 0:
            for i in range(7):
                self.low_public[pos][i] = 0
            return
        taken = self.low_secret[np.random.randint(len(self.low_secret))]
        for i in range(7):
            self.low_public[pos][i] = self.low_banquet[taken][i]
        self.low_secret.remove(taken)

    def refresh_middle(self, pos) -> None:
        if len(self.middle_secret) == 0:
            for i in range(7):
                self.middle_public[pos][i] = 0
            return
        taken = self.middle_secret[np.random.randint(len(self.middle_secret))]
        for i in range(7):
            self.middle_public[pos][i] = self.middle_banquet[taken][i]
        self.middle_secret.remove(taken)

    def refresh_high(self, pos) -> None:
        if len(self.high_secret) == 0:
            for i in range(7):
                self.high_public[pos][i] = 0
            return
        taken = self.high_secret[np.random.randint(len(self.high_secret))]
        for i in range(7):
            self.high_public[pos][i] = self.high_banquet[taken][i]
        self.high_secret.remove(taken)

    def get_total(self):
        total = 0
        for i in range(6):
            total = total + self.agent_money[self.agent_role][i]
        return total

    def earnMoney(self, first, second, third):
        if self.public_money[first] > 0 and self.get_total() < 10:
            self.public_money[first] -= 1
            self.agent_money[self.agent_role][first] += 1
        if self.public_money[second] > 0 and self.get_total() < 10:
            self.public_money[second] -= 1
            self.agent_money[self.agent_role][second] += 1
        if self.public_money[third] > 0 and self.get_total() < 10:
            self.public_money[third] -= 1
            self.agent_money[self.agent_role][third] += 1

    # TO-DO: REWARD?
    def update_state(self, action):
        reward = 0
        terminated = False

        if action == 0:
            self.purchase(self.low_public[0])
            reward += 1
            reward += 10 * self.low_public[0][6]
            self.refresh_low(0)
        elif action == 1:
            self.purchase(self.low_public[1])
            reward += 1
            reward += 10 * self.low_public[1][6]
            self.refresh_low(1)
        elif action == 2:
            self.purchase(self.low_public[2])
            reward += 1
            reward += 10 * self.low_public[2][6]
            self.refresh_low(2)
        elif action == 3:
            self.purchase(self.low_public[3])
            reward += 1
            reward += 10 * self.low_public[3][6]
            self.refresh_low(3)
        elif action == 4:
            self.purchase(self.middle_public[0])
            reward += 1
            reward += 10 * self.middle_public[0][6]
            self.refresh_middle(0)
        elif action == 5:
            self.purchase(self.middle_public[1])
            reward += 1
            reward += 10 * self.middle_public[1][6]
            self.refresh_middle(1)
        elif action == 6:
            self.purchase(self.middle_public[2])
            reward += 1
            reward += 10 * self.middle_public[2][6]
            self.refresh_middle(2)
        elif action == 7:
            self.purchase(self.middle_public[3])
            reward += 1
            reward += 10 * self.middle_public[3][6]
            self.refresh_middle(3)
        elif action == 8:
            self.purchase(self.high_public[0])
            reward += 1
            reward += 10 * self.high_public[0][6]
            self.refresh_high(0)
        elif action == 9:
            self.purchase(self.high_public[1])
            reward += 1
            reward += 10 * self.high_public[1][6]
            self.refresh_high(1)
        elif action == 10:
            self.purchase(self.high_public[2])
            reward += 1
            reward += 10 * self.high_public[2][6]
            self.refresh_high(2)
        elif action == 11:
            self.purchase(self.high_public[3])
            reward += 1
            reward += 10 * self.high_public[3][6]
            self.refresh_high(3)

        elif action == 12:
            self.purchase(self.agent_book[self.agent_role])
            reward += 1
            reward += 10 * self.agent_book[self.agent_role][6]

        elif action == 13:
            self.agent_money[self.agent_role][0] += 2
            self.public_money[0] -= 2
        elif action == 14:
            self.agent_money[self.agent_role][1] += 2
            self.public_money[1] -= 2
        elif action == 15:
            self.agent_money[self.agent_role][2] += 2
            self.public_money[2] -= 2
        elif action == 16:
            self.agent_money[self.agent_role][3] += 2
            self.public_money[3] -= 2
        elif action == 17:
            self.agent_money[self.agent_role][4] += 2
            self.public_money[4] -= 2

        elif action == 18:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
        elif action == 19:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.low_public[0]
            self.refresh_low(0)
        elif action == 20:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.low_public[1]
            self.refresh_low(1)
        elif action == 21:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.low_public[2]
            self.refresh_low(2)
        elif action == 22:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.low_public[3]
            self.refresh_low(3)
        elif action == 23:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.middle_public[0]
            self.refresh_middle(0)
        elif action == 24:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.middle_public[1]
            self.refresh_middle(1)
        elif action == 25:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.middle_public[2]
            self.refresh_middle(2)
        elif action == 26:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.middle_public[3]
            self.refresh_middle(3)
        elif action == 27:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.high_public[0]
            self.refresh_high(0)
        elif action == 28:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.high_public[1]
            self.refresh_high(1)
        elif action == 29:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.high_public[2]
            self.refresh_high(2)
        elif action == 30:
            self.agent_money[self.agent_role][5] += 1
            self.public_money[5] -= 1
            self.agent_book[self.agent_role] = self.high_public[3]
            self.refresh_high(3)

        elif action == 31:
            self.earnMoney(0, 1, 2)
        elif action == 32:
            self.earnMoney(0, 1, 3)
        elif action == 33:
            self.earnMoney(0, 1, 4)
        elif action == 34:
            self.earnMoney(0, 2, 3)
        elif action == 35:
            self.earnMoney(0, 2, 4)
        elif action == 36:
            self.earnMoney(0, 3, 4)
        elif action == 37:
            self.earnMoney(1, 2, 3)
        elif action == 38:
            self.earnMoney(1, 2, 4)
        elif action == 39:
            self.earnMoney(1, 3, 4)
        elif action == 40:
            self.earnMoney(2, 3, 4)
        
        
        noble_temp = self.noble_usable.copy()
        for i in range(len(self.noble_usable)):
            noble_id = noble_temp[i]
            flag = True
            for j in range(5):
                if self.noble_banquet[noble_id][j] > self.agent_card[self.agent_role][j]:
                    flag = False
                    break
            if flag:
                self.noble_usable.remove(noble_id)
                self.agent_score[self.agent_role] += self.noble_banquet[noble_id][5]
                reward += 10 * self.noble_banquet[noble_id][5]

        for i in range(2):
            if self.agent_score[self.agent_role] >= 15:
                terminated = True
                reward += 100
        self.agent_role = (self.agent_role + 1) % 2
        return reward, terminated

    def judge_low(self, pos):
        total = 0
        for i in range(7):
            total = total + self.low_public[pos][i]
        if total == 0:
            return 0
        loss = 0
        for i in range(5):
            loss = loss + max(0, self.low_public[pos][i] - (
                    self.agent_money[self.agent_role][i] + self.agent_card[self.agent_role][i]))
        if loss > self.agent_money[self.agent_role][5]:
            return 0
        return 1

    def judge_middle(self, pos):
        total = 0
        for i in range(7):
            total = total + self.middle_public[pos][i]
        if total == 0:
            return 0
        loss = 0
        for i in range(5):
            loss = loss + max(0, self.middle_public[pos][i] - (
                    self.agent_money[self.agent_role][i] + self.agent_card[self.agent_role][i]))
        if loss > self.agent_money[self.agent_role][5]:
            return 0
        return 1

    def judge_high(self, pos):
        total = 0
        for i in range(7):
            total = total + self.high_public[pos][i]
        if total == 0:
            return 0
        loss = 0
        for i in range(5):
            loss = loss + max(0, self.high_public[pos][i] - (
                    self.agent_money[self.agent_role][i] + self.agent_card[self.agent_role][i]))
        if loss > self.agent_money[self.agent_role][5]:
            return 0
        return 1

    def judge_book(self):
        total = 0
        for i in range(7):
            total = total + self.agent_book[self.agent_role][i]
        if total == 0:
            return 0
        loss = 0
        for i in range(5):
            loss = loss + max(0, self.agent_money[self.agent_role][i] + self.agent_card[self.agent_role][i] -
                              self.agent_book[self.agent_role][i])
        if loss > self.agent_money[self.agent_role][5]:
            return 0
        return 1

    def judge_double(self, kind):
        if self.public_money[kind] >= 4:
            return 1
        return 0

    def judge_book_low(self, pos):
        if self.public_money[5] == 0 or self.get_total() >= 10:
            return 0
        total = 0
        for i in range(7):
            total = total + self.low_public[pos][i]
        if total == 0:
            return 0
        return 1

    def judge_book_middle(self, pos):
        if self.public_money[5] == 0 or self.get_total() >= 10:
            return 0
        total = 0
        for i in range(7):
            total = total + self.middle_public[pos][i]
        if total == 0:
            return 0
        return 1

    def judge_book_high(self, pos):
        if self.public_money[5] == 0 or self.get_total() >= 10:
            return 0
        total = 0
        for i in range(7):
            total = total + self.high_public[pos][i]
        if total == 0:
            return 0
        return 1

    def update_mask(self):
        action_mask = []
        for i in range(4):
            action_mask.append(self.judge_low(i))
        for i in range(4):
            action_mask.append(self.judge_middle(i))
        for i in range(4):
            action_mask.append(self.judge_high(i))
        action_mask.append(self.judge_book())
        for i in range(5):
            action_mask.append(self.judge_double(i))
        if self.public_money[5] > 0 and self.get_total() < 10:
            action_mask.append(1)
        else:
            action_mask.append(0)
        for i in range(4):
            action_mask.append(self.judge_book_low(i))
        for i in range(4):
            action_mask.append(self.judge_book_middle(i))
        for i in range(4):
            action_mask.append(self.judge_book_high(i))
        for i in range(10):
            action_mask.append(1)
        return action_mask

    def sample_action(self, action_mask) -> int:
        length = len(action_mask)
        sample_list = []
        for i in range(length):
            if action_mask[i] == 1:
                sample_list.append(i)
        return sample_list[np.random.randint(len(sample_list))]

    def reset(self, seed=None, options=None):
        """Resets the environment (starts a new game)."""
        super().reset(seed=seed)

        self.low_banquet = [[1, 0, 1, 1, 1, 1, 0],
                            [1, 2, 0, 2, 0, 2, 0],
                            [3, 0, 0, 0, 0, 4, 0],
                            [0, 3, 0, 0, 0, 2, 0],
                            [4, 0, 0, 0, 0, 3, 1],
                            [0, 0, 3, 0, 0, 3, 0],
                            [2, 1, 0, 0, 2, 3, 0],
                            [0, 3, 1, 1, 0, 3, 0],
                            [1, 1, 1, 1, 0, 4, 0],
                            [0, 0, 0, 0, 4, 1, 1],
                            [1, 1, 2, 1, 0, 4, 0],
                            [1, 0, 0, 1, 3, 4, 0],
                            [0, 1, 2, 0, 0, 3, 0],
                            [0, 1, 0, 3, 1, 1, 0],
                            [1, 1, 0, 2, 1, 2, 0],
                            [0, 0, 0, 3, 0, 0, 0],
                            [0, 0, 0, 4, 0, 2, 1],
                            [3, 0, 1, 0, 1, 2, 0],
                            [0, 2, 0, 1, 0, 4, 0],
                            [2, 0, 2, 1, 0, 4, 0],
                            [0, 0, 0, 0, 3, 1, 0],
                            [1, 1, 1, 0, 1, 3, 0],
                            [0, 0, 2, 2, 0, 0, 0],
                            [0, 0, 1, 2, 2, 1, 0],
                            [1, 1, 0, 1, 1, 2, 0],
                            [0, 4, 0, 0, 0, 0, 1],
                            [0, 2, 0, 0, 2, 1, 0],
                            [2, 0, 1, 0, 0, 1, 0],
                            [0, 0, 2, 0, 2, 3, 0],
                            [0, 0, 0, 2, 1, 0, 0],
                            [2, 1, 1, 0, 1, 3, 0],
                            [1, 0, 1, 1, 2, 1, 0],
                            [0, 1, 1, 1, 1, 0, 0],
                            [0, 2, 1, 1, 1, 0, 0],
                            [2, 0, 0, 2, 0, 4, 0],
                            [2, 2, 0, 0, 0, 2, 0],
                            [0, 0, 4, 0, 0, 4, 1],
                            [1, 0, 0, 0, 2, 2, 0],
                            [0, 2, 2, 0, 1, 0, 0],
                            [1, 1, 3, 0, 0, 0, 0]]
        self.low_secret = []

        self.middle_banquet = [[5, 0, 0, 0, 0, 0, 2],
                               [6, 0, 0, 0, 0, 0, 3],
                               [0, 0, 5, 0, 0, 2, 2],
                               [3, 0, 0, 2, 2, 2, 1],
                               [0, 0, 3, 2, 3, 3, 1],
                               [0, 2, 0, 3, 2, 1, 1],
                               [1, 2, 4, 0, 0, 3, 2],
                               [0, 0, 0, 0, 5, 3, 2],
                               [2, 0, 2, 0, 3, 3, 1],
                               [0, 5, 0, 3, 0, 4, 2],
                               [3, 3, 0, 0, 2, 4, 1],
                               [0, 3, 5, 0, 0, 3, 2],
                               [0, 0, 0, 5, 3, 1, 2],
                               [2, 3, 2, 0, 0, 0, 1],
                               [0, 6, 0, 0, 0, 1, 3],
                               [0, 1, 0, 4, 2, 0, 2],
                               [0, 0, 6, 0, 0, 2, 3],
                               [0, 0, 0, 5, 0, 1, 2],
                               [5, 0, 3, 0, 0, 2, 2],
                               [0, 0, 0, 6, 0, 3, 3],
                               [3, 2, 0, 3, 0, 1, 1],
                               [3, 0, 0, 0, 5, 0, 2],
                               [0, 3, 2, 0, 3, 2, 1],
                               [0, 2, 3, 2, 0, 4, 1],
                               [0, 5, 0, 0, 0, 4, 2],
                               [0, 0, 0, 0, 6, 4, 3],
                               [4, 0, 2, 0, 1, 1, 2],
                               [2, 0, 0, 1, 4, 2, 2],
                               [2, 0, 3, 3, 0, 0, 1],
                               [0, 4, 1, 2, 0, 4, 2]]
        self.middle_secret = []

        self.high_banquet = [[3, 3, 5, 0, 3, 3, 3],
                             [0, 0, 0, 0, 7, 0, 4],
                             [5, 0, 3, 3, 3, 1, 3],
                             [3, 5, 3, 3, 0, 4, 3],
                             [3, 3, 0, 3, 5, 2, 3],
                             [0, 3, 0, 6, 3, 4, 4],
                             [0, 3, 3, 5, 3, 0, 3],
                             [0, 6, 3, 3, 0, 3, 4],
                             [3, 3, 6, 0, 0, 1, 4],
                             [0, 7, 0, 0, 0, 3, 4],
                             [0, 0, 0, 7, 3, 4, 5],
                             [0, 7, 0, 3, 0, 3, 5],
                             [0, 0, 7, 0, 0, 1, 4],
                             [6, 0, 3, 0, 3, 2, 4],
                             [7, 0, 3, 0, 0, 2, 5],
                             [0, 3, 7, 0, 0, 1, 5],
                             [3, 0, 0, 3, 6, 0, 4],
                             [3, 0, 0, 0, 7, 0, 5],
                             [0, 0, 0, 7, 0, 4, 4],
                             [7, 0, 0, 0, 0, 2, 4]]
        self.high_secret = []

        self.noble_banquet = [[0, 4, 0, 4, 0, 3],
                              [3, 3, 3, 0, 0, 3],
                              [3, 0, 0, 3, 3, 3],
                              [0, 3, 3, 3, 0, 3],
                              [0, 3, 0, 3, 3, 3],
                              [0, 0, 0, 4, 4, 3],
                              [4, 0, 0, 0, 4, 3],
                              [3, 0, 3, 0, 3, 3],
                              [4, 0, 4, 0, 0, 3],
                              [0, 4, 4, 0, 0, 3]]
        self.noble_usable = []

        self.public_money = [5, 5, 5, 5, 5, 5]  # 0-blue, 1-red, 2-green, 3-black, 4-white, 5- gold
        self.agent_money = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
        self.agent_card = [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]  # 0-blue, 1-red, 2-green, 3-black, 4-white
        self.agent_book = [[1, 1, 1, 1, 0, 0, 0],
                           [1, 1, 1, 1, 0, 0, 0]]
        self.agent_score = [0, 0]
        self.agent_role = 0

        self.low_public = [[0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0]]

        self.middle_public = [[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]]

        self.high_public = [[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]

        for i in range(len(self.low_banquet)):
            self.low_secret.append(i)
        for i in range(len(self.middle_banquet)):
            self.middle_secret.append(i)
        for i in range(len(self.high_banquet)):
            self.high_secret.append(i)

        for i in range(4):
            self.refresh_low(i)
            self.refresh_middle(i)
            self.refresh_high(i)
        noble = []
        for i in range(len(self.noble_banquet)):
            noble.append(i)
        for i in range(4):
            random_int = noble[np.random.randint(len(noble))]
            self.noble_usable.append(random_int)
            noble.remove(random_int)
        return self._get_observation(), self.update_mask()

    def render(self) -> None:
        """Renders the next frame."""
        print(":" + str(self.public_money))
        print(":" + str(self.agent_money))
        print(":" + str(self.agent_card))
        print(":" + str(self.agent_book))
        print(":" + str(self.agent_score))
        print(":" + str(self.agent_role))
        print(":" + str(self.low_public))
        print(":" + str(self.middle_public))
        print(":" + str(self.high_public))

    def close(self):
        """Closes the environment."""
        super().close()
