import os
import csv
import ray
from copy import deepcopy
from leaderboard.nocrash_evaluator import NoCrashEvaluator
import random
import numpy as np
import dill as pickle

class NoCrashEvalRunner():
    def __init__(self, args, town, weather, port=1000, tm_port=1002, debug=False):
        args = deepcopy(args)
        # Inject args
        args.scenario_class = 'nocrash_eval_scenario'
        args.port = port
        args.trafficManagerPort = tm_port
        args.debug = debug
        args.record = ''

        args.town = town
        args.weather = weather

        self.runner = NoCrashEvaluator(args, StatisticsManager(args))
        self.args = args

        self.pop_size = 50

        self.next_range = [[4200, 7000], [0.1, 0.2], [1, 3], [0.2, 0.4], [0.3, 0.6], [8, 12], [2040, 2700], [0.2, 0.5],
                           [1, 3.9],
                           [0.2, 0.3], [31.7, 37], [1200, 1650]]
        self.next_max_nums = 12
        self.or_sd = 2.2
        self.or_value = [5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500]
        self.all_sol = []
        self.all_config = []
        self.all_comb = []

        self.cur_eva = 0
        self.max_evaluation = 5000
        self.flag = True
        self.stop_signal = 0
        self.cur_5_phy_config = []
        self.cur_5_sol = []

    def rerun(self):
        pickle_file = open('myalgorithm.pkl', 'rb')
        runner = pickle.load(pickle_file)
        pickle_file.close()
        self.all_sol = runner.all_sol
        self.all_config = runner.all_config
        self.all_comb = runner.all_comb

        self.cur_eva = runner.cur_eva
        self.flag = runner.flag
        self.stop_signal = runner.stop_signal

        self.next_range = runner.next_range
        self.next_max_nums = runner.next_max_nums

        self.cur_eva = self.pop_size * (int)(self.cur_eva / self.pop_size)
        self.all_sol = self.all_sol[:self.cur_eva]
        self.all_config = self.all_config[:self.cur_eva]
        self.all_comb = self.all_comb[:self.cur_eva]

        self.cur_5_sol = runner.cur_5_sol
        self.cur_5_phy_config = runner.cur_5_phy_config

        index = self.pop_size * (int)(len(self.cur_5_sol) / self.pop_size)
        self.cur_5_sol=self.cur_5_sol[:index]
        self.cur_5_phy_config = self.cur_5_phy_config[:index]
        print(len(self.all_comb))
        print(len(self.cur_5_phy_config))
        print(len(self.cur_5_sol))

    def run(self):
        phy0 =[]
        try:
            self.runner.run(self.args, phy0)
        except Exception as e:
            print("catch error:", e.args)
            self.runner = None
            self.problem = None
            save_file = open('myalgorithm.pkl', 'wb')
            pickle.dump(self, save_file)
            save_file.close()
            print("save pkl")
            print("cur_eva :", self.cur_eva)
        return
    def algothrim_run(self):
        while self.cur_eva < self.max_evaluation and self.flag == True:
            cur_gen_config = self.random_gen(self.pop_size, self.next_range, self.next_max_nums)
            cur_gen_sol = []
            cur_gen_resl_phy = []
            for i in range(0, self.pop_size):
                sol = []
                x0 = float('%.3f' % cur_gen_config[i][0])
                x1 = float('%.3f' % cur_gen_config[i][1])
                x2 = float('%.3f' % cur_gen_config[i][2])
                x3 = float('%.3f' % cur_gen_config[i][3])
                x4 = float('%.3f' % cur_gen_config[i][4])
                x5 = float('%.3f' % cur_gen_config[i][5])
                x6 = float('%.3f' % cur_gen_config[i][6])
                x7 = float('%.3f' % cur_gen_config[i][7])
                x8 = float('%.3f' % cur_gen_config[i][8])
                x9 = float('%.3f' % cur_gen_config[i][9])
                x10 = float('%.3f' % cur_gen_config[i][10])
                x11 = float('%.3f' % cur_gen_config[i][11])

                global f_r
                f_r = open("file_storage.txt", 'a')

                f_r.write(str(x0))
                f_r.write(' ')
                f_r.write(str(x1))
                f_r.write(' ')
                f_r.write(str(x2))
                f_r.write(' ')
                f_r.write(str(x3))
                f_r.write(' ')
                f_r.write(str(x4))
                f_r.write(' ')
                f_r.write(str(x5))
                f_r.write(' ')
                f_r.write(str(x6))
                f_r.write(' ')
                f_r.write(str(x7))
                f_r.write(' ')
                f_r.write(str(x8))
                f_r.write(' ')
                f_r.write(str(x9))
                f_r.write(' ')
                f_r.write(str(x10))
                f_r.write(' ')
                f_r.write(str(x11))
                f_r.write(' ')

                change = []
                change_ratio = []

                # changes' precision
                d0 = float('%.3f' % (np.abs(x0 - 5800) / (7000 - 4200)))
                d1 = float('%.3f' % (np.abs(x1 - 0.15) / (0.2 - 0.1)))
                d2 = float('%.3f' % (np.abs(x2 - 2) / (3 - 1)))
                d3 = float('%.3f' % (np.abs(x3 - 0.35) / (0.4 - 0.2)))
                d4 = float('%.3f' % (np.abs(x4 - 0.5) / (0.6 - 0.3)))
                d5 = float('%.3f' % (np.abs(x5 - 10) / (12 - 8)))
                d6 = float('%.3f' % (np.abs(x6 - 2404) / (2700 - 2040)))
                d7 = float('%.3f' % (np.abs(x7 - 0.3) / (0.5 - 0.2)))
                d8 = float('%.3f' % (np.abs(x8 - 3.5) / (3.9 - 1)))
                d9 = float('%.3f' % (np.abs(x9 - 0.25) / (0.3 - 0.2)))
                d10 = float('%.3f' % (np.abs(x10 - 35.5) / (37.0 - 31.7)))
                d11 = float('%.3f' % (np.abs(x11 - 1500) / (1650 - 1200)))

                # changes' ratio
                r0 = float('%.3f' % (np.abs(x0 - 5800) / 5800))
                r1 = float('%.3f' % (np.abs(x1 - 0.15) / 0.15))
                r2 = float('%.3f' % (np.abs(x2 - 2) / 2))
                r3 = float('%.3f' % (np.abs(x3 - 0.35) / 0.35))
                r4 = float('%.3f' % (np.abs(x4 - 0.5) / 0.5))
                r5 = float('%.3f' % (np.abs(x5 - 10) / 10))
                r6 = float('%.3f' % (np.abs(x6 - 2404) / 2404))
                r7 = float('%.3f' % (np.abs(x7 - 0.3) / 0.3))
                r8 = float('%.3f' % (np.abs(x8 - 3.5) / 3.5))
                r9 = float('%.3f' % (np.abs(x9 - 0.25) / 0.25))
                r10 = float('%.3f' % (np.abs(x10 - 35.5) / 35.5))
                r11 = float('%.3f' % (np.abs(x11 - 1500) / 1500))

                change_ratio.append(r0)
                change_ratio.append(r1)
                change_ratio.append(r2)
                change_ratio.append(r3)
                change_ratio.append(r4)
                change_ratio.append(r5)
                change_ratio.append(r6)
                change_ratio.append(r7)
                change_ratio.append(r8)
                change_ratio.append(r9)
                change_ratio.append(r10)
                change_ratio.append(r11)

                change_max = max(change_ratio)

                # max_rpm
                if d0 < 0.01:

                    x0 = 5800.0
                else:
                    change.append(d0)

                # damping_rate_full_throttle
                if d1 < 0.08:

                    x1 = 0.15
                else:
                    change.append(d1)

                # damping_rate_zero_throttle_clutch_engaged
                if d2 < 0.04:

                    x2 = 2.0
                else:
                    change.append(d2)

                # damping_rate_zero_throttle_clutch_disengaged
                if d3 < 0.08:
                    x3 = 0.35
                else:
                    change.append(d3)

                # gear_switch_time
                if d4 < 0.08:

                    x4 = 0.5
                else:
                    change.append(d4)
                # clutch_strength
                if d5 < 0.04:
                    x5 = 10.0
                else:
                    change.append(d5)

                # mass
                if d6 < 0.02:

                    x6 = 2404.0
                else:
                    change.append(d6)
                # drag_coefficient
                if d7 < 0.08:

                    x7 = 0.3
                else:
                    change.append(d7)

                    # tire_friction
                if d8 < 0.04:

                    x8 = 3.5
                else:
                    change.append(d8)

                    # damping_rate
                if d9 < 0.08:

                    x9 = 0.25
                else:
                    change.append(d9)

                    # radius
                if d10 < 0.04:

                    x10 = 35.5
                else:
                    change.append(d10)
                    # max_brake_torque
                if d11 < 0.02:

                    x11 = 1500.0
                else:
                    change.append(d11)
                #
                physics = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11]
                if len(change) == 0:
                    physics = [5800.000, 0.150, 2.000, 0.350, 0.500, 10.000, 2404.00, 0.300, 3.500, 0.150, 35.500,
                               1500.000]
                try:

                    SD, tet, tit, avgDece = self.runner.run(self.args, physics)
                except Exception as e:
                    print("error:", e.args)
                    self.runner = None
                    self.cur_eva = self.pop_size * (int)(self.cur_eva / self.pop_size)
                    self.all_sol = self.all_sol[:self.cur_eva]
                    self.all_config = self.all_config[:self.cur_eva]
                    self.all_comb = self.all_comb[:self.cur_eva]
                    f12 = float('%.3f' % change_max)
                    f13 = float('%.3f' % 6.0)
                    f14 = len(change)
                    f_r.write(str(f12))
                    f_r.write(' ')
                    f_r.write(str(f13))
                    f_r.write(' ')
                    f_r.write(str(f14))
                    f_r.write(' ')
                    f_r.write(str(tet))
                    f_r.write(' ')
                    f_r.write(str(tit))
                    f_r.write(' ')
                    f_r.write(str(avgDece))
                    f_r.write(' ')
                    f_r.write("\n")
                    f_r.close()
                    raise Exception(e)
                MaxPC = float('%.3f' % change_max)
                SD = float('%.3f' % SD)
                MaxNums = len(change)

                sol.append(MaxPC)
                sol.append(SD)
                sol.append(MaxNums)
                tmp = np.concatenate((np.array(cur_gen_config[i]), np.array(sol)), axis=0)

                self.all_comb.append(tmp.tolist())
                self.all_sol.append(sol)
                self.all_config.append(cur_gen_config[i])
                self.cur_eva += 1
                print("cur_eva:", self.cur_eva)
                cur_gen_sol.append(sol)

                cur_gen_resl_phy.append(physics)
                self.cur_5_phy_config.append(physics)
                self.cur_5_sol.append(sol)

                f_r.write(str(MaxPC))
                f_r.write(' ')
                f_r.write(str(SD))
                f_r.write(' ')
                f_r.write(str(MaxNums))
                f_r.write(' ')
                f_r.write(str(tet))
                f_r.write(' ')
                f_r.write(str(tit))
                f_r.write(' ')
                f_r.write(str(avgDece))
                f_r.write(' ')
                f_r.write("\n")
                f_r.close()
            isConverge = False
            if self.cur_eva > 500:
                isConverge = self.is_converge(self.all_comb)
                if isConverge:
                    self.stop_signal += 1
                else:
                    self.stop_signal = 0
                if self.stop_signal == 5:
                    self.flag = False
            if self.cur_eva >= 500 and self.cur_eva % (self.pop_size * 5) == 0:
                self.next_max_nums, self.next_range = self.cal_next_range_maxNums(self.cur_5_phy_config, self.cur_5_sol,
                                                                                  self.or_sd,
                                                                                  self.or_value,
                                                                                  self.next_range, self.next_max_nums)
                self.cur_5_phy_config = []
                self.cur_5_sol = []
            global f_e
            f_e = open("gen_next_nums_range.txt", 'a')
            f_e.write(str(self.next_max_nums))
            f_e.write(":")
            for i in self.next_range:
                f_e.write(str(i))
            f_e.write("::stop_signal:")
            f_e.write(str(self.stop_signal))
            f_e.write("::isConverage::")
            f_e.write(str(isConverge))
            f_e.write("\n")
            f_e.close()
            print("cur_gen:", self.cur_eva / self.pop_size)
        final_comb = self.cal_final_best(self.all_comb)
        np.savetxt('final_comb.txt', final_comb, fmt="%.3f", delimiter=" ")

    def random_gen(self, pop_size, rangeRefer, maxChangedNum):

        res_params = [[5800, 0.15, 2.0, 0.35, 0.5, 10.0, 2404, 0.3, 3.5, 0.25, 35.5, 1500] for _ in range(0, pop_size)]
        for i in range(0, pop_size):
            res_changedNum = random.randint(1, maxChangedNum)
            selected_arr = random.sample(range(0, 12), res_changedNum)
         
            for j in range(0, len(selected_arr)):
                x = random.uniform(rangeRefer[selected_arr[j]][0],
                                   rangeRefer[selected_arr[j]][1])
                while self.isChange(selected_arr[j], x) is False:
                    x = random.uniform(rangeRefer[selected_arr[j]][0],
                                       rangeRefer[selected_arr[j]][1])

                res_params[i][selected_arr[j]] = x
        return res_params
    def isChange(self, index, value):
        value = float('%.3f' % value)
        if index == 0:
            d0 = float('%.3f' % (np.abs(value - 5800) / (7000 - 4200)))
            if d0 < 0.01:
                return False
            else:
                return True
        elif index == 1:
            d1 = float('%.3f' % (np.abs(value - 0.15) / (0.2 - 0.1)))
            if d1 < 0.08:
                return False
            else:
                return True
        elif index == 2:
            d2 = float('%.3f' % (np.abs(value - 2) / (3 - 1)))
            if d2 < 0.04:
                return False
            else:
                return True
        elif index == 3:
            d3 = float('%.3f' % (np.abs(value - 0.35) / (0.4 - 0.2)))
            if d3 < 0.08:
                return False
            else:
                return True
        elif index == 4:
            d4 = float('%.3f' % (np.abs(value - 0.5) / (0.6 - 0.3)))
            if d4 < 0.08:

                return False
            else:
                return True

        elif index == 5:
            d5 = float('%.3f' % (np.abs(value - 10) / (12 - 8)))
            if d5 < 0.04:

                return False
            else:
                return True
        elif index == 6:
            d6 = float('%.3f' % (np.abs(value - 2404) / (2700 - 2040)))
            if d6 < 0.02:

                return False
            else:
                return True
        elif index == 7:
            d7 = float('%.3f' % (np.abs(value - 0.3) / (0.5 - 0.2)))
            if d7 < 0.08:

                return False
            else:
                return True

        elif index == 8:
            d8 = float('%.3f' % (np.abs(value - 3.5) / (3.9 - 1)))
            if d8 < 0.04:

                return False
            else:
                return True

        elif index == 9:
            d9 = float('%.3f' % (np.abs(value - 0.25) / (0.3 - 0.2)))
            if d9 < 0.08:

                return False
            else:
                return True

        elif index == 10:
            d10 = float('%.3f' % (np.abs(value - 35.5) / (37.0 - 31.7)))
            if d10 < 0.04:
                return False
            else:
                return True

        elif index == 11:
            d11 = float('%.3f' % (np.abs(value - 1500) / (1650 - 1200)))
            if d11 < 0.02:
                return False
            else:
                return True
    def cal_next_range_maxNums(self, cur_paras, cur_gen_sol, or_sd, original_value, cur_range, cur_max_nums):

        cur_gen = np.concatenate((np.array(cur_paras), np.array(cur_gen_sol)), axis=1)
        cur_list = cur_gen.tolist()
        cur_gen_sol.sort(reverse=False, key=lambda cur_gen_sol: cur_gen_sol[1])
        filter_sol = list(filter(lambda x: x[1] < or_sd, cur_gen_sol))

        if len(filter_sol) == 0:
            res_next_nums = cur_max_nums
        else:
            mid = np.mean(filter_sol, 0)
            res_next_nums = int(round(mid[2], 0))

        filter_cur = list(filter(lambda x: x[13] < or_sd, cur_list))

        res_next_range = cur_range
        if len(filter_cur) == 0:
            print("No found data : filter_cur < 2.2")
        else:
            for i in range(0, 12):
                positive_arr = []
                negative_arr = []
                for j in range(0, len(filter_cur)):
                    if filter_cur[j][i] > original_value[i]:
                        positive_arr.append(filter_cur[j][i])
                    if filter_cur[j][i] < original_value[i]:
                        negative_arr.append(filter_cur[j][i])
                pos_np = np.array(positive_arr)
                neg_np = np.array(negative_arr)
                if len(neg_np) != 0:
                    res_next_range[i][0] = round(np.mean(neg_np), 3)
                if len(pos_np) != 0:
                    res_next_range[i][1] = round(np.mean(pos_np), 3)
        return res_next_nums, res_next_range
        # test done

    def is_converge(self, all_comb):
        np_al = np.array(all_comb)
        col_sd = np_al[:, 13]
        sd_max = np.max(col_sd)
        sd_min = np.min(col_sd)
        nums_max = 12
        nums_min = 1
        score = []
        for i in range(0, len(all_comb)):
            sum = []
            x = all_comb[i][12]
            #

            y = self.nor_single(all_comb[i][13], sd_min, sd_max)
            #
            z = self.nor_single(all_comb[i][14], nums_min, nums_max)
            sum.append(round((x + 6 * y + z) / 8, 4))
            score.append(sum)

        extra_list = np.concatenate((np.array(all_comb), np.array(score)), axis=1)
        cur_comb_np = extra_list[-self.pop_size:, :]

        pre_comb_np = extra_list[:(len(extra_list) - self.pop_size), :]


        cur_comb = cur_comb_np.tolist()
        pre_comb = pre_comb_np.tolist()
        print("cur_len:", len(cur_comb), " pre_len:", len(pre_comb), " all_len:", len(all_comb))

        cur_comb.sort(reverse=False, key=lambda cur_comb: cur_comb[15])
        pre_comb.sort(reverse=False, key=lambda pre_comb: pre_comb[15])
        best_comb = pre_comb[:self.pop_size]
        best_comb.sort(reverse=True, key=lambda best_comb: best_comb[15])
        if cur_comb[0][15] < best_comb[0][15]:
            return False
        else:
            return True

    def cal_final_best(self, all_comb):
        np_al = np.array(all_comb)
        col_sd = np_al[:, 13]
        sd_max = np.max(col_sd)
        sd_min = np.min(col_sd)
        nums_max = 12
        nums_min = 1
        score = []
        for i in range(0, len(all_comb)):
            sum = []
            x = all_comb[i][12]
            y = self.nor_single(all_comb[i][13], sd_min, sd_max)
            z = self.nor_single(all_comb[i][14], nums_min, nums_max)
            sum.append(round((x + 6 * y + z) / 8, 4))
            score.append(sum)
        extra_list = np.concatenate((np.array(all_comb), np.array(score)), axis=1)
        all_merge = extra_list.tolist()
        all_merge.sort(reverse=False, key=lambda all_merge: all_merge[15])
        best_comb = all_merge[:self.pop_size]
        return best_comb

    def nor(self, arr):
        np_arr = np.array(arr)
        max = np.max(np_arr)
        min = np.min(np_arr)
        nor_arr = (np_arr - min) / (max - min)
        return nor_arr.tolist()

    def nor_single(self, value, minimum, maximum):
        return (value - minimum) / (maximum - minimum)
