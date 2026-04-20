import os
from collections import deque

import numpy as np


DEBUG = int(os.environ.get('HAS_DISPLAY', 0))
# DEBUG=True

class Plotter(object):
    def __init__(self, size):
        self.size = size
        self.clear()
        self.title = str(self.size)

    def clear(self):
        from PIL import Image, ImageDraw

        self.img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        self.draw = ImageDraw.Draw(self.img)

    def dot(self, pos, node, color=(255, 255, 255), r=2):
        x, y = 5.5 * (pos - node)
        x += self.size / 2
        y += self.size / 2

        self.draw.ellipse((x-r, y-r, x+r, y+r), color)

    def show(self):
        if not DEBUG:
            return

        import cv2

        cv2.imshow(self.title, cv2.cvtColor(np.array(self.img), cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance  # 最近距离 初始化给出
        self.max_distance = max_distance  # 最远距离 初始化给出

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9  早期carla版本中常用的设置是选择一个场景中心位置作为原点
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        self.mean = np.array([0.0, 0.0]) # for carla 9.10   
        self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10  这组数值的含义是 纬度方向上1° ≈ 111324 米，经度方向上1° ≈ 111319 米

        self.debug = Plotter(debug_size)  # 用于可视化路径点、当前点、目标点（调试用）

    def set_route(self, global_plan, gps=False, global_plan_world = None): # 把由全局路径规划器生成的路径保存进self.route
        self.route.clear()

        if global_plan_world:
            for (pos, cmd), (pos_word, _ )in zip(global_plan, global_plan_world):
                if gps:  # 如果是GPS路径 先减去均值(中心化) 再乘上比例尺 这样就将经纬度转换为了笛卡尔米单位
                    pos = np.array([pos['lat'], pos['lon']])  # lat 经度 lon 维度  
                    pos -= self.mean
                    pos *= self.scale  # 到这里就将pos转换为了[x,y]的形式了
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean
                
                self.route.append((pos, cmd, pos_word))
        else:  # 验证的时候执行这里
            for pos, cmd in global_plan:
                if gps:
                    pos = np.array([pos['lat'], pos['lon']])
                    pos -= self.mean
                    pos *= self.scale
                    # print("验证的时候执行这里: ", pos)
                else:
                    pos = np.array([pos.location.x, pos.location.y])
                    pos -= self.mean

                self.route.append((pos, cmd))

    # 每帧调用
    def run_step(self, gps):
        """
        该函数根据自车当前位置动态维护路径队列，丢弃已经接近的路径点，并返回下一个要去追踪的目标路径点与高级驾驶指令。
        """
        self.debug.clear()

        if len(self.route) == 1:  # 边界条件 如果此时只剩下一个路径点
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf  # 用以记录最远点到自车当前位置的距离
        cumulative_distance = 0.0    # 用以记录路径总长度

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            # 计算相邻两个路径点之间的欧几里德距离并且叠加 当总距离超过最远距离,就不再处理了
            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)  # 计算路径点与车辆当前位置(gps)的距离

            # 在距离车辆当前位置小于 min_distance 的所有路径点中，选择最远的那个点，然后准备从路径队列中将它前面的点“弹出”（也就是认为这些点已经走过了）。
            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

            r = 255 * int(distance > self.min_distance)
            g = 255 * int(self.route[i][1].value == 4)
            b = 255
            self.debug.dot(gps, self.route[i][0], (r, g, b))

        # 从路径队列 self.route 中，把前面那些“已经走过或太近”的点删掉（也就是弹出），最多弹出 to_pop 个点。
        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()  # 队列的左端依次弹出 使得队列第一个就是

        # 绘制路径点 for debug
        # self.debug.dot(gps, self.route[0][0], (0, 255, 0))  # 绿色
        # self.debug.dot(gps, self.route[1][0], (255, 0, 0))
        for _ in range(len(self.route)):
            self.debug.dot(gps, self.route[_][0], (255, 0, 0))
        self.debug.dot(gps, gps, (0, 0, 255))                  # 自车位置 蓝色
        self.debug.show()

        return self.route[1]  # 返回的是全局路径中距离自车当前位置小于 min_distance 的所有路径点中, 最远的那个点的下一个点,实际上选择最远的那个点也行,只是下一个点更具有前沿性
