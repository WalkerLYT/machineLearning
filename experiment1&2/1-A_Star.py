import numpy as np

class State:
    def __init__(self, state, directionFlag=None, parent=None, f=0):
        self.state = state
        self.direction = ['up', 'down', 'right', 'left']
        if directionFlag:
            self.direction.remove(directionFlag)
        self.parent = parent
        self.f = f

    def getDirection(self):
        return self.direction

    def setF(self, f):
        self.f = f
        return

    # 打印结果
    def showInfo(self):
        for i in range(len(self.state)):
            for j in range(len(self.state)):
                print(self.state[i, j], end='  ')
            print("\n")
        print('->')
        return

    # 获取0点
    def getZeroPos(self):
        postion = np.where(self.state == 0)
        return postion

    # 曼哈顿距离  f = g + h，g=1，如果用宽度优先的评估函数可以不调用该函数
    def getFunctionValue(self, depth):
        cur_node = self.state.copy()
        fin_node = self.answer.copy()
        dist = 0
        N = len(cur_node)

        for i in range(N):
            for j in range(N):
                if cur_node[i][j] != fin_node[i][j]:
                    index = np.argwhere(fin_node == cur_node[i][j])
                    x = index[0][0]  # 最终x距离
                    y = index[0][1]  # 最终y距离
                    dist += (abs(x - i) + abs(y - j))
        return dist + depth

    # 位置不符的个数
    # def getFunctionValue(self, depth):
    #     cur_node = self.state.copy()
    #     fin_node = self.answer.copy()
    #     dist = 0
    #     N = len(cur_node)
    #
    #     for i in range(N):
    #         for j in range(N):
    #             if cur_node[i][j] != fin_node[i][j]:
    #                 dist += 1
    #     return dist + depth

    #逆转数码
    # def getFunctionValue(self, depth):
    #     cur_node = self.state.copy()
    #     fin_node = self.answer.copy()
    #     dist = 0
    #     N = len(cur_node)
    #
    #     for i in range(N):
    #         for j in range(N):
    #             if cur_node[i][j] != fin_node[i][j]:
    #                 index = np.argwhere(fin_node == cur_node[i][j])
    #                 x = index[0][0]  # 最终x距离
    #                 y = index[0][1]  # 最终y距离
    #                 if cur_node[x][y] == fin_node[i][j]:
    #                     dist += 0.5
    #     return 3 * dist + depth

    def nextStep(self, depth):
        if not self.direction:
            return []
        subStates = []
        boarder = len(self.state) - 1
        # 获取0点位置
        x, y = self.getZeroPos()
        # 向左
        if 'left' in self.direction and y > 0:
            s = self.state.copy()
            tmp = s[x, y - 1]
            s[x, y - 1] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='right', parent=self)
            news.setF(news.getFunctionValue(depth))
            # print("left")
            subStates.append(news)
        # 向上
        if 'up' in self.direction and x > 0:
            # it can move to upper place
            s = self.state.copy()
            tmp = s[x - 1, y]
            s[x - 1, y] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='down', parent=self)
            news.setF(news.getFunctionValue(depth))
            # print("up")
            subStates.append(news)
        # 向下
        if 'down' in self.direction and x < boarder:
            # it can move to down place
            s = self.state.copy()
            tmp = s[x + 1, y]
            s[x + 1, y] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='up', parent=self)
            news.setF(news.getFunctionValue(depth))
            # print("down")
            subStates.append(news)
        # 向右
        if self.direction.count('right') and y < boarder:
            # it can move to right place
            s = self.state.copy()
            tmp = s[x, y + 1]
            s[x, y + 1] = s[x, y]
            s[x, y] = tmp
            news = State(s, directionFlag='left', parent=self)
            news.setF(news.getFunctionValue(depth))
            # print("right")
            subStates.append(news)
        # 返回F值最小的下一个点
        subStates.sort(key=compareNum)
        return subStates

    # A* 迭代
    def solve(self):
        # openList
        openTable = []
        # closeList
        closeTable = []
        openTable.append(self)
        depth = 1
        while len(openTable) > 0:
            # 下一步的点移除open
            n = openTable.pop(0)
            print("当前：\n", n.state)
            # 加入close
            closeTable.append(n)
            # 确定下一步点
            subStates = n.nextStep(depth)
            depth += 1
            path = []
            # 判断是否和最终结果相同
            if (subStates[0].state == subStates[0].answer).all():
                while subStates[0].parent and subStates[0].parent != originState:
                    path.append(subStates[0].parent)
                    subStates[0] = subStates[0].parent
                path.reverse()
                return path
            print("\nopenTable:")
            for i in subStates:
                openTable.append(i)
            openTable.sort(key=compareNum)
            for i in openTable:
                print(i.state)
            print("\ncloseTable:")
            for i in closeTable:
                print(i.state)
        else:
            return None, None


def compareNum(state):
    return state.f


if __name__ == '__main__':
    # originState = State(np.array([[1, 5, 3], [2, 4, 6], [7, 0, 8]]))
    # State.answer = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])
    originState = State(np.array([[2, 8, 3], [1, 6, 4], [7, 0, 5]]))
    State.answer = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])

    s1 = State(state=originState.state)
    path = s1.solve()
    if path:
        for node in path:
            node.showInfo()
        print(State.answer)
        print("Total steps is %d" % len(path))
