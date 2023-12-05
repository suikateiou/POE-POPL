from typing import Iterable
from itertools import product

from termcolor import colored

import numpy as np


class TopTrace(object):
    def __init__(
            self,
            num_objs: int,
            *,
            indent_size: int = 4,
        ):

        self.tops = [[] for _ in range(num_objs)]
        self.msgs = [[] for _ in range(num_objs)]
        self.indent_size = indent_size

    def print(
            self,
            new_tops: Iterable[float],
            *,
            show: bool = True,
        ):

        for new_top, top, msg in zip(new_tops, self.tops, self.msgs):
            new_top_msg = f'{new_top * 100.0:.2f}%'
            if top:
                new_top_msg = colored(new_top_msg, 'green' if new_top >= top[-1] else 'red')
                delta = '\u0394=' + colored(f'{(new_top - top[-1]) * 100.0:.2f}%', 'green' if new_top >= top[-1] else 'red')
                abs_delta = 'abs\u0394=' + colored(f'{(new_top - top[0]) * 100.0:.2f}%', 'green' if new_top >= top[0] else 'red')
            top.append(new_top)
            msg.append(new_top_msg)
            if show:
                print(' ' * self.indent_size + ' '.join(msg + [delta, abs_delta]))
                print(flush=True)


def evenly_dist_weights(num_weights, dim):
    # 使用 np.linspace(0.0, 1.0, x) 得到区间 [0,1] 之上长队为 x 的均匀分割，比如 x=5 则得到 [0, 0.25, 0.5, 0.75, 1]
    # product(_list, repeat=2) 就是得到 _list 之内所有元素两两之间的 pair（repeat 对应的是 MTL 中的任务数）
    # 后接的筛选条件是把包含 0 和 1 的 weights 从后选中去掉，比如最终得到 [(0.25, 0.75), (0.5, 0.5). (0.75, 0.25)]
    # 这也就是所谓的"均匀分布的权重候选"
    return [ret for ret in product(
        np.linspace(0.0, 1.0, num_weights), repeat=dim) if round(sum(ret), 3) == 1.0 and all(r not in (0.0, 1.0) for r in ret)]


if __name__ == "__main__":
    ans = evenly_dist_weights(6, 3)
    print(len(ans), ans)
