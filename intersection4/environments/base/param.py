import numpy as np

w = 3.0  # lane width
l = 4.0  # car length
s = 50.0  # length pre-intersection segment

d_min = w / (np.sqrt(2) - 1)  # minimum intersection size to prevent overlap of left turns

right_turn_length = (d_min - w / 2) * (np.pi / 2)
# print(right_turn_length/d_min)
# use 4/3 as ratio

# intersection dimension with right turn being 4/3 times longer than one straight section
d = w / (2 - 4 * 4 / (3 * np.pi))

right_turn_length = (d - w / 2) * np.pi / 2
# print(right_turn_length/d)

x_max = d - w / (np.sqrt(2) - 1)  # max straight section for left turn

# print(x_max)
left_turn_length = 2 * x_max + (d - x_max + w / 2) * np.pi / 2
# print(left_turn_length/d)
# use 11/6 as ratio

# straight section length such that left turn is 11/6 longer than one straight section
x = 1 / (4 - np.pi) * ((11 / 3 - np.pi) * d - w / 2 * np.pi)

left_turn_length = 2 * x + (d - x + w / 2) * np.pi / 2

# print(left_turn_length/d)


class IntersectionParams:
    def __init__(
            self,
            straight_length,
            border_radius,
            lane_width,
            left_turn_straight_length,
            step_size):
        self.straight_length = straight_length
        self.border_radius = border_radius
        self.lane_width = lane_width
        self.left_turn_straight_length = left_turn_straight_length
        self.step_size = step_size
        self.intersection_dim = lane_width + border_radius


step_size = d / 6

intersection_params = IntersectionParams(s, d - w, w, x, step_size)

# plot intersection
theta = np.linspace(0, np.pi / 2, 21)
ones = np.ones(21)

turn_edge = [d - (d - w) * np.cos(theta), d - (d - w) * np.sin(theta)]
short_turn = [d * (1 - np.cos(theta)), d * (1 - np.sin(theta))]
long_turn = [d - x - (d - x + w) * np.cos(theta), d - x - (d - x + w) * np.sin(theta)]


def vert_edge(y0, y1, x):
    return [np.array([x, x]), np.array([y0, y1])]


def horz_edge(x0, x1, y):
    return [np.array([x0, x1]), np.array([y, y])]


def flip(line, i):
    if i == 0:
        return line
    if i == 1:
        return [-line[0], line[1]]
    if i == 2:
        return [-line[0], -line[1]]
    if i == 3:
        return [line[0], -line[1]]


def plot_street(plt):
    b = 4
    for i in range(4):
        plt.plot(*flip(turn_edge, i), 'k')
        plt.plot(*flip(vert_edge(d, d + s + b, w), i), 'k')
        plt.plot(*flip(horz_edge(d, d + s + b, w), i), 'k')
        plt.plot(*flip(vert_edge(d, d + s + b, 0), i), 'k--')
        plt.plot(*flip(horz_edge(d, d + s + b, 0), i), 'k--')
        # plt.plot(*flip(short_turn, i), 'k--')
        # plt.plot(*flip(long_turn, i), 'k--')


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    plt.figure()
    plot_street(plt)
    plt.show()
