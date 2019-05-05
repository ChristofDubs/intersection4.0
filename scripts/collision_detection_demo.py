import time
from random import randint
import numpy as np
import matplotlib.pyplot as plt
import context
from intersection4.environments.base.intersection import Intersection
from intersection4.environments.base.car import Car, CarParams
from intersection4.environments.base.param import intersection_params, plot_street
from intersection4.environments.base.definitions import Target, Action
from intersection4.environments.base.collision_checker import CollisionChecker

sim_duration = 20  # [s]

create_gif = False
frame_rate = 10
speed_up_factor = 1.5

intersection = Intersection(intersection_params)
collision_checker = CollisionChecker(intersection)
cars = []

fig = plt.figure(0)
start_time = time.time()
prev_time = time.time()
alpha = 0
collision_pairs = []

if create_gif:
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    import imageio
    canvas = FigureCanvas(fig)
    images = []

while time.time() - start_time < sim_duration:
    plt.figure(0)
    plt.cla()
    plot_street(plt)

    collisions_flat = [car for pair in collision_pairs for car in pair]

    for i, car in enumerate(cars):
        car.set_action(Action(0))
        in_collision = i in collisions_flat
        style = 'g-' if not in_collision else 'r-'
        car.plot_interpolated(plt, intersection, alpha, style)
        # car.plot(plt, intersection, style)

    plt.show(block=False)
    plt.pause(0.05)

    if create_gif:
        alpha += 1.0 / frame_rate
    else:
        time_passed = time.time() - prev_time
        prev_time = time.time()
        alpha += time_passed
    if alpha > 1:
        for i, car in enumerate(cars):
            in_collision = i in collisions_flat
            if in_collision:
                car.reset()
            if car.is_active():
                car.execute_action()
        alpha = np.fmod(alpha, 1)

        # delete inactive cars
        cars[:] = [car for car in cars if car.is_active()]

        collision_pairs = collision_checker.check_collisions(cars)

        for quadrant in range(4):
            x = randint(0, 5)
            if x > 2:
                continue
            target = x % 3

            cars.append(Car(CarParams()))
            cars[-1].spawn(quadrant, Target(target), 5, intersection)

    if create_gif:
        # convert figure to numpy image:
        # https://stackoverflow.com/questions/21939658/matplotlib-render-into-buffer-access-pixel-data
        fig.canvas.draw()

        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        image = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)

        images.append(image)

if create_gif:
    imageio.mimsave(
        'demo.gif',
        images,
        fps=speed_up_factor *
        frame_rate,
        palettesize=64,
        subrectangles=True)
