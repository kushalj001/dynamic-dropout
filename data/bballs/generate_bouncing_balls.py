"""
Script for generating bouncing ball data set.
"""

# Author: adapted from: https://github.com/simonkamronn/kvae

import pygame
import pymunk.pygame_util
import numpy as np
import os
import imageio
import random
import math


class BallBox:
    def __init__(self, dt=0.2, res=(32, 32), init_pos=(3, 3), init_std=0, wall=None, gravity=(0.0, 0.0),
                 create_gifs=False, dirname="."):
        pygame.init()

        self.dt = dt
        self.res = res
        if os.environ.get('SDL_VIDEODRIVER', '') == 'dummy':
            pygame.display.set_mode(res, 0, 24)
            self.screen = pygame.Surface(res, pygame.SRCCOLORKEY, 24)
            pygame.draw.rect(self.screen, (0, 0, 0), (0, 0, res[0], res[1]), 0)
        else:
            self.screen = pygame.display.set_mode(res, 0, 24)
        self.gravity = gravity
        self.initial_position = init_pos
        self.initial_std = init_std
        self.space = pymunk.Space()
        self.space.gravity = self.gravity
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        self.draw_options.flags = pymunk.SpaceDebugDrawOptions.DRAW_SHAPES
        #self.draw_options.shape_outline_color = self.draw_options.shape_dynamic_color
        self.clock = pygame.time.Clock()
        self.wall = wall
        self.static_lines = None
        self.create_gifs = create_gifs
        self.dd = 1
        self.dirname = dirname

    def _clear(self):
        self.screen.fill(pygame.color.THECOLORS["white"])

    def create_ball(self, radius=3):
        inertia = pymunk.moment_for_circle(1, 0, radius, (0, 0))
        body = pymunk.Body(1, inertia)
        position = np.array(self.initial_position) + self.initial_std * np.random.normal(size=(2,))
        position = np.clip(position, self.dd + radius + 1, self.res[0]-self.dd-radius-1)
        body.position = position
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 0.8
        shape.color = pygame.color.THECOLORS["black"]
        return shape

    def fire(self, angle=50, velocity=20, radius=3):
        speedX = velocity * np.cos(angle * np.pi / 180)
        speedY = velocity * np.sin(angle * np.pi / 180)

        ball = self.create_ball(radius)
        ball.body.velocity = (speedX, speedY)

        self.space.add(ball, ball.body)
        return ball

    def run(self, gravity_setup, iterations=20, sequences=500, angle_limits=(0, 360), velocity_limits=(10, 25),
            radius=3, save=None, filepath_data=None, filepath_labels=None, delay=None):
        if save or self.create_gifs:
            images = np.empty((sequences, 2*iterations, self.res[0], self.res[1]), dtype=np.float32)
            state = np.empty((sequences, 2*iterations, 4), dtype=np.float32)

        dd = self.dd
        self.static_lines = [pymunk.Segment(self.space.static_body, (dd, dd+1), (dd, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (dd, dd+1), (self.res[0]-dd-1, dd+1), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd-1, self.res[1] - dd), (dd, self.res[1]-dd), 0.0),
                             pymunk.Segment(self.space.static_body, (self.res[0] - dd-1, self.res[1] - dd), (self.res[0]-dd-1, dd+1), 0.0)]

        for line in self.static_lines:
            line.elasticity = 1.0
            line.friction = 0.5
            line.color = pygame.color.THECOLORS["black"]
        self.space.add(self.static_lines)

        labels = np.zeros(sequences, dtype=int)

        g_index = 0

        for s in range(sequences):

            if s % 100 == 0:
                print(s)

            # Set the gravity
            labels[s] = g_index
            self.space.gravity = gravity_setup[g_index]
            g_index = (g_index+1) % len(gravity_setup)

            angle = np.random.uniform(*angle_limits)
            velocity = np.random.uniform(*velocity_limits)
            # controls[:, s] = np.array([angle, velocity])
            ball = self.fire(angle, velocity, radius)

            for i in range(2*iterations):
                self._clear()
                self.space.debug_draw(self.draw_options)
                self.space.step(self.dt)
                pygame.display.flip()

                if delay:
                    self.clock.tick(delay)

                if save == 'png':
                    pygame.image.save(self.screen, os.path.join(filepath_data,
                                                                "images/bouncing_balls_%02d_%02d.png" % (s, i)))
                elif save == 'npz' or save == 'npy' or self.create_gifs:
                    images[s, i] = pygame.surfarray.array2d(self.screen).swapaxes(1, 0).astype(np.float32) / (2**24 - 1)
                    state[s, i] = list(ball.body.position) + list(ball.body.velocity)

            # Remove the ball and the wall from the space
            self.space.remove(ball, ball.body)

        # unsqueeze one dimension
        images = np.expand_dims(images, 2)
        images = 1 - images
        eps = 0.2
        images[images < eps] = 0
        images[images > 1-eps] = 1
        images = images.astype(int)

        # downsample images by two
        images = images[:, ::2]
        print(images.shape)

        if save == 'npz':
            np.savez(os.path.abspath(filepath_data), images=images, state=state)

        if save == 'npy':
            np.save(os.path.abspath(filepath_data), images)
            np.save(os.path.abspath(filepath_labels), labels)

        if self.create_gifs:
            if not os.path.exists(self.dirname+'/sample_gifs'):
                os.makedirs(self.dirname+'/sample_gifs')
            for j in range(sequences):
                imageio.mimsave(self.dirname+'/sample_gifs/sample' + str(j) + '.gif', images[j, :, 0, :, :])


if __name__ == '__main__':

    os.environ['SDL_VIDEODRIVER'] = 'dummy'  # Remove and add delay to see the videos

    dirname = "."
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    scale = 1

    np.random.seed(1234)
    random.seed(1234)

    # parameters to change
    num_gravities = 8
    leave_out = True
    seq_len = 10
    init_std = 5
    vel_down = 15.0
    vel_up = 25.0
    radius = 7

    # make sure you don't have too little gravities when leave-one-out
    assert(num_gravities > 4 or not leave_out)

    # uniformly spaced gravities
    theta = np.linspace(0, 2 * np.pi, num_gravities, endpoint=False)
    gravity_setup = list()
    for i in range(num_gravities):
        gravity_setup.append((math.cos(theta[i])*10, math.sin(theta[i])*10))

    gravity_setup_train = gravity_setup.copy()
    gravity_setup_valid = gravity_setup.copy()
    gravity_setup_test = gravity_setup.copy()

    # leave one gravity out of training set if specified
    if leave_out:
        del gravity_setup_train[0]
        del gravity_setup_train[-1]
        del gravity_setup_valid[1:]
        del gravity_setup_test[:-1]

    print("Generating training sequences: shooting angle 300")
    cannon = BallBox(dt=0.1, res=(32*scale, 32*scale), init_pos=(16*scale, 16*scale), init_std=init_std * scale, 
                     wall=None)
    cannon.run(gravity_setup_train, iterations=seq_len, sequences=10000, radius=radius * scale,
               angle_limits=(0, 360), velocity_limits=(vel_down*scale, vel_up*scale),
               filepath_data=dirname+'/bb_train_data.npy', filepath_labels=dirname+'/bb_train_labels.npy', save='npy')

    print("Generating validation sequences: shooting angle 300")
    cannon = BallBox(dt=0.1, res=(32 * scale, 32 * scale), init_pos=(16 * scale, 16 * scale), init_std=init_std * scale,
                     wall=None)
    cannon.run(gravity_setup_valid, iterations=seq_len, sequences=1000, radius=radius * scale,
               angle_limits=(0, 360), velocity_limits=(vel_down * scale, vel_up * scale),
               filepath_data=dirname+'/bb_valid_data.npy', filepath_labels=dirname+'/bb_valid_labels.npy', save='npy')

    print("Generating test sequences: shooting angle 300")
    cannon = BallBox(dt=0.1, res=(32 * scale, 32 * scale), init_pos=(16 * scale, 16 * scale), init_std=init_std * scale,
                     wall=None)
    cannon.run(gravity_setup_test, iterations=seq_len, sequences=1000, radius=radius * scale,
               angle_limits=(0, 360), velocity_limits=(vel_down * scale, vel_up * scale),
               filepath_data=dirname + '/bb_test_data.npy', filepath_labels=dirname+'/bb_test_labels.npy', save='npy')

    print("Creating sample gifs")
    cannon = BallBox(dt=0.1, res=(32 * scale, 32 * scale), init_pos=(16 * scale, 16 * scale), init_std=init_std * scale,
                     wall=None, create_gifs=True, dirname=dirname)
    cannon.run(gravity_setup_train, iterations=seq_len, sequences=10, radius=radius * scale, angle_limits=(0, 360),
               velocity_limits=(vel_down * scale, vel_up * scale))
