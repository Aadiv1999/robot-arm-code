"""
challenge.py
"""
import time
from typing import List, Tuple, Union
import copy
import numpy as np
import pygame
import matplotlib.path as mplPath
import matplotlib.pylab as plt
from math import sin, cos, atan, atan2, pi, sqrt


class Robot:
    JOINT_LIMITS = [-6.28, 6.28]
    MAX_VELOCITY = 15000
    MAX_ACCELERATION = 50000
    DT = 0.033

    link_1: float = 105.  # pixels
    link_2: float = 65.  # pixels
    link_3: float = 60.
    _theta_0: float      # radians
    _theta_1: float      # radians
    _theta_2: float

    def __init__(self) -> None:
        # internal variables
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.all_theta_2: List[float] = []

        self.theta_0 = 0.
        self.theta_1 = pi
        self.theta_2 = 0.

    # Getters/Setters
    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self.all_theta_0.append(value)
        self._theta_0 = value
        # Check limits
        assert self.check_angle_limits(value), \
            f'Joint 0 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_0) < self.MAX_VELOCITY, \
            f'Joint 0 Velocity {self.max_velocity(self.all_theta_0)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_0) < self.MAX_ACCELERATION, \
            f'Joint 0 Accel {self.max_acceleration(self.all_theta_0)} exceeds acceleration limit'

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self.all_theta_1.append(value)
        self._theta_1 = value
        assert self.check_angle_limits(value), \
            f'Joint 1 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_1) < self.MAX_VELOCITY, \
            f'Joint 1 Velocity {self.max_velocity(self.all_theta_1)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_1) < self.MAX_ACCELERATION, \
            f'Joint 1 Accel {self.max_acceleration(self.all_theta_1)} exceeds acceleration limit'

    @property
    def theta_2(self) -> float:
        return self._theta_2

    @theta_2.setter
    def theta_2(self, value: float) -> None:
        self.all_theta_2.append(value)
        self._theta_2 = value
        assert self.check_angle_limits(value), \
            f'Joint 2 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_2) < self.MAX_VELOCITY, \
            f'Joint 2 Velocity {self.max_velocity(self.all_theta_2)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_2) < self.MAX_ACCELERATION, \
            f'Joint 2 Accel {self.max_acceleration(self.all_theta_2)} exceeds acceleration limit'

    # Kinematics
    def joint_1_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 1
        """
        return self.link_1 * np.cos(self.theta_0), self.link_1 * np.sin(self.theta_0)

    def joint_2_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 2
        """
        x = self.link_1 * np.cos(self.theta_0) + self.link_2 * np.cos(self.theta_0 + self.theta_1)
        y = self.link_1 * np.sin(self.theta_0) + self.link_2 * np.sin(self.theta_0 + self.theta_1)
        return x, y

    def joint_3_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 3
        """
        x = self.link_1 * np.cos(self.theta_0) + self.link_2 * np.cos(self.theta_0 + self.theta_1) + self.link_3 * np.cos(self.theta_0 + self.theta_1 + self.theta_2)
        y = self.link_1 * np.sin(self.theta_0) + self.link_2 * np.sin(self.theta_0 + self.theta_1) + self.link_3 * np.sin(self.theta_0 + self.theta_1 + self.theta_2)
        return x, y

    @classmethod
    def forward(cls, theta_0: float, theta_1: float, theta_2: float) -> Tuple[float, float]:
        """
        Compute the x, y position of the end of the links from the joint angles
        """
        x = cls.link_1 * np.cos(theta_0) + cls.link_2 * np.cos(theta_0 + theta_1) + cls.link_3 * np.cos(theta_0 + theta_1 + theta_2)
        y = cls.link_1 * np.sin(theta_0) + cls.link_2 * np.sin(theta_0 + theta_1) + cls.link_3 * np.sin(theta_0 + theta_1 + theta_2)

        return x, y

    @classmethod
    def inverse(cls, x: float, y: float) -> Tuple[float, float]:
        """
        Compute the joint angles from the position of the end of the links
        """
        phi_e =  atan2(y,x)
        xw = x - cls.link_3*cos(phi_e)
        yw = y - cls.link_3*sin(phi_e)

        value: float = (xw ** 2 + yw ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)/(2 * cls.link_1 * cls.link_2)

        # solve for edge case where cos > 1
        if (abs(value)) > 1.0:
            while abs(value) > 1:
                phi_e = phi_e - pi/10
                # print(value)
                xw = x - cls.link_3*cos(phi_e)
                yw = y - cls.link_3*sin(phi_e)
                value: float = (xw ** 2 + yw ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)/(2 * cls.link_1 * cls.link_2)
            theta_1 = np.arccos(value)

        else:
            theta_1 = np.arccos((xw ** 2 + yw ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)/(2 * cls.link_1 * cls.link_2))


        theta_0 = np.arctan2(yw, xw) - np.arctan((cls.link_2 * np.sin(theta_1)) /
                        (cls.link_1 + cls.link_2 * np.cos(theta_1)))

        theta_2 = phi_e - theta_0 - theta_1

        return theta_0, theta_1, theta_2

    @classmethod
    def check_angle_limits(cls, theta: float) -> bool:
        return cls.JOINT_LIMITS[0] < theta < cls.JOINT_LIMITS[1]

    @classmethod
    def max_velocity(cls, all_theta: List[float]) -> float:
        return float(max(abs(np.diff(all_theta) / cls.DT), default=0.))

    @classmethod
    def max_acceleration(cls, all_theta: List[float]) -> float:
        return float(max(abs(np.diff(np.diff(all_theta)) / cls.DT / cls.DT), default=0.))

    @classmethod
    def min_reachable_radius(cls) -> float:
        return max(cls.link_1 - cls.link_2, 0)

    @classmethod
    def max_reachable_radius(cls) -> float:
        return cls.link_1 + cls.link_2 + cls.link_3


class World:
    def __init__(
        self,
        width: int,
        height: int,
        robot_origin: Tuple[int, int],
        goal: Tuple[int, int],
        robot: Robot,
        obstacle_list,
        wall_padding: int = 100
        ) -> None:

        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.goal = goal
        self.robot = robot

        # setting the borders - offset inward from the world
        self.border = [[wall_padding,wall_padding],
                      [wall_padding,self.height-wall_padding],
                      [self.width-wall_padding,self.height-wall_padding],
                      [self.width-wall_padding, wall_padding],
                      [wall_padding,wall_padding]]
        
        b1 = [[wall_padding,wall_padding],
                [wall_padding,self.height-wall_padding]]

        b2 = [[wall_padding,self.height-wall_padding],
                [self.width-wall_padding,self.height-wall_padding]]

        b3 = [[self.width-wall_padding,self.height-wall_padding],
                [self.width-wall_padding, wall_padding]]

        b4 = [[self.width-wall_padding, wall_padding],
                [wall_padding,wall_padding]]

        self.wall_padding = wall_padding


        self.obstacles = [b1, b2, b3 ,b4]

        # fetch other obstacles passed into the class 
        for i in obstacle_list:
           self.get_obstacle(i) 
    
        # define the working area of the robot - area within the border
        self.work_area = mplPath.Path(np.array(self.border))

    def get_obstacle(self, obs):
        """
        Adds obstacle points to the main obstacle list
        """
        for i in range(len(obs)-1):
            self.obstacles.append([obs[i], obs[i+1]])
        


    def move_goal(self) -> bool:
        """
        Move the goal using the arrow keys
        """
        new_goal = self.robot_origin
        for event in pygame.event.get():
            move_goal = 50
            # Keypress
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: # if escape is pressed, quit the program
                    return False, False
                if event.key == pygame.K_LEFT: # move goal left
                    xCoord = self.goal[0] - move_goal
                    yCoord = self.goal[1]
                    new_goal = (xCoord,yCoord)
                    
                if event.key == pygame.K_RIGHT: # move goal right
                    xCoord = self.goal[0] + move_goal
                    yCoord = self.goal[1]
                    new_goal = (xCoord,yCoord)
                    
                if event.key == pygame.K_UP: # move goal up
                    xCoord = self.goal[0]
                    yCoord = self.goal[1] + move_goal
                    new_goal = (xCoord,yCoord)
                    
                if event.key == pygame.K_DOWN: # move goal down
                    xCoord = self.goal[0]
                    yCoord = self.goal[1] - move_goal
                    new_goal = (xCoord,yCoord)

                # if goal is beyond the reach of the robot, do nothing
                if sqrt(new_goal[0]**2+new_goal[1]**2)>self.robot.max_reachable_radius():
                    # print("Out of bounds")
                    return True, True
                else:
                    self.goal = new_goal
                    return False, True
        return False, True


    
    
    def check_goal(self, work_area, goal):
        """
        Check if the goal is within the working area as described by the border
        """
        new_point = self.convert_to_display(goal)
        # print(work_area)
        if(not work_area.contains_point([new_point[0],new_point[1]])):
            # print("Out of workspace")
            return True
        return False

    def generate_new_goal(self):
        """
        Generate a new goal
        """
        theta = (np.random.random() + np.finfo(float).eps) * 2 * np.pi
        # Ensure point is reachable
        min_radius = self.robot.link_1-self.robot.link_2
        max_radius = self.robot.link_1+self.robot.link_2
        r = np.random.uniform(low=min_radius, high=max_radius)
        
        x = int(r * np.cos(theta))
        y = int(r * np.sin(theta))

        # print(goal)
        return x, y

    def get_new_goal(self):
        """
        Generates a new goal and sets it only if the goal is within the work area
        """
        goal = (self.height, self.width)

        while (int(self.width/2-self.wall_padding) < goal[0] or int(-self.width/2+self.wall_padding) > goal[0] or \
            int(self.height/2-self.wall_padding) < goal[1] or int(-self.height/2+self.wall_padding > goal[1])):
            goal = self.generate_new_goal()

        return goal

    def convert_to_display(
            self, point: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
        """
        Convert a point from the robot coordinate system to the display coordinate system
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin

        return int(offset_x + robot_x), int(offset_y - robot_y)

class Controller:
    def __init__(self, goal: Tuple[int, int]) -> None:
        self.goal = goal
        self.goal_theta_0, self.goal_theta_1, self.goal_theta_2 = Robot.inverse(self.goal[0], self.goal[1])
        
        # define the errors as class variables to be used later
        self.theta_0_error = 0
        self.theta_1_error = 0
        self.theta_2_error = 0

    def step(self, robot: Robot) -> Robot:
        """
        Simple P controller
        """
        self.theta_0_error = self.goal_theta_0 - robot.theta_0
        self.theta_1_error = self.goal_theta_1 - robot.theta_1
        self.theta_2_error = self.goal_theta_2 - robot.theta_2

        robot.theta_0 += self.theta_0_error / 15
        robot.theta_1 += self.theta_1_error / 15
        robot.theta_2 += self.theta_2_error / 15

        return robot

class Telemetry():
    def __init__(
        self,
        controller: Controller,
        robot: Robot
        ) -> None:
        self.robot = robot
        self.controller = controller
        self.points = []
        self.theta_0_move = []
        self.theta_1_move = []
        self.theta_2_move = []
        self.velocity = []
        self.acceleration = []


    def update_telemetry(self, controller: Controller, robot: Robot):
        """
        All telemetry information like current position, velocity and acceleration
        is updated and stored using this function
        """
        self.points.append(robot.joint_3_pos())
        self.theta_0_move.append(controller.theta_0_error / 10)
        self.theta_1_move.append(controller.theta_1_error / 10)
        self.theta_2_move.append(controller.theta_2_error / 10)


        # velocity calculation        
        vx = [-robot.link_1*sin(robot.theta_0) - robot.link_2*sin(robot.theta_0 + robot.theta_1) - robot.link_3*sin(robot.theta_0+robot.theta_1+robot.theta_2),
              -robot.link_2*sin(robot.theta_0 + robot.theta_1)-robot.link_2*sin(robot.theta_0 + robot.theta_1 + robot.theta_2),
              -robot.link_3*sin(robot.theta_0 + robot.theta_1 + robot.theta_2)]

        vy = [robot.link_1*cos(robot.theta_0) + robot.link_2*cos(robot.theta_0 + robot.theta_1) + robot.link_3*cos(robot.theta_0+robot.theta_1+robot.theta_2),
              robot.link_2*cos(robot.theta_0 + robot.theta_1)-robot.link_2*cos(robot.theta_0 + robot.theta_1 + robot.theta_2),
              robot.link_3*cos(robot.theta_0 + robot.theta_1 + robot.theta_2)]

        # form the jacobian matrix
        jacob = np.array([vx,vy])

        if len(self.theta_0_move) > 1:

            # matrix of all the 3 theta values at every step of the trajectory
            h = np.vstack([np.array(self.theta_0_move).T, np.array(self.theta_1_move).T, np.array(self.theta_1_move).T])
            
            # multiply the h matrix with the jacobian to get the velocity at every point on the trajectory
            self.velocity = np.matmul(jacob, h)
            # print(self.velocity.shape)
            acceleration = np.zeros_like(self.velocity)

            # acceleration is calculated using the difference of the velocity matrix
            acceleration_calc = np.diff(self.velocity, axis=1)
            acceleration[:, :-1] = acceleration_calc
            self.acceleration = acceleration

class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    def __init__(self, world: World, telemetry: Telemetry) -> None:
        """
        Note: while the Robot and World have the origin in the center of the
        visualization, rendering places (0, 0) in the top left corner.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.telemetry = telemetry
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Gherkin Challenge')
        self.font = pygame.font.SysFont('freesansbolf.tff', 30)

    def draw_arrow(self, start: Tuple[int, int], angle: float, scale: float, color: str='red') -> None:
        """
        Define a basic arrow - use this to visualize the velocity and acceleration in real time
        by scaling, rotating and translating the arrow according to the end effector
        """
        polygon = [[0,-1],
                    [0,0],
                    [0,1],
                    [50,1],
                    [50,4],
                    [55,0],
                    [50,-4],
                    [50,-1]]

        polygon = np.array(polygon) * scale   
        
        
        rot_matrix = np.array([[cos(angle),-sin(angle)],[sin(angle),cos(angle)]])

        new_polygon = []

        for i in polygon:
            new_polygon.append(np.array(list(start)) + i @ rot_matrix)
        
        # change the color of the arrow
        if color == 'red':
            pygame.draw.polygon(self.screen, self.RED, new_polygon, 1)
        elif color == 'blue':
            pygame.draw.polygon(self.screen, self.BLUE, new_polygon, 1)
        else:
            pygame.draw.polygon(self.screen, self.BLACK, new_polygon, 1)    
    
    
    def display_world(self) -> None:
        """
        Display the world
        """
                    
        goal = self.world.convert_to_display(self.world.goal)
        pygame.draw.circle(self.screen, self.RED, goal, 6)      
        
        # display obstacles
        for i in range(len(self.world.obstacles)):
            for j in range(len(self.world.obstacles[i])-1):
                # print("Point 1: ",self.world.obstacles[i][j]," Point 2: ",self.world.obstacles[i][j+1])
                pygame.draw.line(self.screen, self.RED, self.world.obstacles[i][j], self.world.obstacles[i][j+1],5)

        # display borders
        for i in range(len(self.world.border)-1):
            pygame.draw.line(self.screen, self.BLACK, self.world.border[i], self.world.border[i+1],5)

    def display_robot(self, robot: Robot) -> None:
        """
        Display the robot
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.joint_1_pos())
        j2 = self.world.convert_to_display(robot.joint_2_pos())
        j3 = self.world.convert_to_display(robot.joint_3_pos())

        # Draw joint 0
        pygame.draw.circle(self.screen, self.BLACK, j0, 4)
        # Draw link 1
        pygame.draw.line(self.screen, self.BLACK, j0, j1, 2)
        # Draw joint 1
        pygame.draw.circle(self.screen, self.BLACK, j1, 4)
        # Draw link 2
        pygame.draw.line(self.screen, self.BLACK, j1, j2, 2)
        # Draw joint 2
        pygame.draw.circle(self.screen, self.BLACK, j2, 4)
        # Draw link 3
        pygame.draw.line(self.screen, self.BLACK, j2, j3, 2)
        # Draw joint 3
        pygame.draw.circle(self.screen, self.BLACK, j3, 4)



    def draw_path(self, telemetry: Telemetry):
        """
        Draw the path of the current trajectory of the robot
        Also plots the arrows for velocity and acceleration
        """
        # print(len(telemetry.points))
        for i in range(len(telemetry.points)-1):
            # print(i)
            pygame.draw.circle(self.screen, self.BLUE, self.world.convert_to_display(telemetry.points[i]), 1)   
            
            # print(len(self.controller.slopes))
        if len(telemetry.points) > 2:

            point1 = telemetry.points[-1]
            point2 = telemetry.points[-2]

            velocity = np.sqrt(np.sum((telemetry.velocity**2), axis=0))
            vel = velocity[-1]

            angle = atan2((point1[1]-point2[1]),(point1[0]-point2[0]))
            scale = 0.1

            # scale the velocity arrow
            if vel*scale < 0.75:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, 0.75, 'blue')
            elif vel*scale > 7.5:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, 7.5, 'blue')
            else:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, vel*scale, 'blue')
            
            point1 = telemetry.velocity[-1]
            point2 = telemetry.velocity[-2]

            acceleration = np.sqrt(np.sum((telemetry.acceleration**2), axis=0))
            acc = acceleration[-1]

            angle = atan2((point1[1]-point2[1]),(point1[0]-point2[0]))
            scale = 0.1

            # scale the acceleration arrow
            if acc*scale < 0.75:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, 0.75, 'red')
            elif acc*scale > 3:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, 3, 'red')
            else:
                self.draw_arrow(self.world.convert_to_display(telemetry.points[i]),angle, vel*scale, 'red')

    def display_telemetry(self, telemetry: Telemetry, fig):
        """
        Graph the current velocity and acceleration of the robot end effector
        """

        # handle edge case
        if len(telemetry.velocity) == 0:
            telemetry.velocity = np.zeros((2,1))

        if len(telemetry.acceleration) == 0:
            telemetry.acceleration = np.zeros((2,1))

        vel = np.sqrt(np.sum(telemetry.velocity**2, axis=0))
        acc = np.sqrt(np.sum(telemetry.acceleration**2, axis=0))
        
        xaxis = np.arange(0, len(vel), 1)

        # TODO: improve performance, faster render

        # print(type(vel))
        ax = fig.add_subplot(1,1,1)

        ax.plot(xaxis, vel, 'b', label='velocity')
        ax.plot(xaxis, acc, 'r', label='acceleration')

        plt.title("Telemetry Data")
        plt.legend(loc="upper right")
        plt.ylabel('Magnitude')
        plt.xlabel('steps')
        plt.show(block=False)
        plt.pause(0.0000001)
        # plt.close()
        plt.clf()


    def update_display(self, 
                       robot: Robot,
                       telemetry: Telemetry,
                       success: bool,
                       check_goal_workspace: bool,
                       path_collision: bool,
                       fig,
                       goal_out_of_bounds: bool
                       ) -> bool:

        self.screen.fill(self.WHITE)

        text = self.font.render('Use arrow keys to move goal', True, self.BLACK)
        self.screen.blit(text, (400, 1))

        if not path_collision:
            text = self.font.render('Collision in Path, cannot change goal!', True, self.BLACK)
            self.screen.blit(text, (1, 50))        

        if check_goal_workspace:
            text = self.font.render('Out of Workspace!', True, self.BLACK)
            self.screen.blit(text, (1, 30))

        if goal_out_of_bounds:
            text = self.font.render('Out of Robot Bounds!', True, self.BLACK)
            self.screen.blit(text, (1, 30))
            time.sleep(0.5)

        self.display_world()

        self.display_robot(robot)
        
        # comment to hide the path taken
        self.draw_path(telemetry)

        # comment to hide the telemetry graph
        self.display_telemetry(telemetry, fig)

        if success:
            text = self.font.render('Success!', True, self.BLACK)
            self.screen.blit(text, (1, 1))
            # self.display_telemetry(telemetry, fig)

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Configuration:
    """
    This class checks for intersection of the robot arms with the obstacles
    Taken from https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/
    """
    def __init__(
        self,
        robot: Robot,
        world: World,
        controller: Controller,
        telemetry: Telemetry        
    ) -> None:
        self.robot = robot
        self.world = world
        self.controller = controller
        self.telementry = telemetry
        
    # taken from source
    def on_segment(self, p, q, r):
        if r[0] <= max(p[0], q[0]) and r[0] >= min(p[0], q[0]) and r[1] <= max(p[1], q[1]) and r[1] >= min(p[1], q[1]):
            return True
        return False

    def orientation(self, p, q, r):
        val = ((q[1] - p[1]) * ((r[0] - q[0]))) - ((q[0] - p[0]) * (r[1] - q[1]))
        if val == 0 : return 0
        return 1 if val > 0 else -1

    def intersects(self, seg1, seg2):

        p1, q1 = seg1
        p2, q2 = seg2

        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and self.on_segment(p1, q1, p2) : return True
        if o2 == 0 and self.on_segment(p1, q1, q2) : return True
        if o3 == 0 and self.on_segment(p2, q2, p1) : return True
        if o4 == 0 and self.on_segment(p2, q2, q1) : return True

        return False

    # check if goal is intersecting
    def get_goal_config(self, robot: Robot, world: World, goal):
        
        theta_0, theta_1, theta_2 = robot.inverse(goal[0], goal[1])

        j1 = world.convert_to_display((self.robot.link_1*np.cos(theta_0), self.robot.link_1 * np.sin(theta_0)))

        x2 = self.robot.link_1 * np.cos(theta_0) + self.robot.link_2 * np.cos(theta_0 + theta_1)
        y2 = self.robot.link_1 * np.sin(theta_0) + self.robot.link_2 * np.sin(theta_0 + theta_1)
        j2 = world.convert_to_display((x2,y2))

        x3 = self.robot.link_1 * np.cos(theta_0) + self.robot.link_2 * np.cos(theta_0 + theta_1) + self.robot.link_3 * np.cos(theta_0 + theta_1 + theta_2)
        y3 = self.robot.link_1 * np.sin(theta_0) + self.robot.link_2 * np.sin(theta_0 + theta_1) + self.robot.link_3 * np.sin(theta_0 + theta_1 + theta_2)
        j3 = world.convert_to_display((x3,y3))

        origin = (world.width/2,world.height/2)

        seg1 = (origin, j1)
        seg2 = (j1,j2)
        seg3 = (j2,j3)
        
        for i in world.obstacles:
            if(self.intersects(i,seg2) or self.intersects(i,seg1) or self.intersects(i,seg3)):
                return True
        return False

    # check full path of the robot. If collision -> False
    def check_full_path(self, world: World, telemetry: Telemetry):
        
        # self.controller = Controller(self.world.goal, [[10,100]], [], [])
        self.controller = Controller(self.world.goal)

        while True:
            # self.controller = Controller(self.world.goal, self.controller.points, [], [])
            self.controller = Controller(self.world.goal)            
            self.robot = self.controller.step(self.robot)

            telemetry.update_telemetry(self.controller, self.robot)            
            
            if np.allclose(self.robot.joint_3_pos(), world.goal, atol=0.25):
                break

        # points = self.controller.points
        points = telemetry.points

        for i in points:
            if self.get_goal_config(self.robot,self.world,i):
                return False
        return True
        # print("Out of while")

class Runner:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        world: World,
        vis: Visualizer,
        telemetry: Telemetry,
        conf: Configuration
        ) -> None:
        self.robot = robot
        self.controller = controller
        self.world = world
        self.vis = vis
        self.telemetry = telemetry
        self.conf = conf
        self.fig = plt.figure()

    def run(self) -> None:

        # initialise booleans
        running = True
        path_collision = True
        check_goal_workspace = False
        goal_out_of_bounds = False
        
        while running:

            # Step the controller
            self.controller = Controller(self.world.goal)
            self.robot = self.controller.step(self.robot)

            # update telemtry data
            self.telemetry.update_telemetry(self.controller, self.robot)

            # save the current goal before change
            curr_goal = self.world.goal

            # Check collisions
            # TODO -> done in Configuration class

            # Check success
            success = self.check_success(self.robot, self.world.goal)

            if success:
                # if the end effector has reached the goal, reset all telemetry parameters
                self.telemetry.points = []
                self.telemetry.theta_0_move = []
                self.telemetry.theta_1_move = []
                self.telemetry.theta_2_move = []

                # if end effector is at the goal, generate new goal
                self.world.goal = self.world.get_new_goal()
            
            # move goal using arrow keys, set exit condition here
            goal_out_of_bounds, running = self.world.move_goal()

            if self.world.goal != curr_goal:
                # if goal has been changed

                # check if new goal is within the border
                check_goal_workspace = self.world.check_goal(self.world.work_area, self.world.goal)

                # create class copies
                conf_copy = copy.deepcopy(self.conf)
                world_copy = copy.deepcopy(self.world)
                telemetry_copy = copy.deepcopy(self.telemetry)

                # pass the class copies to check if the new path generated is collision free or not
                path_collision = conf_copy.check_full_path(world_copy, telemetry_copy)

            # if any of the conditions are not met, do not change the goal
            # keep the previous goal
            if check_goal_workspace or  (not path_collision) or goal_out_of_bounds:
                self.world.goal = curr_goal



            # Update the display
            self.vis.update_display(self.robot,
                                              self.telemetry,
                                              success,
                                              check_goal_workspace,
                                              path_collision,
                                              self.fig,
                                              goal_out_of_bounds)

            if not running:
                pygame.quit()

            # sleep for Robot DT seconds, to force update rate
            time.sleep(self.robot.DT)

    @staticmethod
    def check_success(robot: Robot, goal: Tuple[int, int]) -> bool:
        """
        Check that robot's joint 2 is very close to the goal.
        Don't not use exact comparision, to be robust to floating point calculations.
        """
        return np.allclose(robot.joint_3_pos(), goal, atol=0.25)

    def cleanup(self) -> None:
        self.vis.cleanup()


def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[int, int]:
    """
    Generate a random goal that is reachable by the robot arm
    """
    # Ensure theta is not 0
    theta = (np.random.random() + np.finfo(float).eps) * 2 * np.pi
    # Ensure point is reachable
    r = np.random.uniform(low=min_radius, high=max_radius)

    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))

    return x, y


def main() -> None:
    height = 700
    width = 700
    wall_padding = 100

    # define points of all obstacles
    obs1 = [[150,150],
            [150,250],
            [160,250],
            [160,150],
            [150,150]]

    obs2 = [[500,100],
            [500,200],
            [510,200],
            [510,100],
            [500,100]]

    obs3 = [[150,500],
            [150,450],
            [200,475]]

    obstacle_list = [obs1, obs2, obs3]

    robot_origin = (int(width / 2), int(height / 2))
 
    robot = Robot()

    # initial goal position, after this goals will be randomly generated
    goal = robot.joint_3_pos()

    controller = Controller(goal)
    world = World(width, height, robot_origin, goal, robot, obstacle_list, wall_padding)
    telemetry = Telemetry(controller, robot)
    vis = Visualizer(world, telemetry)
    conf = Configuration(robot,world, controller, telemetry)


    runner = Runner(robot, controller, world, vis, telemetry, conf)

    try:
        runner.run()
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass
    finally:
        runner.cleanup()


if __name__ == '__main__':
    main()
