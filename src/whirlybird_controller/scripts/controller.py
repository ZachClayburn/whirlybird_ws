#!/usr/bin/env python
# license removed for brevity


# This file is a basic structure to write a controller that
# communicates with ROS. It will be the students responsibility
# tune the gains and fill in the missing information

# As an example this file contains PID gains, but other
# controllers use different types of gains so the class
# will need to be modified to accomodate those changes

from __future__ import division
import rospy
import numpy as np
import control.matlab as ctrl
from whirlybird_msgs.msg import Command
from whirlybird_msgs.msg import Whirlybird
from std_msgs.msg import Float32


class Controller:

    def __init__(self):

        # get parameters
        try:
            param_namespace = '/whirlybird'
            self.param = rospy.get_param(param_namespace)
        except KeyError:
            rospy.logfatal('Parameters not set in ~/whirlybird namespace')
            rospy.signal_shutdown('Parameters not set')

        g = self.param['g']
        l1 = self.param['l1']
        l2 = self.param['l2']
        m1 = self.param['m1']
        m2 = self.param['m2']
        d = self.param['d']
        h = self.param['h']
        r = self.param['r']
        Jx = self.param['Jx']
        Jy = self.param['Jy']
        Jz = self.param['Jz']
        km = self.param['km']

        # Tuning variables
        damping_ratio_theta = 0.8
        damping_ratio_psi = 0.9  # Yaw
        damping_ratio_phi = 0.8  # Roll
        rise_time_theta = 1.0
        rise_time_phi = 0.35
        bandwidth_separation = 7.0
        int_pole_lat = 0.1
        int_pole_lon = 0.1

        denom = (m1 * l1 ** 2 + m2 * l2 ** 2 + Jy)
        Fe = (m1 * l1 - m2 * l2) * g / l1
        b_psi = l1 * Fe / denom
        self.Fe = Fe

        natural_frequency_theta = np.pi / (2 * rise_time_theta * (1 - damping_ratio_theta ** 2) ** (1 / 2))
        natural_frequency_phi = np.pi / (2 * rise_time_phi * (1 - damping_ratio_phi ** 2) ** (1 / 2))
        rise_time_psi = bandwidth_separation * rise_time_phi
        natural_frequency_psi = np.pi / (2 * rise_time_psi * (1 - damping_ratio_psi ** 2) ** (1 / 2))

        # State space controller
        A_lat = np.asarray([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [b_psi, 0, 0, 0],
        ])
        B_lat = np.asarray([
            [0],
            [0],
            [1 / Jx],
            [0],
        ])
        C_lat = np.asarray([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])
        P_lat = np.asarray([
            natural_frequency_phi * (-damping_ratio_phi + 1j * np.sqrt(1 - damping_ratio_phi ** 2)),
            natural_frequency_phi * (-damping_ratio_phi - 1j * np.sqrt(1 - damping_ratio_phi ** 2)),
            natural_frequency_psi * (-damping_ratio_psi + 1j * np.sqrt(1 - damping_ratio_psi ** 2)),
            int_pole_lat,
            natural_frequency_psi * (-damping_ratio_psi - 1j * np.sqrt(1 - damping_ratio_psi ** 2)),
        ])

        A_I_lat, B_I_lat = self.augment_matrices(A_lat, B_lat, C_lat[1:2, 0:4])

        self.check_state_space(A_I_lat, B_I_lat)
        K_lat = ctrl.place(A_I_lat, B_I_lat, P_lat)
        print("latitudinal gains :", K_lat)
        size_lat = A_I_lat.shape[0]
        self.k_lat = K_lat[:, 0:size_lat - 1]
        self.k_int_lat = K_lat.item(size_lat - 1)

        A_lon = np.asarray([
            [0, 1],
            [Fe * np.sin(0) / denom, 0]
        ])
        B_lon = np.asarray([
            [0],
            [l1 / denom],
        ])
        C_lon = np.asarray([[1, 0]])
        P_lon = np.asarray([
            natural_frequency_theta * (-damping_ratio_theta + 1j * np.sqrt(1 - damping_ratio_theta ** 2)),
            natural_frequency_theta * (-damping_ratio_theta - 1j * np.sqrt(1 - damping_ratio_theta ** 2)),
            int_pole_lon
        ])

        A_I_lon, B_I_lon = self.augment_matrices(A_lon, B_lon, C_lon)

        self.check_state_space(A_I_lon, B_I_lon)
        K_lon = ctrl.place(A_I_lon, B_I_lon, P_lon)
        print("longitudinal gains: ", K_lon)
        size_lon = A_I_lon.shape[0]
        self.k_lon = K_lon[:, 0:size_lon - 1]
        self.k_int_lon = K_lon.item(size_lon - 1)

        self.sigma = 0.05

        self.anti_windup_theta = 0.05
        self.anti_windup_psi = 0.08  # Yaw

        self.prev_phi = 0.0
        self.prev_phi_dirty_dot = 0.0

        self.int_theta = 0.0
        self.prev_error_theta = 0.0
        self.prev_error_theta_dirty_dot = 0.0
        self.theta_r = 0.0
        self.prev_theta = 0.0
        self.prev_theta_dirty_dot = 0.0

        self.int_psi = 0.0
        self.prev_error_psi = 0.0
        self.prev_error_psi_dirty_dot = 0.0
        self.psi_r = 0.0
        self.prev_psi = 0.0
        self.prev_psi_dirty_dot = 0.0

        self.prev_time = rospy.Time.now()

        self.command_sub_ = rospy.Subscriber('whirlybird', Whirlybird, self.whirlybird_callback, queue_size=5)
        self.psi_r_sub_ = rospy.Subscriber('psi_r', Float32, self.psi_r_callback, queue_size=5)
        self.theta_r_sub_ = rospy.Subscriber('theta_r', Float32, self.theta_r_callback, queue_size=5)
        self.command_pub_ = rospy.Publisher('command', Command, queue_size=5)
        while not rospy.is_shutdown():
            # wait for new messages and call the callback when they arrive
            rospy.spin()

    @staticmethod
    def check_state_space(A, B):
        cont_mat = ctrl.ctrb(A, B)
        if np.linalg.matrix_rank(cont_mat) != cont_mat.shape[0]:
            rospy.logfatal('System is not controllable!')
            rospy.signal_shutdown('bad system')

    @staticmethod
    def augment_matrices(A, B, C):
        temp = np.concatenate((A, C))
        A_I = np.concatenate((temp, np.zeros((temp.shape[0], 1))), 1)

        B_I = np.concatenate((B, np.zeros((1, 1))))

        return A_I, B_I

    def theta_r_callback(self, msg):
        self.theta_r = msg.data

    def psi_r_callback(self, msg):
        self.psi_r = msg.data

    def whirlybird_callback(self, msg):
        g = self.param['g']
        l1 = self.param['l1']
        l2 = self.param['l2']
        m1 = self.param['m1']
        m2 = self.param['m2']
        d = self.param['d']
        h = self.param['h']
        r = self.param['r']
        Jx = self.param['Jx']
        Jy = self.param['Jy']
        Jz = self.param['Jz']
        km = self.param['km']

        phi = msg.roll
        theta = msg.pitch
        psi = msg.yaw

        # Calculate dt (This is variable)
        now = rospy.Time.now()
        dt = (now - self.prev_time).to_sec()
        self.prev_time = now

        ##################################
        # Implement your controller here

        # Feedback linearization
        equilibrium_force = self.Fe * np.cos(theta)

        theta_dot = self.dirty_derivative(theta, self.prev_theta, self.prev_theta_dirty_dot, dt)
        phi_dot = self.dirty_derivative(phi, self.prev_phi, self.prev_phi_dirty_dot, dt)
        psi_dot = self.dirty_derivative(psi, self.prev_psi, self.prev_psi_dirty_dot, dt)

        state_lon = np.asarray([
            [theta],
            [theta_dot],
        ])

        error_theta = self.theta_r - theta
        error_theta_dot = self.dirty_derivative(error_theta, self.prev_error_theta, self.prev_error_theta_dirty_dot, dt)
        if error_theta_dot <= self.anti_windup_theta:
            self.int_theta += (dt / 2) * (error_theta + self.prev_error_theta)
        self.prev_error_theta = error_theta
        self.prev_error_theta_dirty_dot = error_theta_dot

        force = equilibrium_force - np.dot(self.k_lon, state_lon).item(0) - self.k_int_lon * self.int_theta

        state_lat = np.asarray([
            [phi],
            [psi],
            [phi_dot],
            [psi_dot],
        ])

        error_psi = self.psi_r - psi
        error_psi_dot = self.dirty_derivative(error_psi, self.prev_error_psi, self.prev_error_psi_dirty_dot, dt)
        if error_psi_dot <= self.anti_windup_psi:
            self.int_psi += (dt / 2) * (error_psi + self.prev_error_psi)
            print("Integrating psi")
        else:
            print("Skipping psi integrator")
        self.prev_error_psi = error_psi
        self.prev_error_psi_dirty_dot = error_psi_dot

        torque = np.dot(-self.k_lat, state_lat).item(0) - self.k_int_lat * self.int_psi

        # print('Lat: {}\nlon: {}'.format(self.int_psi, self.int_theta))

        u = np.array([
            [force],
            [torque]
        ])

        transform = np.array([
            [1, 1],
            [d, -d]
        ])
        left_force, right_force = np.linalg.solve(transform, u).T.tolist()[0]

        self.prev_theta = theta
        self.prev_phi = phi
        self.prev_psi = psi

        self.prev_theta_dirty_dot = theta_dot
        self.prev_phi_dirty_dot = phi_dot
        self.prev_psi_dirty_dot = psi_dot

        ##################################

        sat_max = 0.7

        # Scale Output
        l_out = left_force / km
        if l_out < 0:
            l_out = 0
        elif l_out > sat_max:
            # rospy.logerr('Left force saturated!')
            l_out = sat_max

        r_out = right_force / km
        if r_out < 0:
            r_out = 0
        elif r_out > sat_max:
            # rospy.logerr('Right force saturated!')
            r_out = sat_max

        # Pack up and send command
        command = Command()
        command.left_motor = l_out
        command.right_motor = r_out
        self.command_pub_.publish(command)

    def dirty_derivative(self, this_val, last_val, last_dirty_dot, dt):
        return ((2 * self.sigma - dt) / (2 * self.sigma + dt)) * last_dirty_dot + \
               (2 / (2 * self.sigma + dt)) * (this_val - last_val)


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)
    # try:
    controller = Controller()
    # except BaseException:
    #     raise rospy.ROSInterruptException()
    # pass
