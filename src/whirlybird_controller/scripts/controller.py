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
        damping_ratio_psi = 0.7  # Yaw
        damping_ratio_phi = 0.8  # Roll
        self.Fe = (m1 * l1 - m2 * l2) * g / l1

        b_theta = l1 / (m1 * l1 ** 2 + m2 * l2 ** 2 + Jy)
        rise_time_theta = 1.2
        natural_frequency_theta = np.pi / (2 * rise_time_theta * (1 - damping_ratio_theta ** 2) ** (1 / 2))

        b_phi = 1 / Jx
        rise_time_phi = 0.4
        natural_frequency_phi = np.pi / (2 * rise_time_phi * (1 - damping_ratio_phi ** 2) ** (1 / 2))

        b_psi = l1 * self.Fe / (m1 * l1 ** 2 + m2 * l2 ** 2 + Jz)
        bandwidth_separation = 6.0
        rise_time_psi = bandwidth_separation * rise_time_phi
        natural_frequency_psi = np.pi / (2 * rise_time_psi * (1 - damping_ratio_psi ** 2) ** (1 / 2))

        self.sigma = 0.05

        self.anti_windup_theta = 0.05
        self.anti_windup_psi = 0.04  # Yaw
        self.anti_windup_phi = 0.0001  # Roll

        # Roll Gains
        self.P_phi_ = natural_frequency_phi ** 2 / b_phi
        self.I_phi_ = 0.08  # FIXME Tune this
        self.D_phi_ = 2.0 * damping_ratio_phi * natural_frequency_phi / b_phi
        self.Int_phi = 0.0
        self.prev_phi = 0.0
        self.prev_phi_dirty_dot = 0.0
        self.prev_phi_error = 0.0
        self.prev_phi_error_dirty_dot = 0.0

        # Pitch Gains
        self.theta_r = 0.0
        self.P_theta_ = natural_frequency_theta ** 2 / b_theta
        self.I_theta_ = 2.0
        self.D_theta_ = 2 * damping_ratio_theta * natural_frequency_theta / b_theta
        self.prev_theta = 0.0
        self.prev_theta_dirty_dot = 0.0
        self.Int_theta = 0.0
        self.prev_theta_error = 0.0
        self.prev_theta_error_dirty_dot = 0.0

        # Yaw Gains
        self.psi_r = 0.0
        self.P_psi_ = natural_frequency_psi ** 2 / b_psi
        self.I_psi_ = 0.05  # FIXME Tune this
        self.D_psi_ = 2 * damping_ratio_psi * natural_frequency_psi / b_psi
        self.prev_psi = 0.0
        self.prev_psi_dirty_dot = 0.0
        self.Int_psi = 0.0
        self.prev_psi_error = 0.0
        self.prev_psi_error_dirty_dot = 0.0

        self.prev_time = rospy.Time.now()

        self.command_sub_ = rospy.Subscriber('whirlybird', Whirlybird, self.whirlybird_callback, queue_size=5)
        self.psi_r_sub_ = rospy.Subscriber('psi_r', Float32, self.psi_r_callback, queue_size=5)
        self.theta_r_sub_ = rospy.Subscriber('theta_r', Float32, self.theta_r_callback, queue_size=5)
        self.command_pub_ = rospy.Publisher('command', Command, queue_size=5)
        while not rospy.is_shutdown():
            # wait for new messages and call the callback when they arrive
            rospy.spin()

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

        theta_error = self.theta_r - theta
        theta_error_dot = self.dirty_derivative(theta_error, self.prev_theta_error, self.prev_theta_error_dirty_dot, dt)
        if np.abs(theta_error_dot) < self.anti_windup_theta:
            self.Int_theta += (dt / 2) * (theta_error + self.prev_theta_error)
        force_tilde = self.P_theta_ * theta_error + self.Int_theta * self.I_theta_ - self.D_theta_ * theta_dot
        force = force_tilde + equilibrium_force
        self.prev_theta_error = theta_error

        psi_error = self.psi_r - psi
        psi_error_dot = self.dirty_derivative(psi_error, self.prev_psi_error, self.prev_psi_error_dirty_dot, dt)
        if np.abs(psi_error_dot) < self.anti_windup_psi:
            self.Int_psi += (dt / 2) * (psi_error + self.prev_psi_error)
        phi_r = psi_error * self.P_psi_ + self.Int_psi * self.I_psi_ - self.D_psi_ * psi_dot
        self.prev_psi_error = psi_error

        phi_error = phi_r - phi
        phi_error_dot = self.dirty_derivative(phi_error, self.prev_phi_error, self.prev_phi_error_dirty_dot, dt)
        if np.abs(phi_error_dot) < self.anti_windup_phi:
            self.Int_phi += (dt / 2) * (phi_error + self.prev_phi_error)
        torque = phi_error * self.P_phi_ + self.Int_phi * self.I_phi_ - self.D_phi_ * phi_dot
        self.prev_phi_error = phi_error

        print('Pitch: {}\nYaw: {}\nRoll: {}'.format(self.Int_theta, self.Int_psi, self.Int_phi))

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

        self.prev_theta_error_dirty_dot = theta_error_dot
        self.prev_phi_error_dirty_dot = phi_error_dot
        self.prev_psi_error_dirty_dot = psi_error_dot

        ##################################

        sat_max = 0.7

        # Scale Output
        l_out = left_force / km
        if l_out < 0:
            l_out = 0
        elif l_out > sat_max:
            rospy.logerr('Left force saturated!')
            l_out = sat_max

        r_out = right_force / km
        if r_out < 0:
            r_out = 0
        elif r_out > sat_max:
            rospy.logerr('Right force saturated!')
            r_out = sat_max

        # Pack up and send command
        command = Command()
        command.left_motor = l_out
        command.right_motor = r_out
        self.command_pub_.publish(command)

    def dirty_derivative(self, this_val, last_val, last_dirty_dot, dt):
        return ((2 * self.sigma - dt) / (2 * self.sigma + dt)) * last_dirty_dot +\
               (2 / (2 * self.sigma + dt)) * (this_val - last_val)


if __name__ == '__main__':
    rospy.init_node('controller', anonymous=True)
    try:
        controller = Controller()
    except:
        raise rospy.ROSInterruptException
    pass
