#!/usr/bin/env python

import rospy
from whirlybird_msgs.msg import Whirlybird
from whirlybird_msgs.msg import Command

import numpy as np


class WhirlybirdSim():

    def __init__(self):
        # initialize member variables
        self.state = np.zeros((6,1)) # [phi theta psi phid thetad psid]' = [q' qd']'
        self.command = np.zeros((2,1)) # [ul ur]'
        self.command_esc = np.zeros((2,1)) # command throttled to ESC rate with zero-order hold

        self.initialized = False

        # get parameters
        try:
            param_namespace = '/whirlybird'
            self.param = rospy.get_param(param_namespace)
        except KeyError:
            rospy.logfatal('Parameters not set in ~/whirlybird namespace')
            rospy.signal_shutdown('Parameters not set')

        # publish/subscribe:
        self.command_sub = rospy.Subscriber('command', Command, self.command_callback)
        self.whirlybird_pub = rospy.Publisher('whirlybird', Whirlybird, queue_size = 1)

        # setup simulation timer
        esc_rate = rospy.get_param('~esc_rate', 50)
        rospy.Timer(rospy.Duration(1.0/esc_rate), self.esc_timer_callback)

        dynamics_rate = rospy.get_param('~rate', 150)
        rospy.Timer(rospy.Duration(1.0/dynamics_rate), self.dynamics_timer_callback)

        l = .9
        h = 1.1

        self.g  = self.param['g']
        self.l1 = np.random.uniform(low=l, high=h)*self.param['l1']
        self.l2 = np.random.uniform(low=l, high=h)*self.param['l2']
        self.m1 = np.random.uniform(low=l, high=h)*self.param['m1']
        self.m2 = np.random.uniform(low=l, high=h)*self.param['m2']
        self.d  = np.random.uniform(low=l, high=h)*self.param['d']
        self.h  = np.random.uniform(low=l, high=h)*self.param['h']
        self.r  = np.random.uniform(low=l, high=h)*self.param['r']
        self.Jx = np.random.uniform(low=l, high=h)*self.param['Jx']
        self.Jy = np.random.uniform(low=l, high=h)*self.param['Jy']
        self.Jz = np.random.uniform(low=l, high=h)*self.param['Jz']
        self.km = self.param['km']

        # spin
        rospy.spin()

    def command_callback(self, msg):
        self.command[0] = msg.left_motor
        self.command[1] = msg.right_motor

    def esc_timer_callback(self, event):
        self.command_esc = self.command

    def dynamics_timer_callback(self, event):
        if not self.initialized:
            self.initialized = True
            return

        # propagate dynamics
        self.propagate((event.current_real - event.last_real).to_sec())

        # sensors
        enc = self.encoders()
        acc, gyro = self.imu()

        # publish
        self.whirlybird_pub.publish(
            roll=enc[0], pitch=enc[1], yaw=enc[2],
            accel_x=acc[0], accel_y=acc[1], accel_z=acc[2],
            gyro_x=gyro[0], gyro_y=gyro[1], gyro_z=gyro[2])

    def propagate(self, dt):
        # RK4 integration
        k1 = self.dynamics(self.state, self.command_esc)
        k2 = self.dynamics(self.state + dt/2*k1, self.command_esc)
        k3 = self.dynamics(self.state + dt/2*k2, self.command_esc)
        k4 = self.dynamics(self.state + dt*k3, self.command_esc)
        self.state += dt/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Implement hard stops on angles
        phi    = self.state[0]
        theta  = self.state[1]
        psi    = self.state[2]
        phid   = self.state[3]
        thetad = self.state[4]
        psid   = self.state[5]

        if phi > self.param['phi_max']:
            phi = self.param['phi_max']
            if phid > 0:
                phid *= -0.5  # bounce against limit
        elif phi < self.param['phi_min']:
            phi = self.param['phi_min']
            if phid < 0:
                phid *= -0.5  # bounce against limit

        if psi > self.param['psi_max']:
            psi = self.param['psi_max']
            if psid > 0:
                psid *= -0.5  # bounce against limit
        elif psi < self.param['psi_min']:
            psi = self.param['psi_min']
            if psid < 0:
                psid *= -0.5  # bounce against limit

        if theta > self.param['theta_max']:
            theta = self.param['theta_max']
            if thetad > 0:
                thetad *= -0.5  # bounce against limit
        elif theta < self.param['theta_min']:
            theta = self.param['theta_min']
            if thetad < 0:
                thetad *= -0.5  # bounce against limit

        # pack back up
        self.state[0] = phi
        self.state[1] = theta
        self.state[2] = psi
        self.state[3] = phid + np.random.normal(0, 0.001, 1)
        self.state[4] = thetad + np.random.normal(0, 0.001, 1)
        self.state[5] = psid

    def dynamics(self, state, command):
        # Get parameters of ros param server
        g  = self.g
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        d  = self.d
        h  = self.h
        r  = self.r
        Jx = self.Jx
        Jy = self.Jy
        Jz = self.Jz
        km = self.km

        phi = state[0]
        theta = state[1]
        psi = state[2]
        phid = state[3]
        thetad = state[4]
        psid = state[5]

        # adjust forces for gains
        fl = km * command[0]
        fr = km * command[1]

        xdot = np.zeros((6,1))

        # angle dynamics
        xdot[0:3] = state[3:6]

        # angle rate dynamics
        sphi   = np.sin(phi)
        cphi   = np.cos(phi)
        stheta = np.sin(theta)
        ctheta = np.cos(theta)
        spsi   = np.sin(psi)
        cpsi   = np.cos(psi)

        ################################################
        # Implement Dynamics for Accelerations Here    #
        M11 = Jx
        M22 = m1 * l1**2 + m2 * l2**2 + Jy * cphi**2 + Jz * sphi**2
        recuring_val = m1 * l1**2 + m2 * l2**2 + Jy * sphi**2 + Jz * cphi**2
        M33 = recuring_val * ctheta**2 + Jx * stheta**2
        M13 = -Jx * stheta
        M23 = (Jy - Jz) * sphi * cphi * ctheta
        M = np.array([
            [M11,   0, M13],
            [0,   M22, M23],
            [M13, M23, M33]
        ])

        c1 = -thetad**2 * (Jz - Jy) * sphi * cphi + \
            psid**2 * (Jz - Jy) * sphi * cphi * ctheta**2 - \
            thetad * psid * ctheta * (Jx - (Jz - Jy) * (cphi**2 - sphi**2))

        c2 = psid**2 * stheta * ctheta * (-Jx + recuring_val) - \
            2 * phid * thetad * (Jz - Jy) * sphi * cphi - \
            phid * psid * ctheta * (-Jx + (Jz - Jy) * (cphi**2 - sphi**2))

        c3 = thetad**2 * (Jz - Jy) * sphi * cphi * stheta - \
            phid * thetad * ctheta * (Jx + (Jz - Jy) * (cphi**2 - sphi**2)) -\
            2 * phid * psid * (Jz - Jy) * ctheta**2 * sphi * cphi + \
            2 * thetad * psid * stheta * ctheta * (Jx - recuring_val)

        c = np.array([
            [c1],
            [c2],
            [c3]
        ])
        c.shape = (3, 1)

        dPdq = np.array([
            [0],
            [(m1 * l1 - m2 * l2) * g * ctheta],
            [0]
        ])
        dPdq.shape = (3, 1)

        Q = np.array([
            [d * (fl - fr)],
            [l1 * (fl + fr) * cphi],
            [l1 * (fl + fr) * ctheta * sphi + d * (fr - fl) * stheta]
        ])
        Q.shape = (3, 1)

        sol = np.squeeze(np.linalg.solve(M, Q - c - dPdq))
        sol.shape = (3, 1)
        xdot[3:6] = sol[0:3]

        # FIXME Constraining yaw and roll for simplified simulation
        # xdot[0] = 0
        # xdot[2] = 0
        # xdot[3] = 0
        # xdot[5] = 0

        ################################################

        return xdot

    def encoders(self):
        return self.state[0:3]

    def imu(self):
        # imu data is not used in ECEn 483
        phi = self.state[0]
        theta = self.state[1]
        psi = self.state[2]

        # accelerometer
        accel = np.zeros((3, 1))

        # rate gyro
        B = np.zeros((3, 3))
        B[0,0] = 1.0
        B[0,2] = -np.sin(theta)
        B[1,1] = np.cos(phi)
        B[1,2] = np.sin(phi)*np.cos(theta)
        B[2,1] = -np.sin(phi)
        B[2,2] = np.cos(phi)*np.cos(theta)

        gyro = B.dot(self.state[3:6])
        return (accel, gyro)

    # def sat(x, max, min):
    #     if x > max:
    #         x = max
    #     elif x < min:
    #         x = min
    #     return x


if __name__ == '__main__':
    rospy.init_node('whirlybird_sim')
    try:
        whirlybird = WhirlybirdSim()
    except:
        rospy.ROSInterruptException()
    pass
