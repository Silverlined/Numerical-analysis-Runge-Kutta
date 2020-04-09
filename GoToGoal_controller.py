from controller import Robot, Motor, DistanceSensor, PositionSensor
import numpy as np

"""
Mobile robot model used in the Webots simulator, Robot: E-puck
The model represents a differential-drive mobile robot.

Z-axis in the simulator represents the horizontal axis of the world.
X-axis in the simulator represents the vertical axis of the world.
The error along the z-axis must be calculated flipped (z - desired_z)
The orietation of the robot (angle - phi) is kept in the range [-pi;pi] 

Created on Thur Apr 09 2020

@author: Dimitriy Georgiev

"""

TIME_STEP = 64
MAX_MOTOR_SPEED = 6.28    # max angular velocity of a wheel[rad/s]
WHEEL_RADIUS = 0.0205   #[m]
MAX_LINEAR_SPEED = 0.25  # [m/s]
AXLE_LENGTH = 0.052
OBJ_THRESHOLD = 78

robot = None
leftMotor = None
rightMotor = None
leftSpeed = 0
rightSpeed = 0

left_encoder = None
right_encoder = None
encoderValues = []

distance_left = 0
distance_right = 0
delta_phi = 0
delta_phi_old = 0
distance_center_old = 0

#Initial pose in [m] and [rad]:
x = 0
z = 0
phi = 1.57

x_kinematic = 0
z_kinematic = 0
phi_kinematic = 1.57

# Initial robot velocity:
x_velocity = 0.0
z_velocity = 0.0
phi_velocity = 0.0
u_ref = 0.5 * MAX_LINEAR_SPEED

# Controller gains:
kp = 1
ki = 0.06 * TIME_STEP / 1000
kd = 0 #0.001 * 1000 / TIME_STEP
phi_old_err = 0
phi_err_derivative = 0
phi_err_integral = 0

def initSensors():
    global left_encoder, right_encoder
    left_encoder = robot.getPositionSensor("left wheel sensor")
    right_encoder = robot.getPositionSensor("right wheel sensor")
    left_encoder.enable(TIME_STEP)
    right_encoder.enable(TIME_STEP)

def initMotors():
    global leftMotor, rightMotor, leftSpeed, rightSpeed
    leftMotor = robot.getMotor('left wheel motor')
    rightMotor = robot.getMotor('right wheel motor')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))
    leftSpeed = 0.2 * MAX_MOTOR_SPEED
    rightSpeed = 0.2 * MAX_MOTOR_SPEED
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def main():
    global robot
    robot = Robot()
    initSensors()      
    initMotors()
    loop()

def get_desired_position(t):
    x_desired = -0.4
    z_desired = 0.1

    _desired = [x_desired, z_desired]
    return _desired

def update_robot_speeds(u_ref, w_ref):
    u = np.sign(u_ref) * min(abs(u_ref), MAX_LINEAR_SPEED)
    w = np.sign(w_ref) * min(abs(w_ref), MAX_MOTOR_SPEED)

    v = np.matrix([[u], [w]])

    A = np.matrix(
        [
            [np.sin(phi),  0],
            [np.cos(phi), 0],
            [0, 1],
        ]
    )

    speeds = [A * v, u, w]
    return speeds
    
def pid_go_to_goal(desired_x, desired_z, kp, ki=0, kd=0):
    global phi_old_err, phi_err_derivative, phi_err_integral, u_ref
    x_err = desired_x - x
    z_err = z - desired_z
    desired_phi = -np.arctan2(x_err, z_err)
    phi_err = desired_phi - phi
    phi_err = np.arctan2(np.sin(phi_err), np.cos(phi_err))
    phi_err_derivative = phi_err - phi_old_err
    phi_err_integral += phi_err 
    
    phi_old_err = phi_err
    if (abs(x_err) < 0.06) and (abs(z_err) < 0.06):
        phi_err = 0
        u_ref = 0

    v_ref = np.matrix([[u_ref], [kp * phi_err + ki * phi_err_integral + kd * phi_err_derivative]])

    return v_ref

def readSensors():
    global encoderValues
    encoderValues.insert(0, left_encoder.getValue())
    encoderValues.insert(1, right_encoder.getValue())

def computeOdometry():
    global distance_left, distance_right, distance_center_old, delta_phi, delta_phi_old, x, z, phi
    distance_left = encoderValues[0] * WHEEL_RADIUS
    distance_right = encoderValues[1] * WHEEL_RADIUS
    distance_center = (distance_left + distance_right) / 2
    delta_phi = (distance_right - distance_left) / AXLE_LENGTH
    phi += (delta_phi - delta_phi_old)
    # Keep phi within [-pi,pi]:
    if phi > np.pi:
        phi = phi - (2 * np.pi)
    elif phi < -np.pi:
        phi = phi + (2 * np.pi)
        
    x += (distance_center_old - distance_center) * np.sin(phi)
    z += (distance_center_old - distance_center) * np.cos(phi)
    distance_center_old = distance_center
    delta_phi_old = delta_phi

    print("new pose (encoders/odometry): %2.2f %2.2f %2.2f\n" %(x, z, phi))
    # print("estimated distance covered by left wheel: %5.2f m" %(distance_left))
    # print("estimated distance covered by right wheel: %5.2f m" %(distance_right))
    # print("estimated change of orientation: %5.2f rad" %(delta_phi))
    
def processBehaviour():
    global x_velocity, z_velocity, phi_velocity, x_kinematic, z_kinematic, phi_kinematic, x, z, phi, rightSpeed, leftSpeed, u_ref
    
    position = get_desired_position(0)
    desire_x = position[0]
    desired_z = position[1]
    # Call the controller to generate reference commands u_ref and w_ref:
    v_ref = pid_go_to_goal(desire_x, desired_z, kp, ki, kd)
    u_ref = v_ref[(0, 0)]
    w_ref = v_ref[(1, 0)]

    # Calculate robot speeds on absolute reference frame using robot model:
    speeds = update_robot_speeds(u_ref, w_ref)
    x_velocity = speeds[0][(0, 0)]
    z_velocity = speeds[0][(1, 0)]
    phi_velocity = speeds[0][(2, 0)]
    u = speeds[1]  # Actual linear speed of the robot after saturation
    w = speeds[2]  # Actual angular speed of the robot after saturation
    rightSpeed = (2 * u + AXLE_LENGTH * w) / (2 * WHEEL_RADIUS)
    leftSpeed = (2 * u - AXLE_LENGTH * w) / (2 * WHEEL_RADIUS)

    # Integrate to update the robot pose - Euler's method:
    x_kinematic -= (x_velocity * TIME_STEP / 2000)
    z_kinematic -= (z_velocity * TIME_STEP / 2000)
    phi_kinematic += (phi_velocity * TIME_STEP / 2000)
    # Keep phi within [-pi,pi]:
    if phi_kinematic > np.pi:
        phi_kinematic -= (2 * np.pi)
    elif phi_kinematic < -np.pi:
        phi_kinematic += (2 * np.pi)
    print("new pose (kinematically/velocity): %2.2f %2.2f %2.2f" %(x_kinematic, z_kinematic, phi_kinematic))

def setActuators():
    global leftSpeed, rightSpeed, leftMotor, rightMotor
    leftSpeed = np.sign(leftSpeed) * min(abs(leftSpeed), MAX_MOTOR_SPEED)
    rightSpeed = np.sign(rightSpeed) * min(abs(rightSpeed), MAX_MOTOR_SPEED)
    leftMotor.setVelocity(leftSpeed)
    rightMotor.setVelocity(rightSpeed)

def loop():
    
    while robot.step(TIME_STEP) != -1:
        readSensors()
        computeOdometry()
        processBehaviour()
        setActuators()

if __name__ == "__main__":
    main()
    