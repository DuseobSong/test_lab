import rospy
import os, sys, time
import rospkg
import numpy as np
import json

from std_msgs.msg import UInt8
from sensor_msgs.msg import Range
from test_lab.msg import RangeDetection
from nav_msgs.msg import Odometry
from line_following_ros.msg import SystemStatus

def deg2rad(deg):
    return deg * np.pi / 180.0

def rad2deg(rad):
    return rad * 180.0 / np.pi

# Mode change command bytes (SBC -> Mainboard)
CMD_NO_CHANGE       = 0
CMD_TO_MANUAL       = 10
CMD_TO_FOLLOW       = 11
CMD_TO_POS_MODE     = 12
CMD_TO_REMOTE       = 13
CMD_TO_VEL_MODE     = 14

CMD_CHANGE_MODE = [CMD_TO_MANUAL, CMD_TO_FOLLOW, CMD_TO_POS_MODE, CMD_TO_REMOTE, CMD_TO_VEL_MODE]

# Mode feedback bytes (Mainboard -> SBC)
FEED_MANUAL         = 0
FEED_FOLLOW         = 1
FEED_POS_MODE       = 2
FEED_REMOTE         = 3
FEED_VEL_MODE       = 4

CON_MODE_MANUAL = FEED_MANUAL
CON_MODE_VEL    = FEED_VEL_MODE
CON_MODE_FOLLOW = FEED_FOLLOW
CON_MODE_POS    = FEED_POS_MODE
CON_MODE_REMOTE = FEED_REMOTE
CON_MODE_ERROR  = 9

MODES = [FEED_MANUAL, FEED_FOLLOW, FEED_POS_MODE, FEED_REMOTE, FEED_VEL_MODE]

# Rotation Flag
ROTATION_WAIT = 1
ROTATION_START = 2
ROTATION_ON_GOING = 3
ROTATION_END = 4

ROTATION_DIR_CW = 1
ROTATION_DIR_CCW = 2

ANGULAR_VELEOCITY_CALC_COEFF = 1.5

# camera direction
CAM_DIR_LEFT    = 1
CAM_DIR_RIGHT   = 3

# Control status
STATUS_IDLE = 3
STATUS_RUNNING = 5

# External command : drive mode change
REMOTE_ACCELERATE   = 1
REMOTE_DECELERATE   = 2
REMOTE_TURN_LEFT    = 3
REMOTE_TURN_RIGHT   = 4
REMOTE_STOP         = 9

REMOTE_MODE_CHANGE_MANUAL   = 11
REMOTE_MODE_CHANGE_VEL_CON  = 12
REMOTE_MODE_CHANGE_VEL_POS  = 13
REMOTE_MODE_CHANGE_FOLLOW   = 14

REMOTE_DRIVE_MODE_DISTANCE  = 33
REMOTE_DRIVE_MODE_DETECTION = 35
REMOTE_DRIVE_MODE_MAP       = 37

REMOTE_START_RECORD         = 71
REMOTE_STOP_RECORD          = 75
REMOTE_QUIT_RECORD          = 79
REMOTE_ACTIVATION           = 99

# Sensor detection frequency
DECT_FREQ = 100

# Limitation
V_LIN_LIM = 0.5 # [m/s]
V_ANG_LIM = deg2rad(40)

def load_map_data(dir):
    with open(dir) as f:
        tmp_data = json.loads(f.read())
        
    if 'WAYPOINTS' in tmp_data.keys():
        return True, tmp_data
    else:
        return False, None
    
def load_topic_list(dir):
    try:
        with open(dir) as f:
            topic_list = json.loads(f.read())
        return True, topic_list
    except:
        print('[ERROR] Topic list not found.')
        return False, None
    

class Test:
    def __init__(self, 
                 error_code_list_dir,
                 topic_list_dir,
                 camera_direction, 
                 drive_mode=REMOTE_DRIVE_MODE_DETECTION, 
                 halt_distance=0.0,
                 map_data=None,
                 dect_freq=DECT_FREQ,
                 v_lin_lim=V_LIN_LIM,
                 v_ang_lim=V_ANG_LIM):
        self.rate = rospy.Rate(100)
        
        self.command_mode = CON_MODE_MANUAL
        self.remote_control = False
        
        self.target_x = 0.0
        self.target_y = 0.0
        
        self.obstacle_detected = False
        self.cws_stop = False
        
        self.cmd_vel_lin = 0.0
        self.cmd_Vel_ang = 0.0
        
        self.rotation_flag = ROTATION_WAIT
        self.rotation_dir = ROTATION_DIR_CW
        self.rotation_margin = deg2rad(10)
        self.rotation_angular_velosity = deg2rad(20)
        self.accumulated_angle = 0.0
        self.rotation_chk_time = 0.0
        self.rotation_end_chk_cnt = 0
        
        self.target_orient = 0.0
        self.rotation_start_flag = False
        self.guide_line_missing_flag = False
        
        self.line_tracer_start = False
        self.wait_start_time = 0.0
        
        self.system_monitoring_packet = []
        self.current_mode = CON_MODE_MANUAL
        self.v = 0.0
        self.distance = 0.0
        self.orient = 0.0
        self.gz = 0.0
        self.x = 0.0
        self.y = 0.0
        
        self.error_code_list = load_error_code(error_code_list_dir)
        ret, self.topic_list = load_topic_list(topic_list_dir)
        if not ret:
            exit(3) # Topic list not found
        
        self.command_mode_topic = '/line_tracking/command_mode'
        self.cmd_vel_topic = 'line_tracking/cmd_vel'
        self.obstacle_detection_topic = '/line_tracking/obstacle'
        
        self.drive_mode = drive_mode
        self.halt_distance = halt_distance
        
        self.radiation_type_left = None
        self.radiation_type_right = None
        
        self.range_left = 0.0
        self.range_right = 0.0
        
        # Velocity
        self.v_linear = 0.0
        
        # Position
        self.prev_x = 0.0
        self.prev_y = 0.0
        self.prev_orient = [0.0, 0.0, 0.0, 0.0]
        self.cur_x = 0.0
        self.cur_y = 0.0
        self.cur_orient = [0.0, 0.0, 0.0, 0.0]
        self.tmp_distance = 0.0
        
        self.v_lin_lim = v_lin_lim
        self.v_ang_ccw_lim = -v_ang_lim
        self.v_ang_cw_lim = v_ang_lim
        self.dect_freq = dect_freq
        
        self.emergency_stop = False
        self.running_status = STATUS_IDLE
        self.algo_status = STATUS_IDLE
        self.pause = False
        
        self.camera_direction = camera_direction
        
        self.object_detected_left = False
        self.object_detected_right = False
        
        self.command_mode_topic          = self.topic_list["command_mode_topic"]["NAME"]
        self.cmd_vel_topic               = self.topic_list["cmd_vel_topic"]["NAME"]
        self.obstacle_detection_topic    = self.topic_list["obstacle_detection_topic"]["NAME"]
        self.detected_target_topic       = self.topic_list["detected_target_topic"]["NAME"]
        self.system_status_topic         = self.topic_list["system_status_topic"]["NAME"]
        self.guide_line_missing_topic    = self.topic_list["guide_line_missing_topic"]["NAME"]
        self.rotation_start_topic        = self.topic_list["rotation_start_topic"]["NAME"]
        self.rotation_state_topic        = self.topic_list["rotation_state_topic"]["NAME"]
        self.remote_control_topic        = self.topic_list["remote_control_topic"]["NAME"]
        self.activate_range_sensor_topic = self.topic_list["activate_range_sensor"]["NAME"]
        self.odom_topic                  = self.topic_list["odometry_topic"]["NAME"]
        self.out_topic                   = self.topic_list["range_detection_topic"]["NAME"]
        
        self.map = map_data # waypoint coordinate information and their routes
        
        self.detection_msg = RangeDetection()
        
        self.system_status_sub = rospy.Subscriber(self.system_status_topic, SystemStatus, )
        self.odom_mode_sub     = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)
        self.drive_mode_sub    = rospy.Subscriber(self.remote_control_topic, UInt8, self.remote_callback)
        
        self.detection_sub     = rospy.Publisher(self.out_topic, RangeDetection, self.range_detection_callback)
    
    def remote_callback(self, msg: UInt8):
        cmd = msg.data
        
        if cmd == REMOTE_STOP:
            self.stop()
            self.command_mode = CON_MODE_MANUAL
            
        elif cmd == REMOTE_ACCELERATE:
            if self.remote_control and not self.line_tracer_start:
                if self.current_mode in [CON_MODE_VEL, CON_MODE_POS]:
                    self.accelerate()
        
        elif cmd in [REMOTE_DRIVE_MODE_DETECTION, REMOTE_DRIVE_MODE_DISTANCE, REMOTE_DRIVE_MODE_MAP]:
            self.drive_mode = cmd
            string = ''
            if cmd == REMOTE_DRIVE_MODE_DETECTION:
                string = '1D-LiDAR DETECTION'
            elif cmd == REMOTE_DRIVE_MODE_DISTANCE:
                string = 'DISTANCE - {:.2f} m'.format(self.halt_distance)
            elif cmd == REMOTE_DRIVE_MODE_MAP:
                string = 'MAP'
        print('[DRIVE CONTROL] REFERENCE CHANGED: {}'.format(string))
        
    def odom_callback(self, msg: Odometry):
        self.prev_x = self.cur_x
        self.prev_y = self.cur_y
        self.prev_orient = self.cur_orient
        self.cur_X = msg.pose.pose.position.x
        self.cur_y = msg.pose.pose.position.y
        self.cur_orient = [msg.pose.pose.orientation.x,
                           msg.pose.pose.orientation.y,
                           msg.pose.pose.orientation.z,
                           msg.pose.pose.orientation.w]
        
    def range_detection_callback(self):
        pass
    
    
    
    def cacluate_accumulated_distance(self):
        self.tmp_distance = np.sqrt((self.cur_x - self.prev_x)**2 + (self.cur_y - self.prev_y)**2)
        self.prev_x = self.cur_x
        self.prev_y = self.cur_y
        
    def drive_mode_detection(self):
        if self.emergency_stop:
            return
        elif self.running_status == STATUS_IDLE:
            if self.algo_status != STATUS_IDLE:
                self.tmp_distance = 0.0
                self.algo_status = STATUS_IDLE
                return
        else:
            
            
        pass
    
    def drive_mode_distance(self):
        if self.emergency_stop:
            pass
        pass
    
    def drive_mode_map(self):
        pass

    def run(self):
        
        while True:
            pass
        
    
if __name__ == '__main__':
    rospy.init_node('~', anonymous=True)
    rp = rospkg.RosPack()
    pkg_dir = rp.get_path('test_lab')
    root_pkg_dir = rp.get_path('line_following_ros')
    
    error_code_list_dir = root_pkg_dir + '/config/error_code_list.json'
    topic_list_dir = root_pkg_dir + '/config/topic_list.json'
    
    odom_topic = ''
    left_sensor_topic = ''
    right_sensor_topic = ''
    out_topic = ''
    camera_direction = CAM_DIR_LEFT,
    drive_mode = REMOTE_DRIVE_MODE_DETECTION,
    halt_distance = 0.0
    map_data = None
    dect_freq = DECT_FREQ
    v_lin_lim = V_LIN_LIM
    v_ang_lim = V_ANG_LIM
    
    if rospy.has_param('odom_topic'):
        odom_topic = rospy.get_param('odom_topic')
    
    if rospy.has_param('left_sensor_topic'):
        left_sensor_topic = rospy.get_param('left_sensor_topic')
    
    if rospy.has_param('right_sensor_topic'):
        right_sensor_topic = rospy.get_param('right_sensor_topic')
        
    if rospy.has_param('out_topic'):
        out_topic = rospy.get_param('out_topic')
    
    if rospy.has_param('camera_direction'):
        tmp_string = rospy.get_param('camera_direction')
        if tmp_string == 'left':
            camera_direction = CAM_DIR_LEFT
        elif tmp_string == 'right':
            camera_direction = CAM_DIR_RIGHT
        else:
            print('[DRIVE_CONTROL] ERROR: INVALID CAMERA DIRECTION: {}'.format(tmp_string))
    
    if rospy.has_param('drive_mode'):
        tmp_string = rospy.get_param('drive_mode')
        if tmp_string == 'detection':
            drive_mode = REMOTE_DRIVE_MODE_DETECTION
        elif tmp_string == 'distance':
            drive_mode = REMOTE_DRIVE_MODE_DISTANCE
        elif tmp_string == 'map':
            if rospy.has_param('map'):
                ret, map_data = load_map_data(rospy.get_param('map'))

                if ret: 
                    drive_mode = REMOTE_DRIVE_MODE_MAP
                else:
                    drive_mode = REMOTE_DRIVE_MODE_DETECTION
                    print('[DRIVE_CONTROL] ERROR: Unknown mode - {}\r\nDefalut(Detection) mode activated.'.format(tmp_string))
                    
    if rospy.has_param('halt_distance'):
        halt_distance = float(rospy.get_param('halt_distance'))
        
    if rospy.has_param('detection_frequency'):
        dect_freq = float(rospy.get_param('detection_frequency'))
        
    if rospy.has_param('v_lin_lim'):
        v_lin_lim = float(rospy.get_param('v_lin_lim'))
    
    if rospy.has_param('v_ang_lim'):
        v_ang_lim = deg2rad(float(rospy.get_param('v_ang_lim')))
    
    node = Test(
        error_code_list_dir=error_code_list_dir,
        topic_list_dir=topic_list_dir,
        camera_direction=camera_direction,
        drive_mode=drive_mode,
        map_data=map_data,
        halt_distance=halt_distance,
        dect_freq=dect_freq,
        v_lin_lim=v_lin_lim,
        v_ang_lim=v_ang_lim
    )
    
    node.run()