#!/usr/bin/env python
import rospy
import sys
import tf
import tf_conversions
import math

from giskardpy.symengine_wrappers import frame3_quaternion, inverse_frame
from gebsyas.utils import real_quat_from_matrix

from gazebo_msgs.msg import ModelStates as ModelStatesMsg

robot_name = None
base_link      = None
tf_broadcaster = None
tf_listener    = None
map_trans      = None  
map_quat       = None


def cb_states(states_msg):
    global map_trans, map_quat
    try:
        idx = states_msg.name.index(robot_name)
        pose = states_msg.pose[idx]

        map_base_link = inverse_frame(frame3_quaternion(pose.position.x, pose.position.y, 0, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w))

        map_trans = map_base_link[:3, 3]
        map_quat  = real_quat_from_matrix(map_base_link)
    except (ValueError, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        pass

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Needed arguments: <robot name in gazebo> <name of base link in tf>')
        exit(0)

    base_link = sys.argv[2]
    robot_name = sys.argv[1]

    rospy.init_node('gazebo_odom_publisher')

    tf_listener    = tf.TransformListener()
    tf_broadcaster = tf.TransformBroadcaster()

    sub_state = rospy.Subscriber('/gazebo/model_states', ModelStatesMsg, callback=cb_states, queue_size=1)

    while not rospy.is_shutdown():
        if map_trans is not None and map_quat is not None:
            tf_broadcaster.sendTransform(map_trans,
                                         map_quat,
                                         rospy.Time.now(),
                                         'map',
                                         base_link)

    sub_state.unregister()

