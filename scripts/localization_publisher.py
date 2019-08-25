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
base_link        = None
tf_broadcaster   = None
tf_listener      = None


def cb_states(states_msg):
    try:
        idx = states_msg.name.index(robot_name)
        pose = states_msg.pose[idx]

        trans_base_odom, quat_base_odom = tf_listener.lookupTransform('/odom', base_link, rospy.Time(0))
        base_in_odom = frame3_quaternion(*(trans_base_odom + quat_base_odom))
        base_in_map  = frame3_quaternion(pose.position.x, pose.position.y, 0, pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
        map_in_odom = base_in_odom * inverse_frame(base_in_map)
        odom_in_map = inverse_frame(map_in_odom)

        tf_broadcaster.sendTransform(odom_in_map[:3, 3],
                                     real_quat_from_matrix(odom_in_map),
                                     rospy.Time.now(),
                                      'odom',
                                      'map')
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
        try:
            rospy.sleep(1000)
        except rospy.exceptions.ROSInterruptException:
            pass

    sub_state.unregister()

