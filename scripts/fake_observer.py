#!/usr/bin/env python
import rospy
import sys
import tf
import tf_conversions
import numpy as np

from gebsyas.sdf_loader    import world_to_links, SDF, load_xml, res_sdf_path
from kineverse.model.paths import Path
from kineverse.utils       import res_pkg_path, real_quat_from_matrix
from kineverse.bpb_wrapper import pb, create_object, create_cube_shape, create_sphere_shape, create_cylinder_shape, create_compound_shape, load_convex_mesh_shape, matrix_to_transform
from kineverse.visualization.bpb_visualizer import ROSBPBVisualizer

from gazebo_msgs.msg import LinkStates as LinkStatesMsg
from faster_rcnn_object_detector.msg import ObjectInfo as ObjectInfoMsg
from faster_rcnn_object_detector.srv import ImageToObject         as ImageToObjectSrv, \
                                            ImageToObjectResponse as ImageToObjectResponseMsg

phy_objects = {}
inv_phy_obj = {}
world       = None
tf_listener = None
camera_base_tf = np.eye(4)
base_tf     = None
robot_name  = 'fetch'
visualizer  = None

def tf_to_np(transform):
    x  = transform.rotation.x
    y  = transform.rotation.y
    z  = transform.rotation.z
    w  = transform.rotation.w
    x2 = x**2
    y2 = y**2
    z2 = z**2
    w2 = w**2
    return np.array([[w2 + x2 - y2 - z2, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y, transform.origin[0]],
              [2 * x * y + 2 * w * z, w2 - x2 + y2 - z2, 2 * y * z - 2 * w * x, transform.origin[1]],
              [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, w2 - x2 - y2 + z2, transform.origin[2]],
              [0, 0, 0, 1]])

def cb_model_states(states_msg):
    global base_tf, camera_base_tf, robot_name

    for n, p in zip(states_msg.name, states_msg.pose):
        if n in phy_objects:
            phy_objects[n].transform = pb.Transform(pb.Quaternion(p.orientation.x, 
                                                                  p.orientation.y, 
                                                                  p.orientation.z, 
                                                                  p.orientation.w), 
                                                     pb.Vector3(p.position.x,
                                                                p.position.y,
                                                                p.position.z))
        elif n == robot_name:
            base_tf = tf_to_np(pb.Transform(pb.Quaternion(p.orientation.x, 
                                                                  p.orientation.y, 
                                                                  p.orientation.z, 
                                                                  p.orientation.w), 
                                                     pb.Vector3(p.position.x,
                                                                p.position.y,
                                                                p.position.z)))

    world.update_aabbs()
    visualizer.begin_draw_cycle()
    visualizer.draw_world('objects', world)
    try:
        trans, rot = tf_listener.lookupTransform('/base_link', '/head_camera_link', rospy.Time(0))
        #print('{} {} {}\n{} {} {} {}'.format(trans[0], trans[1], trans[2], rot[0], rot[1], rot[2], rot[3]))
        
        camera_base_tf = tf_to_np(pb.Transform(pb.Quaternion(*rot), pb.Vector3(*trans)))
        #fake_observation()
    except (ValueError, tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        pass

    visualizer.render('objects')

width  = 640
height = 480
near   = 0.2
far    = 7.0
fov    = 54.5 * (np.pi / 180)
right  = np.tan(0.5 * fov) * near
top    = right * (height / float(width))
# self._projection_matrix = np.array([[near / r,        0,                            0,                                0],
#                                     [       0, near / t,                            0,                                0],
#                                     [       0,        0, (-far - near) / (far - near), (-2 * far * near) / (far - near)],
#                                     [       0,        0,                           -1,                                0]])
# # Project along x-axis to be aligned with the "x forwards" convention
# self._projection_matrix = np.array([[0, 1, 0, 0], 
#                                     [0, 0, 1, 0], 
#                                     [1, 0, 0, 0], 
#                                     [0, 0, 0, 1]]).dot(self._projection_matrix)
# Primitive projection. Projects y into x, z into y- points in view should range from -0.5 to 0.5

projection_matrix = None 
screen_translation = None
frustum_vertices = None
frustum_lines    = None
aabb_vertex_template = np.array([[1, 1, 1, 1],
                                 [1, 1, 0, 1],
                                 [1, 0, 1, 1],
                                 [1, 0, 0, 1],
                                 [0, 1, 1, 1],
                                 [0, 1, 0, 1],
                                 [0, 0, 1, 1],
                                 [0, 0, 0, 1]]) + np.array([-0.5, -0.5, -0.5, 0])

def generate_matrices():
    global projection_matrix, screen_translation, frustum_vertices, frustum_lines
    projection_matrix = np.array([[0, -1 / right,          0, 0],
                                  [0,            0, -1 / top, 0],
                                  [1 / near,     0,          0, 0]])
    screen_translation = np.array([[0.5 * width,            0, 0.5 * width],
                                   [          0, 0.5 * height, 0.5 * height]])
    frustum_vertices = np.array([[near, -right,  top, 1],
                                 [near, -right, -top, 1],
                                 [near,  right, -top, 1],
                                 [near,  right,  top, 1],
                                 [ far, -(right / near) * far,  (top / near) * far, 1],
                                 [ far, -(right / near) * far, -(top / near) * far, 1],
                                 [ far,  (right / near) * far, -(top / near) * far, 1],
                                 [ far,  (right / near) * far,  (top / near) * far, 1]])
    frustum_lines    = np.array([frustum_vertices[0], frustum_vertices[1], # tl to ll
                             frustum_vertices[0], frustum_vertices[3], # tl to tr
                             frustum_vertices[0], frustum_vertices[4],
                             frustum_vertices[1], frustum_vertices[5], 
                             frustum_vertices[2], frustum_vertices[3],
                             frustum_vertices[2], frustum_vertices[1],
                             frustum_vertices[2], frustum_vertices[6],
                             frustum_vertices[3], frustum_vertices[7],
                             frustum_vertices[4], frustum_vertices[5],
                             frustum_vertices[4], frustum_vertices[7],
                             frustum_vertices[6], frustum_vertices[7],
                             frustum_vertices[6], frustum_vertices[5]
                             ]).T
    frustum_vertices = frustum_vertices.T


def fake_observation():
    visualizer.begin_draw_cycle('frustum', 'aabbs')
    camera_tf = base_tf.dot(camera_base_tf)
    frustum   = camera_tf.dot(frustum_vertices)
    frust_min = frustum.min(axis=1)
    frust_max = frustum.max(axis=1)
    frust_aabb_center = (frust_min + frust_max) * 0.5 
    visualizer.draw_lines('frustum', pb.Transform.identity(), 0.05, camera_tf.dot(frustum_lines).T.tolist())
    visualizer.draw_mesh('frustum', pb.Transform(frust_aabb_center[0], frust_aabb_center[1], frust_aabb_center[2]), (frust_max - frust_min), 'package://gebsyas/meshes/bounding_box.dae')

    objects = world.overlap_aabb(pb.Vector3(*frust_min[:3]), pb.Vector3(*frust_max[:3]))
    aabbs   = [o.aabb for o in objects]
    centers = [a * 0.5 + b * 0.5 for a, b in aabbs]
    ray_starts = [pb.Vector3(*camera_tf[:3,3])] * len(centers)

    ray_results = world.closest_ray_test_batch(ray_starts, centers)
    visible_set = {r.collision_object for r in ray_results}.intersection(set(objects))
    #print(sorted([inv_phy_obj[o] for o in visible_set]))

    i_camera_tf = np.eye(4)
    i_camera_tf[:3, :3] = camera_tf[:3, :3].T
    i_camera_tf[:3,  3] = -i_camera_tf[:3, :3].dot(camera_tf[:3, 3])

    bounding_boxes = []

    for o in visible_set:
        aabb_min, aabb_max = o.aabb
        aabb_min     = np.array([aabb_min[x] for x in range(3)] + [1])
        aabb_max     = np.array([aabb_max[x] for x in range(3)] + [1])
        aabb_dim     = aabb_max - aabb_min
        aabb_corners = (aabb_vertex_template * aabb_dim + aabb_min + 0.5 * aabb_dim).T
        visualizer.draw_points('aabbs', pb.Transform.identity(), 0.05, aabb_corners.T.tolist())

        corners_in_camera = i_camera_tf.dot(aabb_corners)
        projected_corners = projection_matrix.dot(corners_in_camera)
        projected_corners = np.divide(projected_corners, projected_corners[2,:])
        screen_bb_min = projected_corners.min(axis=1)
        screen_bb_max = projected_corners.max(axis=1)

        if screen_bb_max.min() <= -1 or screen_bb_min.max() > 1:
            print('Bounding box for object {} is off screen.\n  Min: {}\n  Max: {}'.format(inv_phy_obj[o], screen_bb_min, screen_bb_max))
        else:
            pixel_bb_min = screen_translation.dot(screen_bb_min.clip(-1, 1))
            pixel_bb_max = screen_translation.dot(screen_bb_max.clip(-1, 1))
            print('Bounding box for object {}.\n  Min: {}\n  Max: {}'.format(inv_phy_obj[o], pixel_bb_min, pixel_bb_max))
            bounding_boxes.append((i_camera_tf.dot(aabb_min + 0.5 * aabb_dim)[0], inv_phy_obj[o], pixel_bb_min, pixel_bb_max))    

    min_box_width = 5
    visible_boxes = []
    for _, label, bmin, bmax in reversed(sorted(bounding_boxes)):
        new_visible_boxes = [(label, bmin, bmax)]
        for vlabel, vmin, vmax in visible_boxes:
            rmin = vmin - bmin
            rmax = vmax - bmax

            if rmin.min() > 0: # Min corner is outside v-box
                if rmax.max() < 0: # Max corner is outside of v-box
                    print('Box for {} is completely occluded.'.format(vlabel))
                elif rmax.min() > 0: # Max corner is inside v-box -> L-shape
                    top_bar   = (vlabel, np.array([vmin[0], bmax[1]]), vmax)
                    right_bar = (vlabel, np.array([bmax[0], vmin[1]]), np.array([vmax[0], bmax[1]]))
                    new_visible_boxes.extend([top_bar, right_bar])
                else: # I/bar shape
                    if rmax[0] >= 0: # bar shape
                        top_bar   = (vlabel, np.array([vmin[0], bmax[1]]), vmax)
                        new_visible_boxes.append(top_bar)
                    else: # I shape
                        right_bar = (vlabel, np.array([bmax[0], vmin[1]]), vmax)
                        new_visible_boxes.append(right_bar)
            elif rmin.max() < 0: # Min corner is inside v-box
                left_bar = (vlabel, vmin, np.array([bmin[0], vmax[1]]))
                bot_bar  = (vlabel, np.array([bmin[0], vmin[1]]), np.array([vmax[0], bmin[1]]))
                new_visible_boxes.extend([left_bar, bot_bar]) # These always go in
                if rmax.max() < 0: # Max corner is outside v-box -> L-shape
                    pass
                elif rmax.min() > 0: # Max corner is inside v-box -> O-shape
                    right_bar = (vlabel, np.array([bmax[0], bmin[1]]), np.array([vmax[0], bmax[1]]))
                    top_bar   = (vlabel, np.array([bmin[0], bmax[1]]), vmax)
                    new_visible_boxes.extend([right_bar, top_bar])
                else: # C/U shape
                    if rmax[0] > 0: # U shape
                        right_bar = (vlabel, np.array([bmax[0], bmin[1]]), vmax)
                        new_visible_boxes.append(right_bar)
                    else: # C shape
                        top_bar   = (vlabel, np.array([bmin[0], bmax[1]]), vmax)
                        new_visible_boxes.append(top_bar)
            else: # Min corner is partially overlapping v-box
                left_bar = (vlabel, vmin, np.array([bmin[0], vmax[1]]))
                bot_bar  = (vlabel, vmin, np.array([vmax[0], bmin[1]]))
                if rmin[0] < 0: # I/II/angle/cap
                    new_visible_boxes.append(left_bar)
                    if rmax.max() < 0: # Max corner is outside shape -> I shape
                        pass
                    elif rmax.min() > 0: # Max corner is inside shape -> Cap
                        right_bar = (vlabel, np.array([bmax[0], vmin[1]]), vmax)
                        top_bar   = (vlabel, np.array([bmin[0], bmax[1]]), np.array([bmax[0], vmax[1]]))
                        new_visible_boxes.extend([right_bar, top_bar])
                    else: # Max corner is overlapping -> II/Angle
                        if rmax[0] > 0: # II shape
                            right_bar = (vlabel, np.array([bmax[0], vmin[1]]), vmax)
                            new_visible_boxes.append(right_bar)
                        else: # Angle shape
                            top_bar   = (vlabel, np.array([bmin[0], bmax[1]]), vmax)
                            new_visible_boxes.append(top_bar)
                else: # bar/=/angle
                    new_visible_boxes.append(bot_bar)
                    if rmax.max() < 0: # Max corner is outside shape -> bar shape
                        pass
                    elif rmax.min() > 0: # Max corner is inside shape -> reversed C
                        right_bar = (vlabel, np.array([bmax[0], bmin[1]]), vmax)
                        top_bar   = (vlabel, np.array([vmin[0], bmax[1]]), np.array([bmax[0], vmax[1]]))
                        new_visible_boxes.extend([right_bar, top_bar])
                    else: # Max corner is overlapping -> =/Angle
                        if rmax[0] > 0: # Angle shape
                            right_bar = (vlabel, np.array([bmax[0], bmin[1]]), vmax)
                            new_visible_boxes.append(right_bar)
                        else: # = shape
                            top_bar   = (vlabel, np.array([vmin[0], bmax[1]]), vmax)
                            new_visible_boxes.append(top_bar)

        visible_boxes = [(vlabel, vmin, vmax) for vlabel, vmin, vmax in new_visible_boxes if (vmax - vmin).min() >= min_box_width]


    bboxes = {}
    for label, bmin, bmax in visible_boxes:
        if label not in bboxes:
            bboxes[label] = ObjectInfoMsg(label=label)
        msg = bboxes[label]
        msg.bbox_xmin.append(bmin[0])
        msg.bbox_ymin.append(bmin[1]) 
        msg.bbox_xmax.append(bmax[0])
        msg.bbox_ymax.append(bmax[1]) 
        msg.score.append(1.0)

    visualizer.render('frustum', 'aabbs')

    return bboxes.values()


def srv_image_to_object(req):
    res = ImageToObjectResponseMsg()
    res.objects = fake_observation()
    return res


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('SDF file needed.')
        exit(0)

    kin_links = None
    for a in sys.argv[1:]:
        if not ':=' in a:
            world_path = res_pkg_path(a)
            print('Loading world file {}'.format(world_path))
            sdf       = SDF(load_xml(world_path))
            kin_links = world_to_links(sdf.worlds.values()[0], Path('bla'), False)
            break
    if kin_links is None:
        print('No SDF file was given.')
        exit(0)

    rospy.init_node('fake_observation')

    width  = rospy.get_param('~width',  640)
    height = rospy.get_param('~height',  480)
    near   = rospy.get_param('~near',  0.2)
    far    = rospy.get_param('~far',  7.0)
    fov    = rospy.get_param('~fov',  54.5)
    right  = np.tan(0.5 * fov) * near
    top    = right * (height / float(width))
    generate_matrices()

    tf_listener = tf.TransformListener()

    visualizer = ROSBPBVisualizer('/observation_vis', 'map')

    world = pb.KineverseWorld()
    for n, l in kin_links.items():
        if l.collision is not None:
            shape = create_compound_shape()
            for c in l.collision.values():
                if c.type == 'mesh':
                    sub_shape = load_convex_mesh_shape(c.mesh)
                    shape.add_child(matrix_to_transform(c.to_parent), sub_shape)
                elif c.type == 'box':
                    sub_shape = create_cube_shape(c.scale)
                    shape.add_child(matrix_to_transform(c.to_parent), sub_shape)
                elif c.type == 'cylinder':
                    sub_shape = create_cylinder_shape(c.scale[0], c.scale[2])
                    shape.add_child(matrix_to_transform(c.to_parent), sub_shape)
                elif c.type == 'sphere':
                    sub_shape = create_sphere_shape(c.scale[0])
                    shape.add_child(matrix_to_transform(c.to_parent), sub_shape)
                else:
                    raise Exception('Unrecognized geometry type in collision of link {}. Type is "{}"'.format(str(key), c.type))
            obj = create_object(shape)
            phy_objects['::'.join(n[2:])] = obj
            world.add_collision_object(obj)
            obj.transform = matrix_to_transform(l.pose)
    inv_phy_obj = {o: n for n, o in phy_objects.items()}

    print('Loaded objects: {}'.format(phy_objects.keys()))
    sub_states = rospy.Subscriber('/gazebo/link_states', LinkStatesMsg, cb_model_states, queue_size=1)
    srv_server = rospy.Service('/image_to_object', ImageToObjectSrv, srv_image_to_object)

    while not rospy.is_shutdown():
        rospy.sleep(1000)



