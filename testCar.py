import time
import sys
from pathlib import Path
import math

import carla
import random
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.demo.carla_vitdet_runtime import VitDetCarlaRuntime


def get_camera_intrinsic(image_w: int, image_h: int, fov: float):
    f = image_w / (2.0 * math.tan(fov * math.pi / 360.0))
    return np.array([[f, 0, image_w / 2.0], [0, f, image_h / 2.0], [0, 0, 1]], dtype=np.float32)


def project_world_to_image(points_world, world2cam, K, image_w, image_h):
    pixels = []
    for p in points_world:
        p_h = np.array([p.x, p.y, p.z, 1.0], dtype=np.float32)
        p_cam = world2cam @ p_h
        if p_cam[2] <= 0:
            continue
        p_img = K @ (p_cam[:3] / p_cam[2])
        u, v = p_img[0], p_img[1]
        if 0 <= u < image_w and 0 <= v < image_h:
            pixels.append((u, v))
        else:
            pixels.append((u, v))  # 仍然保留用于外接框
    if len(pixels) == 0:
        return None
    us, vs = zip(*pixels)
    x1, x2 = min(us), max(us)
    y1, y2 = min(vs), max(vs)
    return [int(x1), int(y1), int(x2), int(y2)]


def get_gt_bboxes_for_camera(camera_actor, world, image_w, image_h, fov):
    K = get_camera_intrinsic(image_w, image_h, fov)
    world2cam = np.array(camera_actor.get_transform().get_inverse_matrix(), dtype=np.float32)
    boxes = []
    actors = world.get_actors()
    targets = list(actors.filter('vehicle.*')) + list(actors.filter('walker.*'))
    for actor in targets:
        bb = actor.bounding_box
        verts = bb.get_world_vertices(actor.get_transform())
        box = project_world_to_image(verts, world2cam, K, image_w, image_h)
        if box is None:
            continue
        cls = 1 if 'vehicle' in actor.type_id else 2  # 简单分类示例
        boxes.append({"bbox": box, "class": cls, "id": actor.id})
    return boxes


def draw_gt_boxes(img, boxes, color=(255, 0, 0)):
    if boxes is None:
        return img
    vis = img.copy()
    for b in boxes:
        x1, y1, x2, y2 = b["bbox"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            vis,
            f"GT-{b['class']}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis

client = carla.Client('172.16.1.24', 2000)
client.set_timeout(10.0)
world = client.get_world()

settings = world.get_settings()
settings.synchronous_mode = True  # 开启同步模式
settings.fixed_delta_seconds = 0.05  # 稳定时间步，避免同步卡住
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)  # 与世界同步
traffic_manager.set_random_device_seed(0)  # 可复现或改成时间种子
traffic_manager.set_global_distance_to_leading_vehicle(2.5)
traffic_manager.set_hybrid_physics_mode(True)  # 远处简化物理，减少卡顿
traffic_manager.set_respawn_dormant_vehicles(True)
traffic_manager.set_boundaries_respawn_dormant_vehicles(25, 200)  # 允许卡住车辆重生

blueprint_library = world.get_blueprint_library()
vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
spawn_points_all = world.get_map().get_spawn_points()
random.shuffle(spawn_points_all)
vehicle = None
for sp in spawn_points_all:
    vehicle = world.try_spawn_actor(vehicle_bp, sp)
    if vehicle is not None:
        break
if vehicle is None:
    raise RuntimeError("没有可用的ego车spawn点，请重启或清理场景后重试")

# 摄像头参数
camera_bp = blueprint_library.find('sensor.camera.rgb')
camera_bp.set_attribute('image_size_x', '1024')
camera_bp.set_attribute('image_size_y', '1024')
camera_bp.set_attribute('fov', '90')
cam_w, cam_h, cam_fov = 1024, 1024, 90.0

camera_transforms = {
    'front': carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)),
    'left': carla.Transform(carla.Location(x=1.0, y=-0.8, z=2.2), carla.Rotation(yaw=-60)),
    'right': carla.Transform(carla.Location(x=1.0, y=0.8, z=2.2), carla.Rotation(yaw=60)),
    'rear': carla.Transform(carla.Location(x=-1.0, z=1.4), carla.Rotation(yaw=180))
}

# 用于保存图像数据
image_buffers, prev_frames = {}, {}

def make_callback(name):
    def callback(image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = np.reshape(array, (image.height, image.width, 4))[:, :, :3]
        image_buffers[name] = array
    return callback

# 生成摄像头
cameras = {}
for name, transform in camera_transforms.items():
    cam = world.spawn_actor(camera_bp, transform, attach_to=vehicle)
    cam.listen(make_callback(name))
    cameras[name] = cam

# 启动自动驾驶
vehicle.set_autopilot(True, traffic_manager.get_port())

vehicle_blueprints = blueprint_library.filter('vehicle.*')
spawn_points = world.get_map().get_spawn_points()

# 防止重复生成在 ego 车位置
spawn_points = [sp for sp in spawn_points if sp.location.distance(vehicle.get_location()) > 8.0]

random.shuffle(spawn_points)
num_npc_vehicles = 80  # 可根据需要设置数量
npc_vehicles = []

for i in range(min(num_npc_vehicles, len(spawn_points))):
    bp = random.choice(vehicle_blueprints)
    transform = spawn_points[i]
    npc = world.try_spawn_actor(bp, transform)
    if npc is not None:
        npc.set_autopilot(True, traffic_manager.get_port())  # 让NPC车自己开
        npc_vehicles.append(npc)

detector = VitDetCarlaRuntime()

try:
    while True:
        world.tick()

        # 收集当前帧（确保每个相机都拿到最新图）
        frames = {}
        for name, img in list(image_buffers.items()):
            frames[name] = img

        # 逐相机检测并收集结果
        annotated_frames = {}
        latency_stats = {"model": [], "tracker": [], "system": []}
        for name, img in frames.items():
            result = detector.infer(name, img)
            if result is None:
                continue
            # 叠加真值框
            gt_boxes = get_gt_bboxes_for_camera(cameras[name], world, cam_w, cam_h, cam_fov)
            annotated = draw_gt_boxes(result["annotated"], gt_boxes)
            annotated_frames[name] = annotated
            latency_stats["model"].append(result.get("model_latency_ms", 0.0))
            latency_stats["tracker"].append(result.get("tracker_latency_ms", 0.0))
            latency_stats["system"].append(result.get("system_latency_ms", 0.0))

        # 组合成 2x2 窗口显示
        order = ["front", "left", "right", "rear"]
        cell_w, cell_h = 512, 512
        blank = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        cells = []
        for name in order:
            frame = annotated_frames.get(name)
            if frame is None:
                cell = blank.copy()
            else:
                cell = cv2.resize(frame, (cell_w, cell_h))
            cv2.putText(cell, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
            cells.append(cell)
        top = np.concatenate(cells[0:2], axis=1)
        bottom = np.concatenate(cells[2:4], axis=1)
        grid = np.concatenate([top, bottom], axis=0)

        def _avg(xs):
            return sum(xs) / len(xs) if xs else 0.0
        stats_text = [
            f"model(ms): { _avg(latency_stats['model']):.1f}",
            f"tracker(ms): { _avg(latency_stats['tracker']):.1f}",
            f"system(ms): { _avg(latency_stats['system']):.1f}",
        ]
        for i, t in enumerate(stats_text):
            cv2.putText(
                grid,
                t,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("multi_cam (front, left, right, rear)", grid)

        if cv2.waitKey(1) == ord('q'):
            break

finally:
    for cam in cameras.values():
        cam.stop()
        cam.destroy()
    vehicle.destroy()
    for npc in npc_vehicles:
        npc.destroy()
    cv2.destroyAllWindows()