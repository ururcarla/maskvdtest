import time
import sys
from pathlib import Path

import carla
import random
import cv2
import numpy as np

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.demo.carla_vitdet_runtime import VitDetCarlaRuntime

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
num_npc_vehicles = 30  # 可根据需要设置数量
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

        # 逐相机检测并显示
        for name, img in frames.items():
            result = detector.infer(name, img)
            if result is None:
                continue
            cv2.imshow(f"{name}_det", result["annotated"])

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