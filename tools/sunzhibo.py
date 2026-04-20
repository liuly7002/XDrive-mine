import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 列出可用地图
print(client.get_available_maps())

# 载入目标地图，比如 Town03
world = client.load_world('/Game/kuangshan/Maps/kuangshan/kuangshan')

print("Now loaded:", world.get_map().name)
