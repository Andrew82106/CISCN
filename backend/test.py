# -- coding: utf-8 --
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
import io
# 指定要读取的 TensorBoard 日志目录
log_dir = 'G:\CISCN作品赛\log'

# 创建 EventAccumulator 对象来读取 TensorBoard 日志文件
event_acc = EventAccumulator(log_dir)
event_acc.Reload()

# 获取所有的 summary keys
keys = event_acc.Tags()['images']

# 遍历每个 summary，提取图像并保存到文件中
for key in keys:
    events = event_acc.Images(key)
    for event in events:
        image = event.encoded_image_string
        image_stream = io.BytesIO(image)
        # 处理图像数据，例如保存到文件中
        # ...
        image = Image.open(image_stream)
        image.show()
