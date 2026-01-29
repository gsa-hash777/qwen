import pandas as pd
import os
import requests
from PIL import Image
import time

MAX_DATA_NUMBER = 5000

# 检查目录是否已存在
if not os.path.exists('coco_2014_caption'):
    # 从CSV文件读取数据
    df = pd.read_csv('coco_data.csv')  # 请将文件名改为您的CSV文件名
    
    # 设置处理的图片数量上限
    total = min(MAX_DATA_NUMBER, len(df))
    
    # 创建保存图片的目录
    os.makedirs('coco_2014_caption', exist_ok=True)
    
    # 初始化存储图片路径和描述的列表
    image_paths = []
    captions = []
    
    for i in range(total):
        try:
            # 获取每行的信息
            row = df.iloc[i]
            image_id = str(row['image_id'])
            caption = str(row['caption'])
            image_url = str(row['image'])
            
            # 下载图片
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            
            # 保存图片
            image_path = os.path.abspath(f'coco_2014_caption/{image_id}.jpg')
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # 验证图片是否有效
            try:
                img = Image.open(image_path)
                img.verify()
                
                # 将路径和描述添加到列表中
                image_paths.append(image_path)
                captions.append(caption)
            except:
                # 如果图片无效，删除文件
                if os.path.exists(image_path):
                    os.remove(image_path)
                print(f'图片无效，跳过: {image_id}')
                continue
            
            # 每处理50张图片打印一次进度
            if (i + 1) % 50 == 0:
                print(f'Processing {i+1}/{total} images ({(i+1)/total*100:.1f}%)')
                
            # 避免请求过快
            time.sleep(0.1)
                
        except Exception as e:
            print(f'处理图片 {image_id} 时出错: {e}')
            continue
    
    # 将图片路径和描述保存为CSV文件
    result_df = pd.DataFrame({
        'image_path': image_paths,
        'caption': captions
    })
    
    # 将数据保存为CSV文件
    result_df.to_csv('./coco-2024-dataset.csv', index=False)
    
    print(f'数据处理完成，共处理了{len(image_paths)}张图片')
    if len(image_paths) < total:
        print(f'警告: {total - len(image_paths)}张图片处理失败')

else:
    print('coco_2014_caption目录已存在,跳过数据处理步骤')