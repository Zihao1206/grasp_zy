## 开发板信息
1. Jetson Orin NX 
2. 内存: 8G
3. 存储: 256GB 
4. 接口:
   1. WIFI * 1
   2. 网口 * 1
   3. USB3.0 * 4
   4. DP * 1
   5. GPIO * 40
5. 账户:
   1. jet / 空格 （支持sudo）


## 网络信息
1. CTU
   1. 192.168.127.253  255.255.255.0
2. 开发板
   1. 有线: 192.168.127.102   255.255.255.0
   2. 无线: 192.168.2.51（CMCC-92FZ）  255.255.255.0
3. 机械臂
   1. 有线: 192.168.127.101  255.255.255.0


## 启动
```zsh
# 进入目录
cd /home/jet/zoneyung/grasp_static

# 启动conda环境
conda activate zy_torch

# 启动grasp
python grasp_zy_test.py
python ctu_conn.py
python RunCtu.py
```

## 自启动
```bash
# systemd配置文件
cp /home/jet/zoneyung/grasp_static/RunGraspd.service /etc/systemd/system/

# 启用frpcd
systemctl enable RunGraspd.service

# 重载
systemctl daemon-reload

# 启动
systemctl start RunGraspd.service

# 重启
systemctl restart RunGraspd.service

# 停止
systemctl stop RunGraspd.service

# 查看状态
systemctl status RunGraspd.service
```

## 拷贝
```
scp -rC -P 33322 jet@127.0.0.1:/home/jet/zoneyung/grasp_static/ ./
```