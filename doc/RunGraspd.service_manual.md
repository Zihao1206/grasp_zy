# RunGraspd.service 使用手册

> 说明：你提到的 `RunGrasp.service`，在本仓库里的实际文件名是 **`RunGraspd.service`**。

## 1. 文件在哪里

### 仓库内源文件位置

- service 文件：`/home/zh/zh/grasp_zy_py310/RunGraspd.service`
- 启动主程序：`/home/zh/zh/grasp_zy_py310/ctu_conn.py`
- 主抓取逻辑：`/home/zh/zh/grasp_zy_py310/grasp_zy_zhiyuan1215.py`
- 可选启动脚本：`/home/zh/zh/grasp_zy_py310/RunGrasp.sh`

### service 当前默认配置

`RunGraspd.service` 当前写死了以下运行信息：

- 运行用户：`jet`
- 工作目录：`/home/jet/zoneyung/grasp_static`
- Python 解释器：`/home/jet/anaconda3/envs/zy_torch/bin/python`
- 启动命令：`/home/jet/anaconda3/envs/zy_torch/bin/python /home/jet/zoneyung/grasp_static/ctu_conn.py`

也就是说，开发板上的项目目录默认应该是：

`/home/jet/zoneyung/grasp_static`

## 2. 在开发板上怎么运行

以下流程适合 Jetson 开发板直接部署。

### 第一步：把项目放到开发板

推荐把代码放到和 service 一致的目录：

`/home/jet/zoneyung/grasp_static`

如果你已经放在别的目录，也可以，但后面必须同步修改 service 里的路径。

### 第二步：确认 Python 环境存在

service 里默认使用：

`/home/jet/anaconda3/envs/zy_torch/bin/python`

所以开发板上至少要满足下面两点：

1. `jet` 用户存在
2. conda 环境 `zy_torch` 已创建，并且 Python 路径可用

### 第三步：先手动试跑一次

在开发板上先不要急着挂 systemd，先手动验证：

```bash
cd /home/jet/zoneyung/grasp_static
/home/jet/anaconda3/envs/zy_torch/bin/python ctu_conn.py
```

如果手动运行都不通，systemd 方式通常也不会通。

### 第四步：安装 systemd 服务

把仓库里的 service 文件复制到 systemd 目录：

```bash
sudo cp /home/jet/zoneyung/grasp_static/RunGraspd.service /etc/systemd/system/RunGraspd.service
sudo systemctl daemon-reload
sudo systemctl enable RunGraspd.service
sudo systemctl start RunGraspd.service
```

### 第五步：检查服务状态

```bash
sudo systemctl status RunGraspd.service
sudo journalctl -u RunGraspd.service -f
```

## 3. 服务实际会做什么

`RunGraspd.service` 启动后会：

1. 等待 30 秒（`ExecStartPre=/bin/sleep 30`）
2. 进入工作目录 `/home/jet/zoneyung/grasp_static`
3. 用 `zy_torch` 环境里的 Python 启动 `ctu_conn.py`
4. `ctu_conn.py` 会创建 `Grasp(hardware=True)`
5. 程序会连接：
   - CTU：`192.168.127.253:8899`
   - 机械臂：`192.168.127.101:8080`

因此开发板运行前，除了代码和 Python 环境，还要确保：

- 网络通
- 相机、机械臂 SDK 已装好
- CTU 和机械臂 IP 与代码配置一致

## 4. 怎么改路径

如果你的项目不在 `/home/jet/zoneyung/grasp_static`，重点改下面 3 个字段。

### 4.1 修改 service 文件

打开：

```bash
sudo vim /etc/systemd/system/RunGraspd.service
```

假设你把项目改到了：

`/home/jet/project/grasp_static`

那么至少要把下面三项一起改掉：

```ini
[Service]
User=jet
Group=jet
ExecStartPre=/bin/sleep 30
WorkingDirectory=/home/jet/project/grasp_static
Environment="PATH=/home/jet/anaconda3/envs/zy_torch/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/jet/anaconda3/envs/zy_torch/bin/python /home/jet/project/grasp_static/ctu_conn.py
```

改完后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl restart RunGraspd.service
sudo systemctl status RunGraspd.service
```

### 4.2 如果 conda 环境路径也变了

比如 Python 不在：

`/home/jet/anaconda3/envs/zy_torch/bin/python`

而是在：

`/opt/miniconda3/envs/zy_torch/bin/python`

那么要同时改两处：

1. `Environment="PATH=..."`
2. `ExecStart=...python ...`

例如：

```ini
Environment="PATH=/opt/miniconda3/envs/zy_torch/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/opt/miniconda3/envs/zy_torch/bin/python /home/jet/project/grasp_static/ctu_conn.py
```

### 4.3 如果你是改仓库根目录，不只要看 service

当前主运行链路里，下面这些关键文件使用的是**相对路径**，通常只要 `WorkingDirectory` 正确，就能正常找到：

- `grasp_zy_zhiyuan1215.py`
  - `doc/single_new.txt`
  - `dataset/cornell.data`
  - `models/test_250927_1644__zoneyung_/epoch_84_accuracy_1.00`
  - `models/mmdetection/configs/myconfig_zy.py`
  - `models/weights/epoch_20_last.pth`

这意味着：

- **只改项目目录时，优先改 `WorkingDirectory` 和 `ExecStart`**
- 如果 `WorkingDirectory` 指到了新目录，主流程里的相对路径一般会跟着生效

### 4.4 哪些旧文件里还有绝对路径

仓库里仍然有一些旧文件/辅助文件保留了 `/home/jet/zoneyung/grasp_static` 绝对路径，比如：

- `RunGrasp.sh`
- `models/mmdetection/configs/myconfig.py`
- `others/torch2onxx.py`

其中：

- `RunGrasp.sh`：如果你打算用它启动，也要同步修改 `WORK_DIR`
- `models/mmdetection/configs/myconfig.py`：这是旧配置文件，当前主流程实际使用的是 `myconfig_zy.py`
- `others/torch2onxx.py`：更像辅助脚本，不是当前 service 主入口

所以，**开发板正常跑 `RunGraspd.service` 的最小改动集合**通常是：

1. 改 `RunGraspd.service` 的 `WorkingDirectory`
2. 改 `RunGraspd.service` 的 `ExecStart`
3. 如果 conda 位置变化，再改 `Environment PATH`
4. 重新 `daemon-reload + restart`

## 5. 常用命令

### 启动

```bash
sudo systemctl start RunGraspd.service
```

### 停止

```bash
sudo systemctl stop RunGraspd.service
```

### 重启

```bash
sudo systemctl restart RunGraspd.service
```

### 开机自启

```bash
sudo systemctl enable RunGraspd.service
```

### 取消开机自启

```bash
sudo systemctl disable RunGraspd.service
```

### 查看状态

```bash
sudo systemctl status RunGraspd.service
```

### 实时看日志

```bash
sudo journalctl -u RunGraspd.service -f
```

## 6. 常见问题

### 6.1 `systemctl start` 后马上退出

优先检查：

- `ExecStart` 里的 Python 路径是否存在
- `WorkingDirectory` 是否真实存在
- `ctu_conn.py` 能不能手动运行

### 6.2 提示找不到模型、数据文件

优先检查：

- `WorkingDirectory` 是否指向项目根目录
- 项目目录下是否存在 `dataset/`、`models/`、`doc/`

### 6.3 服务启动了，但机械臂/CTU 不工作

优先检查：

- CTU 地址：`192.168.127.253:8899`
- 机械臂地址：`192.168.127.101:8080`
- 开发板网卡 IP 是否与现场网络一致

## 7. 建议的排查顺序

推荐按这个顺序查：

1. 手动执行 `python ctu_conn.py`
2. 检查 `RunGraspd.service` 的路径
3. `sudo systemctl daemon-reload`
4. `sudo systemctl restart RunGraspd.service`
5. `sudo journalctl -u RunGraspd.service -f`

如果你后面要把开发板上的**实际安装文件**也整理出来，优先看：

- 仓库源文件：`RunGraspd.service`
- 开发板安装位置：`/etc/systemd/system/RunGraspd.service`

前者是源码，后者才是 systemd 实际读取的版本。
