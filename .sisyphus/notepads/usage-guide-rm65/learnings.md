# Learnings - usage-guide-rm65

## 2026-04-20 Initial Analysis
- 项目是一个基于视觉的机械臂抓取系统，部署在 Jetson Orin NX 上
- 核心运行链路：RunGraspd.service → ctu_conn.py → Grasp(hardware=True)
- 参数散落在4个文件中：grasp_zy_zhiyuan1215.py、ctu_conn.py、config.py、camera.py
- config.py 和 grasp_zy_zhiyuan1215.py 中有重复的参数定义（如Tcam2base）
- gripper_zhiyuan.py 的 Modbus 参数换了夹爪型号才需要改
- Metis 和 Momus 子代理在当前环境中不稳定（均 Task aborted）
