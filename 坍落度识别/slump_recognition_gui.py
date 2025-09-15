#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
坍落度识别系统 - 可视化界面
功能：基于深度学习的混凝土坍落度自动识别
支持实时识别、批量处理、结果分析等功能
"""

# 解决OpenMP冲突问题 - 必须在导入其他库之前设置
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk, ImageDraw, ImageFont
import glob
import logging
from datetime import datetime
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import seaborn as sns

# 导入模型
from models import create_model, get_available_models

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SlumpRecognitionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("坍落度识别系统 - 基于深度学习的智能识别")
        self.root.geometry("1400x1000")
        
        # 设置日志
        self.setup_logging()
        
        # 变量初始化
        self.current_model = None
        self.model_config = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 详细的27类坍落度分类标签
        self.class_names = [
            "类别1_120-130", "类别2_131-135", "类别3_136-140", "类别4_150-160",
            "类别5_161-165", "类别6_166-169", "类别7_171-174", "类别8_176-179",
            "类别9_181-184", "类别10_185-190", "类别11_190-195", "类别12_194-200",
            "类别13_200-205", "类别14_205-210", "类别15_210-215", "类别16_215-220",
            "类别17_220-225", "类别18_225-230", "类别19_230-235", "类别20_235-240",
            "类别21_240-250", "类别22_250-260", "类别23_260-265", "类别24_262-270",
            "类别25_271-275", "类别26_276-280", "类别27_281-285"
        ]
        
        # 批量处理相关变量
        self.input_folder = ""
        self.output_folder = ""
        self.is_processing = False
        
        # 批量处理选项
        self.save_images = tk.BooleanVar(value=False)
        self.save_results = tk.BooleanVar(value=True) 
        self.generate_report = tk.BooleanVar(value=True)
        
        # 视频相关变量
        self.current_video = None
        self.recognition_video = None  # 独立的识别视频对象
        self.current_video_path = ""
        self.video_frame_count = 0
        self.current_frame_index = 0
        self.video_fps = 30
        self.is_video_playing = False
        self.video_thread = None
        self.is_recognizing = False
        self.recognition_thread = None
        self.video_lock = threading.Lock()  # 视频访问锁
        
        # 实时识别相关变量
        self.realtime_recognition_enabled = tk.BooleanVar(value=False)
        self.recognition_interval = tk.IntVar(value=10)  # 每N帧识别一次
        self.last_recognition_frame = -1
        self.recognition_error_count = 0
        self.max_recognition_errors = 5  # 最大连续错误次数
        
        # 识别结果
        self.recognition_results = []
        self.batch_results = {}
        
        # 显示相关
        self.canvas_width = 400
        self.canvas_height = 300
        
        # 模型设置
        self.model_name = tk.StringVar(value="resnet18")
        self.confidence_threshold = tk.DoubleVar(value=0.7)
        self.batch_size = tk.IntVar(value=32)
        self.image_size = tk.IntVar(value=224)
        
        # 预处理设置 - 使用与训练时一致的[-1,1]归一化
        self.auto_enhance = tk.BooleanVar(value=True)
        self.normalize_enabled = tk.BooleanVar(value=True)  # 启用[-1,1]归一化
        self.resize_enabled = tk.BooleanVar(value=True)
        
        self.setup_ui()
        self.load_config()
        self.init_ui_state()
        
        # 绑定窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
    def setup_logging(self):
        """设置日志系统"""
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        log_filename = f"logs/slump_recognition_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("坍落度识别系统启动")
    
    def init_ui_state(self):
        """初始化UI状态"""
        # 禁用需要模型和视频的控件
        self.recognize_btn.config(state=tk.DISABLED)
        self.batch_start_btn.config(state=tk.DISABLED)
        self.play_btn.config(state=tk.DISABLED)
        self.realtime_check.config(state=tk.DISABLED)
        
        # 设置初始状态文本
        self.realtime_status_var.set("实时识别：等待模型和视频加载")
        self.model_status_var.set("未加载模型")
        self.batch_status_var.set("准备就绪")
        
        self.logger.info("UI状态初始化完成")
        
    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 标题
        title_label = ttk.Label(main_frame, text="坍落度识别系统 - 智能混凝土质量检测", 
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 15))
        
        # 创建Notebook多标签页
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 模型配置标签页
        self.model_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.model_frame, text="🤖 模型配置")
        
        # 实时识别标签页
        self.realtime_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.realtime_frame, text="🎯 实时识别")
        
        # 批量处理标签页
        self.batch_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.batch_frame, text="📁 批量处理")
        
        # 结果分析标签页
        self.analysis_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_frame, text="📊 结果分析")
        
        # 设置各个标签页
        self.setup_model_config()
        self.setup_realtime_recognition()
        self.setup_batch_processing()
        self.setup_result_analysis()
        
    def setup_model_config(self):
        """设置模型配置界面"""
        # 模型选择区域
        model_selection_frame = ttk.LabelFrame(self.model_frame, text="模型选择与配置", padding="10")
        model_selection_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 模型类型选择
        ttk.Label(model_selection_frame, text="模型类型:").grid(row=0, column=0, sticky=tk.W, pady=5)
        model_combo = ttk.Combobox(model_selection_frame, textvariable=self.model_name, 
                                  values=['resnet18', 'resnet50', 'vgg16'], state="readonly", width=20)
        model_combo.grid(row=0, column=1, padx=(10, 20), pady=5)
        
        # 模型权重路径
        ttk.Label(model_selection_frame, text="模型权重:").grid(row=0, column=2, sticky=tk.W, pady=5)
        self.weight_path_var = tk.StringVar(value="pretrained/resnet18/best_model.pth")
        ttk.Entry(model_selection_frame, textvariable=self.weight_path_var, width=30).grid(row=0, column=3, padx=(10, 5), pady=5)
        ttk.Button(model_selection_frame, text="浏览", command=self.select_model_weights).grid(row=0, column=4, pady=5)
        
        # 设备选择
        ttk.Label(model_selection_frame, text="计算设备:").grid(row=1, column=0, sticky=tk.W, pady=5)
        device_label = ttk.Label(model_selection_frame, text=f"当前: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        device_label.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # 分类设置
        ttk.Label(model_selection_frame, text="分类标签:").grid(row=1, column=2, sticky=tk.W, pady=5)
        self.class_names_var = tk.StringVar(value=", ".join(self.class_names))
        ttk.Entry(model_selection_frame, textvariable=self.class_names_var, width=30).grid(row=1, column=3, padx=(10, 5), pady=5)
        
        # 模型参数设置
        param_frame = ttk.LabelFrame(self.model_frame, text="推理参数设置", padding="10")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 置信度阈值
        ttk.Label(param_frame, text="置信度阈值:").grid(row=0, column=0, sticky=tk.W, pady=5)
        confidence_scale = ttk.Scale(param_frame, from_=0.1, to=1.0, variable=self.confidence_threshold, 
                                    orient=tk.HORIZONTAL, length=200)
        confidence_scale.grid(row=0, column=1, padx=(10, 5), pady=5)
        ttk.Label(param_frame, textvariable=self.confidence_threshold).grid(row=0, column=2, pady=5)
        
        # 批处理大小
        ttk.Label(param_frame, text="批处理大小:").grid(row=1, column=0, sticky=tk.W, pady=5)
        batch_scale = ttk.Scale(param_frame, from_=1, to=64, variable=self.batch_size, 
                               orient=tk.HORIZONTAL, length=200)
        batch_scale.grid(row=1, column=1, padx=(10, 5), pady=5)
        ttk.Label(param_frame, textvariable=self.batch_size).grid(row=1, column=2, pady=5)
        
        # 图像尺寸
        ttk.Label(param_frame, text="输入图像尺寸:").grid(row=2, column=0, sticky=tk.W, pady=5)
        size_combo = ttk.Combobox(param_frame, textvariable=self.image_size, 
                                 values=[224, 256, 299, 384], state="readonly", width=10)
        size_combo.grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
        
        # 预处理选项
        preprocess_frame = ttk.LabelFrame(self.model_frame, text="图像预处理", padding="10")
        preprocess_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Checkbutton(preprocess_frame, text="自动增强", variable=self.auto_enhance).grid(row=0, column=0, sticky=tk.W, padx=(0, 20), pady=5)
        ttk.Checkbutton(preprocess_frame, text="[-1,1]归一化", variable=self.normalize_enabled).grid(row=0, column=1, sticky=tk.W, padx=(0, 20), pady=5)
        ttk.Checkbutton(preprocess_frame, text="自动调整尺寸", variable=self.resize_enabled).grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # 添加预处理说明
        ttk.Label(preprocess_frame, text="注意：归一化方式已匹配训练时的预处理 (x/127.5-1.0)", 
                 font=("Arial", 8), foreground="blue").grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=(5, 0))
        
        # 模型操作按钮
        button_frame = ttk.Frame(self.model_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.load_model_btn = ttk.Button(button_frame, text="🔄 加载模型", command=self.load_model)
        self.load_model_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.test_model_btn = ttk.Button(button_frame, text="🧪 测试模型", command=self.test_model, state=tk.DISABLED)
        self.test_model_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.save_config_btn = ttk.Button(button_frame, text="💾 保存配置", command=self.save_config)
        self.save_config_btn.pack(side=tk.LEFT)
        
        # 模型状态显示
        status_frame = ttk.LabelFrame(self.model_frame, text="模型状态", padding="10")
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.model_status_var = tk.StringVar(value="未加载模型")
        ttk.Label(status_frame, textvariable=self.model_status_var, font=("Arial", 10)).pack(anchor=tk.W)
        
    def setup_realtime_recognition(self):
        """设置实时识别界面"""
        # 文件选择区域
        file_frame = ttk.LabelFrame(self.realtime_frame, text="视频文件选择", padding="10")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 视频文件选择
        ttk.Label(file_frame, text="选择视频:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.video_path_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.video_path_var, width=60).grid(row=0, column=1, padx=(10, 5), pady=5)
        ttk.Button(file_frame, text="浏览", command=self.select_video_file).grid(row=0, column=2, pady=5)
        
        # 主显示区域
        display_frame = ttk.Frame(self.realtime_frame)
        display_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 左侧：图像显示
        left_frame = ttk.LabelFrame(display_frame, text="图像显示", padding="10")
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        # 图像画布
        self.image_canvas = tk.Canvas(left_frame, width=self.canvas_width, height=self.canvas_height, 
                                     bg='lightgray', relief=tk.SUNKEN, bd=2)
        self.image_canvas.pack(pady=(0, 10))
        
        # 视频控制导航
        nav_frame = ttk.Frame(left_frame)
        nav_frame.pack(fill=tk.X)
        
        # 视频控制按钮
        self.play_btn = ttk.Button(nav_frame, text="▶ 播放", command=self.toggle_video_play, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(nav_frame, text="⏮ 前10帧", command=self.video_prev_10).pack(side=tk.LEFT, padx=(0, 2))
        ttk.Button(nav_frame, text="⏭ 后10帧", command=self.video_next_10).pack(side=tk.LEFT, padx=(2, 5))
        
        self.video_info_var = tk.StringVar(value="未选择视频文件")
        ttk.Label(nav_frame, textvariable=self.video_info_var).pack(side=tk.RIGHT)
        
        # 右侧：识别结果
        right_frame = ttk.LabelFrame(display_frame, text="识别结果", padding="10")
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        
        # 识别控制区域
        recognition_control_frame = ttk.LabelFrame(right_frame, text="识别控制", padding="5")
        recognition_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 手动识别按钮
        self.recognize_btn = ttk.Button(recognition_control_frame, text="🎯 识别当前帧", command=self.recognize_current_image, 
                                       state=tk.DISABLED)
        self.recognize_btn.pack(fill=tk.X, pady=(0, 5))
        
        # 实时识别开关
        self.realtime_check = ttk.Checkbutton(recognition_control_frame, text="🔄 实时识别", 
                                             variable=self.realtime_recognition_enabled,
                                             command=self.on_realtime_recognition_toggle)
        self.realtime_check.pack(anchor=tk.W, pady=(0, 5))
        
        # 识别间隔设置
        interval_frame = ttk.Frame(recognition_control_frame)
        interval_frame.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(interval_frame, text="间隔:").pack(side=tk.LEFT)
        interval_combo = ttk.Combobox(interval_frame, textvariable=self.recognition_interval, 
                                     values=[1, 5, 10, 15, 30], state="readonly", width=8)
        interval_combo.pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(interval_frame, text="帧").pack(side=tk.LEFT, padx=(2, 0))
        
        # 实时识别状态显示
        self.realtime_status_var = tk.StringVar(value="实时识别：未启用")
        status_label = ttk.Label(recognition_control_frame, textvariable=self.realtime_status_var, 
                                font=("Arial", 8), foreground="gray")
        status_label.pack(anchor=tk.W)
        
        # 结果显示
        result_display_frame = ttk.Frame(right_frame)
        result_display_frame.pack(fill=tk.BOTH, expand=True)
        
        # 预测类别
        ttk.Label(result_display_frame, text="预测坍落度:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.prediction_var = tk.StringVar(value="--")
        prediction_label = ttk.Label(result_display_frame, textvariable=self.prediction_var, 
                                    font=("Arial", 14, "bold"), foreground="blue")
        prediction_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 置信度
        ttk.Label(result_display_frame, text="置信度:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.confidence_var = tk.StringVar(value="--")
        confidence_label = ttk.Label(result_display_frame, textvariable=self.confidence_var, 
                                    font=("Arial", 12), foreground="green")
        confidence_label.pack(anchor=tk.W, pady=(0, 10))
        
        # 详细概率
        ttk.Label(result_display_frame, text="各类别概率:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        
        # 概率显示框架 - 使用滚动框
        prob_container = ttk.Frame(result_display_frame)
        prob_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # 创建带滚动条的概率显示区域
        self.prob_canvas = tk.Canvas(prob_container, height=150)
        prob_scrollbar = ttk.Scrollbar(prob_container, orient="vertical", command=self.prob_canvas.yview)
        self.prob_frame = ttk.Frame(self.prob_canvas)
        
        self.prob_frame.bind("<Configure>", lambda e: self.prob_canvas.configure(scrollregion=self.prob_canvas.bbox("all")))
        
        self.prob_canvas.create_window((0, 0), window=self.prob_frame, anchor="nw")
        self.prob_canvas.configure(yscrollcommand=prob_scrollbar.set)
        
        self.prob_canvas.pack(side="left", fill="both", expand=True)
        prob_scrollbar.pack(side="right", fill="y")
        
        # 处理时间
        ttk.Label(result_display_frame, text="处理时间:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(0, 5))
        self.process_time_var = tk.StringVar(value="--")
        ttk.Label(result_display_frame, textvariable=self.process_time_var).pack(anchor=tk.W)
        
    def setup_batch_processing(self):
        """设置批量处理界面"""
        # 批量设置区域
        batch_settings_frame = ttk.LabelFrame(self.batch_frame, text="批量识别设置", padding="10")
        batch_settings_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 输入文件夹
        ttk.Label(batch_settings_frame, text="输入文件夹:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.batch_input_var = tk.StringVar()
        ttk.Entry(batch_settings_frame, textvariable=self.batch_input_var, width=50).grid(row=0, column=1, padx=(10, 5), pady=5)
        ttk.Button(batch_settings_frame, text="浏览", command=self.select_batch_input).grid(row=0, column=2, pady=5)
        
        # 输出文件夹
        ttk.Label(batch_settings_frame, text="输出文件夹:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.batch_output_var = tk.StringVar()
        ttk.Entry(batch_settings_frame, textvariable=self.batch_output_var, width=50).grid(row=1, column=1, padx=(10, 5), pady=5)
        ttk.Button(batch_settings_frame, text="浏览", command=self.select_batch_output).grid(row=1, column=2, pady=5)
        
        # 处理选项
        options_frame = ttk.LabelFrame(self.batch_frame, text="识别选项", padding="10")
        options_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 采样间隔
        ttk.Label(options_frame, text="采样间隔:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sample_interval = tk.IntVar(value=1)
        interval_combo = ttk.Combobox(options_frame, textvariable=self.sample_interval, 
                                     values=[1, 5, 10, 20, 50], state="readonly", width=10)
        interval_combo.grid(row=0, column=1, sticky=tk.W, padx=(10, 20), pady=5)
        ttk.Label(options_frame, text="(每隔N张图片识别一次)").grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # 置信度过滤
        ttk.Label(options_frame, text="最低置信度:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.min_confidence = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(options_frame, from_=0.1, to=1.0, variable=self.min_confidence, 
                              orient=tk.HORIZONTAL, length=150)
        conf_scale.grid(row=1, column=1, padx=(10, 5), pady=5)
        ttk.Label(options_frame, textvariable=self.min_confidence).grid(row=1, column=2, sticky=tk.W, padx=(5, 0), pady=5)
        
        # 保存选项
        save_options_frame = ttk.LabelFrame(options_frame, text="保存选项", padding="5")
        save_options_frame.grid(row=2, column=0, columnspan=3, sticky=tk.W+tk.E, pady=10)
        
        ttk.Checkbutton(save_options_frame, text="保存标注图片", variable=self.save_images).grid(row=0, column=0, sticky=tk.W, padx=10)
        ttk.Checkbutton(save_options_frame, text="保存识别结果", variable=self.save_results).grid(row=0, column=1, sticky=tk.W, padx=10)
        ttk.Checkbutton(save_options_frame, text="生成分析报告", variable=self.generate_report).grid(row=0, column=2, sticky=tk.W, padx=10)
        
        # 控制按钮
        control_frame = ttk.Frame(self.batch_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.batch_start_btn = ttk.Button(control_frame, text="▶ 开始批量处理", command=self.start_batch_processing, 
                                         state=tk.DISABLED)
        self.batch_start_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        self.batch_stop_btn = ttk.Button(control_frame, text="⏹ 停止处理", command=self.stop_batch_processing, 
                                        state=tk.DISABLED)
        self.batch_stop_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # 进度显示
        progress_frame = ttk.LabelFrame(self.batch_frame, text="处理进度", padding="10")
        progress_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.batch_progress_bar = ttk.Progressbar(progress_frame, maximum=100)
        self.batch_progress_bar.pack(fill=tk.X, pady=(0, 5))
        
        self.batch_status_var = tk.StringVar(value="准备就绪")
        ttk.Label(progress_frame, textvariable=self.batch_status_var).pack(anchor=tk.W)
        
        # 结果显示
        result_frame = ttk.LabelFrame(self.batch_frame, text="识别结果", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建表格显示结果
        columns = ('文件名', '预测类别', '置信度', '处理时间')
        self.result_tree = ttk.Treeview(result_frame, columns=columns, show='headings', height=15)
        
        # 设置列标题和宽度
        self.result_tree.heading('文件名', text='文件名')
        self.result_tree.heading('预测类别', text='预测类别')
        self.result_tree.heading('置信度', text='置信度')
        self.result_tree.heading('处理时间', text='处理时间(ms)')
        
        self.result_tree.column('文件名', width=200)
        self.result_tree.column('预测类别', width=150)
        self.result_tree.column('置信度', width=100)
        self.result_tree.column('处理时间', width=100)
        
        # 添加滚动条
        result_scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=result_scrollbar.set)
        
        self.result_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右键菜单
        self.result_menu = tk.Menu(self.result_tree, tearoff=0)
        self.result_menu.add_command(label="导出选中结果", command=self.export_selected_results)
        self.result_menu.add_command(label="导出全部结果", command=self.export_all_results)
        self.result_menu.add_command(label="清除结果", command=self.clear_batch_results)
        
        self.result_tree.bind("<Button-3>", self.show_result_menu)
        
    def setup_result_analysis(self):
        """设置结果分析界面"""
        # 分析控制区域
        analysis_control_frame = ttk.LabelFrame(self.analysis_frame, text="分析控制", padding="10")
        analysis_control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # 数据源选择
        ttk.Label(analysis_control_frame, text="数据源:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.data_source_var = tk.StringVar(value="当前会话")
        data_source_combo = ttk.Combobox(analysis_control_frame, textvariable=self.data_source_var, 
                                        values=["当前会话", "导入结果文件"], state="readonly", width=20)
        data_source_combo.grid(row=0, column=1, padx=(10, 20), pady=5)
        
        ttk.Button(analysis_control_frame, text="导入数据", command=self.import_results).grid(row=0, column=2, pady=5)
        ttk.Button(analysis_control_frame, text="生成报告", command=self.generate_analysis_report).grid(row=0, column=3, padx=(10, 0), pady=5)
        
        # 图表显示区域
        chart_frame = ttk.LabelFrame(self.analysis_frame, text="统计图表", padding="10")
        chart_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(12, 8), dpi=100)
        self.canvas_plot = FigureCanvasTkAgg(self.fig, chart_frame)
        self.canvas_plot.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 初始化空图表
        self.update_analysis_chart()
        
    def load_config(self):
        """加载配置文件"""
        config_path = "config/recognition_config.json"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # 应用配置
                self.model_name.set(config.get('model_name', 'resnet18'))
                self.confidence_threshold.set(config.get('confidence_threshold', 0.7))
                self.batch_size.set(config.get('batch_size', 32))
                self.image_size.set(config.get('image_size', 224))
                self.weight_path_var.set(config.get('weight_path', 'pretrained/resnet18/best_model.pth'))
                
                if 'class_names' in config:
                    self.class_names = config['class_names']
                    self.class_names_var.set(", ".join(self.class_names))
                
                self.logger.info("配置文件加载成功")
                
            except Exception as e:
                self.logger.error(f"配置文件加载失败: {e}")
                messagebox.showerror("错误", f"配置文件加载失败: {e}")
        
    def save_config(self):
        """保存配置文件"""
        if not os.path.exists("config"):
            os.makedirs("config")
            
        config = {
            'model_name': self.model_name.get(),
            'weight_path': self.weight_path_var.get(),
            'confidence_threshold': self.confidence_threshold.get(),
            'batch_size': int(self.batch_size.get()),
            'image_size': int(self.image_size.get()),
            'class_names': [name.strip() for name in self.class_names_var.get().split(',')],
            'auto_enhance': self.auto_enhance.get(),
            'normalize_enabled': self.normalize_enabled.get(),
            'resize_enabled': self.resize_enabled.get()
        }
        
        try:
            with open("config/recognition_config.json", 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            self.logger.info("配置保存成功")
            messagebox.showinfo("成功", "配置保存成功")
            
        except Exception as e:
            self.logger.error(f"配置保存失败: {e}")
            messagebox.showerror("错误", f"配置保存失败: {e}")
        
    def select_model_weights(self):
        """选择模型权重文件"""
        filetypes = [
            ('PyTorch模型', '*.pth'),
            ('所有文件', '*.*')
        ]
        
        weight_path = filedialog.askopenfilename(
            title="选择模型权重文件",
            filetypes=filetypes,
            initialdir="pretrained"
        )
        
        if weight_path:
            self.weight_path_var.set(weight_path)
            self.logger.info(f"选择权重文件: {weight_path}")
        
    def load_model(self):
        """加载模型"""
        try:
            # 更新分类标签
            self.class_names = [name.strip() for name in self.class_names_var.get().split(',')]
            num_classes = len(self.class_names)
            
            # 创建模型配置
            self.model_config = {
                'name': self.model_name.get(),
                'pretrained': False,  # 因为要加载自定义权重
                'num_classes': num_classes,
                'dropout': 0.5
            }
            
            self.model_status_var.set("正在加载模型...")
            self.root.update()
            
            # 创建模型
            self.current_model = create_model(self.model_config)
            
            # 加载权重
            weight_path = self.weight_path_var.get()
            if not os.path.exists(weight_path):
                raise FileNotFoundError(f"权重文件不存在: {weight_path}")
            
            # 加载权重到模型
            checkpoint = torch.load(weight_path, map_location=self.device, weights_only=True)
            
            # 处理不同的权重文件格式
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            self.current_model.load_state_dict(state_dict)
            self.current_model.to(self.device)
            self.current_model.eval()
            
            # 更新UI状态
            self.model_status_var.set(f"模型加载成功 - {self.model_name.get().upper()} ({num_classes}类)")
            self.test_model_btn.config(state=tk.NORMAL)
            self.recognize_btn.config(state=tk.NORMAL)
            self.batch_start_btn.config(state=tk.NORMAL)
            
            # 更新实时识别状态
            if self.current_video:
                self.realtime_check.config(state=tk.NORMAL)
                self.realtime_status_var.set("实时识别：准备就绪")
            else:
                self.realtime_status_var.set("实时识别：等待视频加载")
            
            self.logger.info(f"模型加载成功: {self.model_name.get()}, 权重: {weight_path}")
            messagebox.showinfo("成功", "模型加载成功！")
            
        except Exception as e:
            error_msg = f"模型加载失败: {e}"
            self.model_status_var.set("模型加载失败")
            self.logger.error(error_msg)
            messagebox.showerror("错误", error_msg)
        
    def test_model(self):
        """测试模型"""
        if self.current_model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        try:
            # 创建测试输入
            test_input = torch.randn(1, 3, self.image_size.get(), self.image_size.get()).to(self.device)
            
            start_time = time.time()
            with torch.no_grad():
                output = self.current_model(test_input)
                probabilities = F.softmax(output, dim=1)
            
            inference_time = time.time() - start_time
            
            # 显示测试结果
            test_result = (
                f"模型测试成功！\n"
                f"输入形状: {list(test_input.shape)}\n"
                f"输出形状: {list(output.shape)}\n"
                f"推理时间: {inference_time:.3f}秒\n"
                f"设备: {self.device}\n"
                f"分类数量: {len(self.class_names)}"
            )
            
            messagebox.showinfo("模型测试", test_result)
            self.logger.info(f"模型测试成功，推理时间: {inference_time:.3f}秒")
            
        except Exception as e:
            error_msg = f"模型测试失败: {e}"
            self.logger.error(error_msg)
            messagebox.showerror("错误", error_msg)
    def preprocess_image(self, image_path):
        """预处理图像"""
        try:
            # 使用更安全的方式读取图像，支持中文路径
            image = self.safe_imread(image_path)
            if image is None:
                raise ValueError(f"无法读取图像: {image_path}")
            
            # 转换为RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original_size = image.shape[:2]
            
            # 图像增强（如果启用）
            if self.auto_enhance.get():
                image = self.enhance_image(image)
            
            # 调整尺寸
            if self.resize_enabled.get():
                target_size = self.image_size.get()
                image = cv2.resize(image, (target_size, target_size))
            
            # 转换为PIL Image
            pil_image = Image.fromarray(image)
            
            # 转换为张量
            import torchvision.transforms as transforms
            
            # 先转换为张量
            tensor = transforms.ToTensor()(pil_image)  # [0,1]范围
            
            # 使用与训练时一致的归一化方式：[0,1] → [-1,1]
            if self.normalize_enabled.get():
                # 匹配训练时的预处理：x /= 127.5; x -= 1.0
                # ToTensor已经将[0,255] → [0,1]，所以我们需要[0,1] → [-1,1]
                tensor = tensor * 2.0 - 1.0  # [0,1] → [-1,1]
            
            tensor = tensor.unsqueeze(0)  # 添加batch维度
            
            return tensor, pil_image, original_size
            
        except Exception as e:
            self.logger.error(f"图像预处理失败: {e}")
            raise
    
    def safe_imread(self, image_path):
        """安全读取图像，支持中文路径"""
        try:
            # 方法1：使用cv2.imdecode和numpy来支持中文路径
            import numpy as np
            
            # 标准化路径
            image_path = os.path.normpath(image_path)
            
            # 检查文件是否存在
            if not os.path.exists(image_path):
                self.logger.error(f"文件不存在: {image_path}")
                return None
            
            # 检查文件大小
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                self.logger.error(f"文件大小为0: {image_path}")
                return None
            
            # 使用numpy读取文件内容，然后用cv2解码
            with open(image_path, 'rb') as f:
                img_data = np.frombuffer(f.read(), np.uint8)
            
            # 解码图像
            image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            
            if image is None:
                # 尝试使用PIL作为备选方案
                self.logger.warning(f"OpenCV读取失败，尝试使用PIL: {image_path}")
                return self.safe_imread_pil(image_path)
            
            return image
            
        except Exception as e:
            self.logger.error(f"图像读取异常: {e}, 路径: {image_path}")
            # 尝试使用PIL作为备选方案
            return self.safe_imread_pil(image_path)
    
    def safe_imread_pil(self, image_path):
        """使用PIL作为备选的图像读取方法"""
        try:
            from PIL import Image
            import numpy as np
            
            # 使用PIL读取图像
            pil_image = Image.open(image_path)
            
            # 确保是RGB模式
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # 转换为numpy数组，注意PIL是RGB，OpenCV是BGR
            image_array = np.array(pil_image)
            
            # 转换为BGR格式（OpenCV格式）
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            return image_bgr
            
        except Exception as e:
            self.logger.error(f"PIL图像读取也失败: {e}, 路径: {image_path}")
            return None
        
    def enhance_image(self, image):
        """图像增强"""
        try:
            # 转换为PIL Image进行增强
            pil_img = Image.fromarray(image)
            
            # 对比度增强
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.2)
            
            # 锐化增强
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.1)
            
            # 亮度调整
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(1.05)
            
            return np.array(pil_img)
            
        except Exception as e:
            self.logger.warning(f"图像增强失败，使用原图像: {e}")
            return image
        
    
    def select_video_file(self):
        """选择视频文件"""
        filetypes = [
            ('视频文件', '*.mp4 *.avi *.mov *.mkv *.wmv *.flv'),
            ('MP4', '*.mp4'),
            ('AVI', '*.avi'),
            ('所有文件', '*.*')
        ]
        
        video_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=filetypes
        )
        
        if video_path:
            self.load_video(video_path)
    
    def load_video(self, video_path):
        """加载视频文件"""
        try:
            # 打开视频
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                messagebox.showerror("错误", "无法打开视频文件")
                return
            
            # 重置识别状态
            self.is_recognizing = False
            self.last_recognition_frame = -1
            self.recognition_error_count = 0
            if hasattr(self, 'recognition_thread') and self.recognition_thread and self.recognition_thread.is_alive():
                # 等待识别线程结束
                self.recognition_thread.join(timeout=1.0)
            
            self.current_video = cap
            self.current_video_path = video_path
            self.video_path_var.set(video_path)
            
            # 为实时识别创建独立的视频对象
            try:
                self.recognition_video = cv2.VideoCapture(video_path)
                if not self.recognition_video.isOpened():
                    self.logger.warning("无法为识别创建独立视频对象，将使用共享对象")
                    self.recognition_video = None
            except Exception as e:
                self.logger.warning(f"创建识别视频对象失败: {e}")
                self.recognition_video = None
            
            # 获取视频信息
            self.video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.video_fps = cap.get(cv2.CAP_PROP_FPS)
            self.current_frame_index = 0
            
            # 显示第一帧
            ret, frame = cap.read()
            if ret:
                self.display_video_frame(frame)
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 重置到第一帧
            
            # 更新UI状态
            self.play_btn.config(state=tk.NORMAL)
            self.recognize_btn.config(state=tk.NORMAL)
            self.recognize_btn.config(text="🎯 识别当前帧")
            self.realtime_check.config(state=tk.NORMAL)
            self.video_info_var.set(f"视频: {os.path.basename(video_path)} ({self.video_frame_count}帧, {self.video_fps:.1f}fps)")
            
            # 重置实时识别状态
            if self.current_model:
                self.realtime_status_var.set("实时识别：准备就绪")
            else:
                self.realtime_status_var.set("实时识别：等待模型加载")
            
            self.logger.info(f"加载视频: {video_path}, 帧数: {self.video_frame_count}")
            
        except Exception as e:
            self.logger.error(f"视频加载失败: {e}")
            messagebox.showerror("错误", f"视频加载失败: {e}")
            
            # 清理视频对象
            if hasattr(self, 'current_video') and self.current_video:
                self.current_video.release()
                self.current_video = None
            if hasattr(self, 'recognition_video') and self.recognition_video:
                self.recognition_video.release()
                self.recognition_video = None
    
    def clear_video_state(self):
        """清除视频状态"""
        if self.current_video:
            self.current_video.release()
            self.current_video = None
        
        self.is_video_playing = False
        if self.video_thread and self.video_thread.is_alive():
            self.video_thread = None
        
        self.play_btn.config(state=tk.DISABLED, text="▶ 播放")
        self.current_video_path = ""
        self.video_path_var.set("")
        self.video_info_var.set("未选择视频文件")
    
    def display_video_frame(self, frame):
        """显示视频帧"""
        try:
            # 转换颜色空间
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # 调整图片大小适应画布
            img_width, img_height = pil_image.size
            scale_w = self.canvas_width / img_width
            scale_h = self.canvas_height / img_height
            scale = min(scale_w, scale_h)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可显示的格式
            self.current_image = ImageTk.PhotoImage(display_image)
            
            # 清除画布并显示
            self.image_canvas.delete("all")
            x = (self.canvas_width - new_width) // 2
            y = (self.canvas_height - new_height) // 2
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.current_image)
            
            # 更新帧信息
            self.video_info_var.set(f"视频帧: {self.current_frame_index}/{self.video_frame_count} "
                                   f"({self.current_frame_index/self.video_fps:.1f}s)")
            
        except Exception as e:
            self.logger.error(f"视频帧显示失败: {e}")
    
    def toggle_video_play(self):
        """切换视频播放状态"""
        if not self.current_video:
            return
        
        if self.is_video_playing:
            self.stop_video()
        else:
            self.play_video()
    
    def play_video(self):
        """播放视频"""
        if not self.current_video:
            return
        
        self.is_video_playing = True
        self.play_btn.config(text="⏸ 暂停")
        
        # 在新线程中播放视频
        self.video_thread = threading.Thread(target=self.video_play_worker)
        self.video_thread.daemon = True
        self.video_thread.start()
    
    def stop_video(self):
        """停止视频播放"""
        self.is_video_playing = False
        self.play_btn.config(text="▶ 播放")
        
        # 更新实时识别状态
        if self.realtime_recognition_enabled.get():
            self.realtime_status_var.set("实时识别：视频已暂停")
    
    def video_play_worker(self):
        """视频播放工作线程"""
        if not self.current_video:
            return
        
        frame_delay = 1.0 / self.video_fps  # 每帧间隔
        
        while self.is_video_playing and self.current_video:
            # 使用线程锁保护视频播放读取
            frame = None
            frame_success = False
            
            with self.video_lock:
                try:
                    ret, frame = self.current_video.read()
                    if not ret:
                        # 视频结束，重置到开头
                        self.current_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.current_frame_index = 0
                        self.last_recognition_frame = -1  # 重置识别帧计数
                        continue
                    
                    self.current_frame_index = int(self.current_video.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    frame_success = True
                    
                except Exception as e:
                    self.logger.error(f"视频播放读取帧失败: {e}")
                    continue
            
            if frame_success and frame is not None:
                # 在主线程中更新界面
                self.root.after(0, lambda f=frame: self.display_video_frame(f))
                
                # 检查是否需要实时识别
                if self.should_recognize_current_frame():
                    self.root.after(0, self.trigger_realtime_recognition)
            
            time.sleep(frame_delay)
        
        # 播放结束
        self.root.after(0, self.stop_video)
    
    def should_recognize_current_frame(self):
        """检查当前帧是否需要实时识别"""
        # 检查实时识别是否启用
        if not self.realtime_recognition_enabled.get():
            self.root.after(0, lambda: self.realtime_status_var.set("实时识别：未启用"))
            return False
        
        # 检查模型是否已加载
        if not self.current_model:
            self.root.after(0, lambda: self.realtime_status_var.set("实时识别：等待模型加载"))
            return False
        
        # 检查是否正在手动识别中
        if self.is_recognizing:
            return False
        
        # 检查帧间隔
        interval = self.recognition_interval.get()
        if self.current_frame_index - self.last_recognition_frame >= interval:
            self.root.after(0, lambda: self.realtime_status_var.set(f"实时识别：激活中 (每{interval}帧)"))
            return True
        else:
            remaining = interval - (self.current_frame_index - self.last_recognition_frame)
            self.root.after(0, lambda: self.realtime_status_var.set(f"实时识别：等待中 ({remaining}帧后识别)"))
            return False
    
    def on_realtime_recognition_toggle(self):
        """实时识别开关回调"""
        if self.realtime_recognition_enabled.get():
            if not self.current_model:
                messagebox.showwarning("警告", "请先加载模型后再启用实时识别")
                self.realtime_recognition_enabled.set(False)
                return
            
            if not self.current_video:
                messagebox.showwarning("警告", "请先选择视频文件")
                self.realtime_recognition_enabled.set(False)
                return
                
            self.realtime_status_var.set("实时识别：已启用")
            self.logger.info(f"实时识别已启用，识别间隔: {self.recognition_interval.get()}帧")
        else:
            self.realtime_status_var.set("实时识别：已关闭")
            self.logger.info("实时识别已关闭")
    
    def trigger_realtime_recognition(self):
        """触发实时识别（主线程安全）"""
        try:
            if not self.should_recognize_current_frame():
                return
                
            # 更新最后识别帧
            self.last_recognition_frame = self.current_frame_index
            
            # 在后台线程中执行识别（避免阻塞播放）
            recognition_thread = threading.Thread(
                target=self.realtime_recognition_worker, 
                args=(self.current_frame_index,)
            )
            recognition_thread.daemon = True
            recognition_thread.start()
            
        except Exception as e:
            self.logger.error(f"触发实时识别失败: {e}")
    
    def realtime_recognition_worker(self, frame_index):
        """实时识别工作线程"""
        try:
            start_time = time.time()
            
            # 优先使用独立的识别视频对象
            video_obj = self.recognition_video if self.recognition_video else self.current_video
            
            if not video_obj:
                self.logger.error("没有可用的视频对象进行识别")
                return
            
            # 使用线程锁保护视频读取
            frame = None
            with self.video_lock:
                try:
                    video_obj.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                    ret, frame = video_obj.read()
                    if not ret:
                        self.logger.warning(f"无法读取帧 {frame_index}")
                        return
                except Exception as e:
                    self.logger.error(f"读取视频帧失败: {e}")
                    return
            
            # 将视频帧预处理为模型输入
            tensor = self.preprocess_video_frame(frame)
            tensor = tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.current_model(tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            processing_time = time.time() - start_time
            
            # 准备结果数据
            predicted_label = self.class_names[predicted_class]
            result = {
                'video_path': self.current_video_path,
                'frame_index': frame_index,
                'timestamp_seconds': frame_index / self.video_fps,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'recognition_type': 'realtime'
            }
            
            # 在主线程中更新UI（不阻塞播放）
            self.root.after(0, lambda: self.update_realtime_recognition_results(result))
            
        except Exception as e:
            self.logger.error(f"实时识别失败: {e}")
            self.root.after(0, self.handle_recognition_error)
    
    def update_realtime_recognition_results(self, result):
        """更新实时识别结果显示（主线程安全）"""
        try:
            # 只有当前播放的帧才更新显示（避免延迟显示）
            if abs(result['frame_index'] - self.current_frame_index) <= 5:
                # 更新结果显示
                self.prediction_var.set(result['predicted_label'])
                self.confidence_var.set(f"{result['confidence']:.2%}")
                self.process_time_var.set(f"{result['processing_time']:.3f} 秒")
                
                # 重建概率张量以更新概率显示
                probabilities_tensor = torch.tensor(result['probabilities'])
                self.update_probability_display(probabilities_tensor)
            
            # 保存结果（始终保存）
            self.recognition_results.append(result)
            
            # 重置错误计数
            self.recognition_error_count = 0
            
            # 记录日志
            self.logger.info(f"实时识别: 帧{result['frame_index']} → {result['predicted_label']} ({result['confidence']:.2%})")
            
        except Exception as e:
            self.logger.error(f"更新实时识别结果失败: {e}")
            self.handle_recognition_error()
    
    def handle_recognition_error(self):
        """处理识别错误"""
        self.recognition_error_count += 1
        
        if self.recognition_error_count >= self.max_recognition_errors:
            # 连续错误过多，自动关闭实时识别
            self.realtime_recognition_enabled.set(False)
            self.realtime_status_var.set("实时识别：错误过多已自动关闭")
            self.logger.warning(f"实时识别因连续{self.recognition_error_count}次错误已自动关闭")
            messagebox.showwarning("警告", f"实时识别因连续错误已自动关闭，请检查模型或视频文件")
            self.recognition_error_count = 0
        else:
            self.realtime_status_var.set(f"实时识别：错误 ({self.recognition_error_count}/{self.max_recognition_errors})")
    
    def on_closing(self):
        """程序关闭时的清理工作"""
        try:
            self.logger.info("程序正在关闭，清理资源...")
            
            # 停止视频播放
            self.is_video_playing = False
            
            # 关闭实时识别
            self.realtime_recognition_enabled.set(False)
            
            # 等待线程结束
            if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.is_alive():
                self.video_thread.join(timeout=2.0)
            
            if hasattr(self, 'recognition_thread') and self.recognition_thread and self.recognition_thread.is_alive():
                self.recognition_thread.join(timeout=2.0)
            
            # 释放视频资源
            if hasattr(self, 'current_video') and self.current_video:
                self.current_video.release()
                self.current_video = None
                
            if hasattr(self, 'recognition_video') and self.recognition_video:
                self.recognition_video.release()
                self.recognition_video = None
            
            self.logger.info("资源清理完成")
            
        except Exception as e:
            self.logger.error(f"程序关闭清理失败: {e}")
        finally:
            # 关闭主窗口
            self.root.destroy()
    
    def video_prev_10(self):
        """视频前进10帧"""
        if not self.current_video:
            return
        
        target_frame = max(0, self.current_frame_index - 10)
        self.goto_video_frame(target_frame)
    
    def video_next_10(self):
        """视频后退10帧"""
        if not self.current_video:
            return
        
        target_frame = min(self.video_frame_count - 1, self.current_frame_index + 10)
        self.goto_video_frame(target_frame)
    
    def goto_video_frame(self, frame_index):
        """跳转到指定帧"""
        if not self.current_video:
            return
        
        try:
            # 使用线程锁保护视频跳转
            with self.video_lock:
                self.current_video.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ret, frame = self.current_video.read()
                
                if ret:
                    self.current_frame_index = frame_index
                    self.display_video_frame(frame)
                else:
                    self.logger.warning(f"无法跳转到帧 {frame_index}")
                
        except Exception as e:
            self.logger.error(f"视频帧跳转失败: {e}")
        
    def select_image_folder(self):
        """选择图片文件夹"""
        folder_path = filedialog.askdirectory(title="选择图片文件夹")
        
        if folder_path:
            self.folder_path_var.set(folder_path)
            self.input_folder = folder_path
            
        # 扫描图片文件，过滤掉无法读取的文件
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        all_files = []
        
        for ext in image_extensions:
            all_files.extend(glob.glob(os.path.join(folder_path, ext)))
            all_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        # 验证文件是否可读
        self.image_files = []
        invalid_files = []
        
        for file_path in all_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                # 简单验证：检查文件扩展名和大小
                self.image_files.append(file_path)
            else:
                invalid_files.append(file_path)
        
        if invalid_files:
            self.logger.warning(f"发现 {len(invalid_files)} 个无效文件，已跳过")
        
        self.image_files.sort()
        
        if self.image_files:
            self.current_index = 0
            self.display_current_image()
            self.image_info_var.set(f"文件夹: {len(self.image_files)} 张图片")
            self.logger.info(f"加载文件夹: {folder_path}, 找到 {len(self.image_files)} 张图片")
        else:
            messagebox.showwarning("警告", "该文件夹中没有找到图片文件")
        
    def display_image(self, image_path):
        """显示图片"""
        try:
            # 使用安全的方式读取图片
            image = self.safe_imread(image_path)
            if image is None:
                self.logger.error(f"无法显示图片: {image_path}")
                # 在画布上显示错误信息
                self.image_canvas.delete("all")
                self.image_canvas.create_text(
                    self.canvas_width//2, self.canvas_height//2,
                    text="图片读取失败\n请检查文件路径和完整性",
                    fill="red", font=("Arial", 12), anchor="center"
                )
                return
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            
            # 计算缩放比例
            img_width, img_height = pil_image.size
            scale_w = self.canvas_width / img_width
            scale_h = self.canvas_height / img_height
            scale = min(scale_w, scale_h)
            
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # 调整图片大小
            display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # 转换为Tkinter可显示的格式
            self.current_image = ImageTk.PhotoImage(display_image)
            
            # 清除画布并显示图片
            self.image_canvas.delete("all")
            
            # 计算居中位置
            x = (self.canvas_width - new_width) // 2
            y = (self.canvas_height - new_height) // 2
            
            self.image_canvas.create_image(x, y, anchor=tk.NW, image=self.current_image)
            
        except Exception as e:
            self.logger.error(f"图片显示失败: {e}")
    
    def display_current_image(self):
        """显示当前索引的图片"""
        if self.image_files and 0 <= self.current_index < len(self.image_files):
            image_path = self.image_files[self.current_index]
            self.current_image_path = image_path
            self.display_image(image_path)
            
            # 更新信息显示
            filename = os.path.basename(image_path)
            self.image_info_var.set(f"{self.current_index + 1}/{len(self.image_files)}: {filename}")
        
    def prev_image(self):
        """上一张图片"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.display_current_image()
        
    def next_image(self):
        """下一张图片"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.display_current_image()
        
    def recognize_current_image(self):
        """识别当前视频帧"""
        if self.current_model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        # 检查是否正在识别中
        if self.is_recognizing:
            messagebox.showinfo("提示", "正在识别中，请稍候...")
            return
            
        if self.current_video:
            self.start_frame_recognition()
        else:
            messagebox.showwarning("警告", "请先选择视频文件")
            return
    
    def start_frame_recognition(self):
        """启动帧识别（在后台线程中执行）"""
        if self.is_recognizing:
            return
            
        # 设置识别状态
        self.is_recognizing = True
        self.recognize_btn.config(state=tk.DISABLED)
        self.recognize_btn.config(text="识别中...")
        
        # 获取当前帧数据
        if not self.current_video:
            self.reset_recognition_ui()
            return
            
        # 在后台线程中执行识别
        self.recognition_thread = threading.Thread(target=self.frame_recognition_worker)
        self.recognition_thread.daemon = True
        self.recognition_thread.start()
    
    def frame_recognition_worker(self):
        """帧识别工作线程"""
        try:
            start_time = time.time()
            
            # 使用线程锁保护视频读取
            frame = None
            with self.video_lock:
                try:
                    self.current_video.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_index)
                    ret, frame = self.current_video.read()
                    if not ret:
                        self.root.after(0, lambda: messagebox.showerror("错误", "无法读取视频帧"))
                        return
                except Exception as e:
                    self.root.after(0, lambda: messagebox.showerror("错误", f"读取视频帧失败: {e}"))
                    return
            
            # 将视频帧预处理为模型输入
            tensor = self.preprocess_video_frame(frame)
            tensor = tensor.to(self.device)
            
            # 模型推理
            with torch.no_grad():
                output = self.current_model(tensor)
                probabilities = F.softmax(output, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            processing_time = time.time() - start_time
            
            # 准备结果数据
            predicted_label = self.class_names[predicted_class]
            result = {
                'video_path': self.current_video_path,
                'frame_index': self.current_frame_index,
                'timestamp_seconds': self.current_frame_index / self.video_fps,
                'predicted_class': predicted_class,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist(),
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat(),
                'recognition_type': 'manual'
            }
            
            # 在主线程中更新UI
            self.root.after(0, lambda: self.update_recognition_results(result, processing_time))
            
        except Exception as e:
            self.logger.error(f"视频帧识别失败: {e}")
            self.root.after(0, lambda: messagebox.showerror("错误", f"识别失败: {e}"))
        finally:
            # 重置UI状态
            self.root.after(0, self.reset_recognition_ui)
    
    def update_recognition_results(self, result, processing_time):
        """更新识别结果显示（在主线程中执行）"""
        try:
            # 更新结果显示
            self.prediction_var.set(result['predicted_label'])
            self.confidence_var.set(f"{result['confidence']:.2%}")
            self.process_time_var.set(f"{processing_time:.3f} 秒")
            
            # 重建概率张量以更新概率显示
            probabilities_tensor = torch.tensor(result['probabilities'])
            self.update_probability_display(probabilities_tensor)
            
            # 保存结果
            self.recognition_results.append(result)
            self.logger.info(f"视频帧识别完成: 帧{result['frame_index']} → {result['predicted_label']} ({result['confidence']:.2%})")
            
        except Exception as e:
            self.logger.error(f"更新识别结果失败: {e}")
    
    def reset_recognition_ui(self):
        """重置识别UI状态"""
        self.is_recognizing = False
        self.recognize_btn.config(state=tk.NORMAL)
        self.recognize_btn.config(text="🎯 开始识别")
    
    def preprocess_video_frame(self, frame):
        """预处理视频帧"""
        try:
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 图像增强（如果启用）
            if self.auto_enhance.get():
                frame_rgb = self.enhance_image(frame_rgb)
            
            # 调整尺寸
            if self.resize_enabled.get():
                target_size = self.image_size.get()
                frame_rgb = cv2.resize(frame_rgb, (target_size, target_size))
            
            # 转换为PIL Image
            pil_image = Image.fromarray(frame_rgb)
            
            # 转换为张量
            import torchvision.transforms as transforms
            
            # 先转换为张量
            tensor = transforms.ToTensor()(pil_image)  # [0,1]范围
            
            # 使用与训练时一致的归一化方式：[0,1] → [-1,1]
            if self.normalize_enabled.get():
                tensor = tensor * 2.0 - 1.0  # [0,1] → [-1,1]
            
            tensor = tensor.unsqueeze(0)  # 添加batch维度
            
            return tensor
            
        except Exception as e:
            self.logger.error(f"视频帧预处理失败: {e}")
            raise
    
    def update_probability_display(self, probabilities):
        """更新概率显示"""
        # 清除之前的显示
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
        
        # 显示各类别概率 - 按概率排序显示前10个
        probs = probabilities.cpu().numpy()
        prob_pairs = list(zip(self.class_names, probs))
        prob_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 显示前10个最高概率的类别
        top_n = min(10, len(prob_pairs))
        
        for i, (class_name, prob) in enumerate(prob_pairs[:top_n]):
            # 创建进度条显示概率
            frame = ttk.Frame(self.prob_frame)
            frame.pack(fill=tk.X, pady=1)
            
            # 简化类别名显示（只显示类别号和范围）
            short_name = class_name.replace("类别", "").replace("_", " ")
            label = ttk.Label(frame, text=f"{short_name}:", width=12, font=("Arial", 8))
            label.pack(side=tk.LEFT)
            
            prob_var = tk.DoubleVar(value=prob * 100)
            prob_bar = ttk.Progressbar(frame, variable=prob_var, maximum=100, length=80)
            prob_bar.pack(side=tk.LEFT, padx=(5, 5))
            
            # 根据概率值设置颜色
            if prob > 0.5:
                color = "green"
            elif prob > 0.1:
                color = "orange"
            else:
                color = "gray"
            
            prob_label = ttk.Label(frame, text=f"{prob:.1%}", foreground=color, font=("Arial", 8))
            prob_label.pack(side=tk.RIGHT)
        
        # 如果有更多类别，显示提示
        if len(prob_pairs) > top_n:
            more_frame = ttk.Frame(self.prob_frame)
            more_frame.pack(fill=tk.X, pady=2)
            ttk.Label(more_frame, text=f"... 还有 {len(prob_pairs) - top_n} 个类别", 
                     font=("Arial", 8), foreground="gray").pack()
        
        # 更新滚动区域
        self.prob_canvas.update_idletasks()
        self.prob_canvas.configure(scrollregion=self.prob_canvas.bbox("all"))
        
    def select_batch_input(self):
        """选择批量处理输入文件夹"""
        folder_path = filedialog.askdirectory(title="选择输入文件夹")
        if folder_path:
            self.batch_input_var.set(folder_path)
            self.input_folder = folder_path
            
            # 扫描文件夹中的图片数量
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_count = 0
            
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        image_count += 1
            
            self.batch_status_var.set(f"找到 {image_count} 张图片，准备识别")
            self.logger.info(f"选择输入文件夹: {folder_path}, 找到 {image_count} 张图片")

    def select_batch_output(self):
        """选择批量处理输出文件夹"""
        folder_path = filedialog.askdirectory(title="选择输出文件夹")
        if folder_path:
            self.batch_output_var.set(folder_path)
            self.output_folder = folder_path
            self.logger.info(f"选择输出文件夹: {folder_path}")
        
    def log_message(self, message):
        """在日志框中添加消息"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        
        # 记录到日志文件
        self.logger.info(message)
        
        # 更新状态显示
        self.batch_status_var.set(log_entry)
        self.root.update()
        
    def start_batch_processing(self):
        """开始批量处理"""
        if self.current_model is None:
            messagebox.showwarning("警告", "请先加载模型")
            return
        
        if not self.input_folder:
            messagebox.showwarning("警告", "请选择输入文件夹")
            return
            
        # 设置默认输出文件夹（如果用户未选择）
        if not self.output_folder:
            self.output_folder = os.path.join(self.input_folder, "recognition_output")
            self.batch_output_var.set(self.output_folder)
        
        # 清除之前的结果
        self.clear_batch_results()
        
        # 在新线程中运行批量处理
        self.is_processing = True
        self.batch_start_btn.config(state=tk.DISABLED)
        self.batch_stop_btn.config(state=tk.NORMAL)
        
        self.batch_thread = threading.Thread(target=self.batch_processing_worker)
        self.batch_thread.daemon = True
        self.batch_thread.start()
        
    def batch_processing_worker(self):
        """批量处理工作线程"""
        try:
            # 获取所有图片文件
            image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
            image_files = []
            
            for root, dirs, files in os.walk(self.input_folder):
                for file in files:
                    if file.lower().endswith(image_extensions):
                        file_path = os.path.join(root, file)
                        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                            image_files.append(file_path)
            
            # 按采样间隔筛选文件
            sample_interval = self.sample_interval.get()
            if sample_interval > 1:
                image_files = image_files[::sample_interval]
            
            total_files = len(image_files)
            if total_files == 0:
                self.root.after(0, lambda: messagebox.showwarning("警告", "输入文件夹中没有找到有效的图片文件"))
                return
            
            # 创建输出文件夹
            os.makedirs(self.output_folder, exist_ok=True)
            
            # 批量处理
            batch_results = []
            batch_size = int(self.batch_size.get())
            total_images = len(image_files)
            
            for i in range(0, total_images, batch_size):
                if not self.is_processing:  # 检查是否停止
                    break
                
                batch_files = image_files[i:i + batch_size]
                batch_tensors = []
                batch_info = []
                
                # 预处理批次图像
                for image_path in batch_files:
                    try:
                        tensor, pil_image, original_size = self.preprocess_image(image_path)
                        batch_tensors.append(tensor)
                        batch_info.append({
                            'path': image_path,
                            'size': original_size,
                            'filename': os.path.basename(image_path),
                            'start_time': time.time()
                        })
                    except Exception as e:
                        self.log_message(f"预处理失败: {os.path.basename(image_path)} - {e}")
                        continue
                
                if not batch_tensors:
                    continue
                
                # 批量推理
                try:
                    batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
                    
                    with torch.no_grad():
                        outputs = self.current_model(batch_tensor)
                        probabilities = F.softmax(outputs, dim=1)
                        predicted_classes = torch.argmax(probabilities, dim=1)
                        confidences = torch.max(probabilities, dim=1)[0]
                    
                    # 处理结果
                    for j, info in enumerate(batch_info):
                        pred_class = predicted_classes[j].item()
                        confidence = confidences[j].item()
                        pred_label = self.class_names[pred_class]
                        
                        result = {
                            'image_path': info['path'],
                            'filename': info['filename'],
                            'predicted_class': pred_class,
                            'predicted_label': pred_label,
                            'confidence': confidence,
                            'probabilities': probabilities[j].cpu().numpy().tolist(),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        batch_results.append(result)
                        
                        # 更新进度
                        progress = ((i + j + 1) / total_images) * 100
                        self.root.after(0, lambda p=progress: self.batch_progress_bar.configure(value=p))
                        
                        status_msg = f"处理中... {i + j + 1}/{total_images} - {pred_label} ({confidence:.2%})"
                        self.root.after(0, lambda msg=status_msg: self.batch_status_var.set(msg))
                        
                        self.log_message(f"已处理: {info['filename']} → {pred_label} ({confidence:.2%})")
                        
                        # 计算处理时间
                        processing_time = (time.time() - info['start_time']) * 1000  # 毫秒
                        
                        # 更新GUI结果表格 (需要在主线程中执行)
                        self.root.after(0, lambda r=result, pt=processing_time: self.add_result_to_tree(r, pt))
                        
                        # 保存标注图片（如果启用）
                        if self.save_images.get():
                            self.save_annotated_image(info['path'], pred_label, confidence, self.output_folder)
                    
                except Exception as e:
                    self.log_message(f"批量推理失败: {e}")
                    continue
            
            # 保存结果
            if batch_results and self.save_results.get():
                self.save_batch_results(batch_results, self.output_folder)
            
            # 生成报告
            if batch_results and self.generate_report.get():
                self.generate_batch_report(batch_results, self.output_folder)
            
            self.batch_results.update({datetime.now().isoformat(): batch_results})
            self.log_message(f"批量处理完成！共处理 {len(batch_results)} 张图片")
            
            # 更新分析图表
            self.root.after(0, self.update_analysis_chart)
            
        except Exception as e:
            self.log_message(f"批量处理错误: {e}")
            self.logger.error(f"批量处理错误: {e}")
            
        finally:
            # 重置UI状态
            self.root.after(0, self.reset_batch_ui)
    
    def add_batch_result_to_table(self, filename, predicted_label, confidence, processing_time):
        """添加批量处理结果到表格"""
        self.result_tree.insert('', 'end', values=(
            filename,
            predicted_label,
            f"{confidence:.2%}",
            f"{processing_time:.1f}"
        ))
        
        # 自动滚动到底部
        children = self.result_tree.get_children()
        if children:
            self.result_tree.see(children[-1])
    
    def clear_batch_results(self):
        """清除批量处理结果"""
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        
        self.batch_status_var.set("准备就绪")
        self.batch_progress_bar.configure(value=0)
    
    def show_result_menu(self, event):
        """显示结果右键菜单"""
        try:
            self.result_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.result_menu.grab_release()
    
    def export_selected_results(self):
        """导出选中的结果"""
        selected_items = self.result_tree.selection()
        if not selected_items:
            messagebox.showwarning("警告", "请先选择要导出的结果")
            return
        
        self.export_results(selected_items)
    
    def export_all_results(self):
        """导出全部结果"""
        all_items = self.result_tree.get_children()
        if not all_items:
            messagebox.showwarning("警告", "没有结果可导出")
            return
        
        self.export_results(all_items)
    
    def export_results(self, items):
        """导出结果到CSV文件"""
        try:
            import csv
            
            # 选择保存文件
            file_path = filedialog.asksaveasfilename(
                title="保存识别结果",
                defaultextension=".csv",
                filetypes=[("CSV文件", "*.csv"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
            
            # 收集数据
            data = []
            for item in items:
                values = self.result_tree.item(item, 'values')
                data.append(values)
            
            # 写入CSV
            with open(file_path, 'w', newline='', encoding='utf-8-sig') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['文件名', '预测类别', '置信度', '处理时间(ms)'])
                writer.writerows(data)
            
            messagebox.showinfo("成功", f"结果已导出到: {file_path}")
            self.logger.info(f"批量结果导出: {file_path}")
            
        except Exception as e:
            error_msg = f"导出失败: {e}"
            messagebox.showerror("错误", error_msg)
            self.logger.error(error_msg)
    
    def reset_batch_ui(self):
        """重置批量处理UI状态"""
        self.is_processing = False
        self.batch_start_btn.config(state=tk.NORMAL)
        self.batch_stop_btn.config(state=tk.DISABLED)
    
    def add_result_to_tree(self, result, processing_time):
        """将识别结果添加到GUI表格中"""
        try:
            # 添加到结果表格
            item_id = self.result_tree.insert('', 'end', values=(
                result['filename'],
                result['predicted_label'],
                f"{result['confidence']:.2%}",
                f"{processing_time:.1f}"
            ))
            
            # 滚动到最新结果
            self.result_tree.see(item_id)
            
        except Exception as e:
            self.logger.warning(f"添加结果到表格失败: {e}")
    
    def save_annotated_image(self, image_path, predicted_label, confidence, output_folder):
        """保存标注图片"""
        try:
            # 读取原图
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
            
            # 创建绘图对象
            draw = ImageDraw.Draw(pil_image)
            
            # 设置字体（尝试不同字体）
            try:
                font = ImageFont.truetype("arial.ttf", 40)
            except:
                font = ImageFont.load_default()
            
            # 绘制文本
            text = f"{predicted_label}\n{confidence:.1%}"
            
            # 计算文本位置
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = 10
            y = 10
            
            # 绘制背景矩形
            draw.rectangle([x-5, y-5, x+text_width+10, y+text_height+10], 
                          fill=(0, 0, 0, 128), outline=(255, 255, 255))
            
            # 绘制文本
            draw.text((x, y), text, font=font, fill=(255, 255, 255))
            
            # 保存图片
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_folder, f"{name}_annotated{ext}")
            
            pil_image.save(output_path)
            
        except Exception as e:
            self.logger.warning(f"保存标注图片失败: {e}")
    
    def save_batch_results(self, results, output_folder):
        """保存批量处理结果"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(output_folder, f"recognition_results_{timestamp}.json")
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            self.log_message(f"结果已保存: {results_file}")
            
        except Exception as e:
            self.log_message(f"保存结果失败: {e}")
    
    def generate_batch_report(self, results, output_folder):
        """生成批量处理报告"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(output_folder, f"recognition_report_{timestamp}.txt")
            
            # 统计分析
            total_count = len(results)
            class_counts = {}
            confidence_sum = 0
            
            for result in results:
                label = result['predicted_label']
                confidence = result['confidence']
                
                class_counts[label] = class_counts.get(label, 0) + 1
                confidence_sum += confidence
            
            avg_confidence = confidence_sum / total_count if total_count > 0 else 0
            
            # 生成报告
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("坍落度识别批量处理报告\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总图片数: {total_count}\n")
                f.write(f"平均置信度: {avg_confidence:.2%}\n\n")
                
                f.write("分类统计:\n")
                f.write("-" * 30 + "\n")
                for label, count in sorted(class_counts.items()):
                    percentage = (count / total_count) * 100
                    f.write(f"{label}: {count} 张 ({percentage:.1f}%)\n")
                
                f.write("\n详细结果:\n")
                f.write("-" * 30 + "\n")
                for result in results:
                    f.write(f"{result['filename']}: {result['predicted_label']} ({result['confidence']:.2%})\n")
            
            self.log_message(f"报告已生成: {report_file}")
            
        except Exception as e:
            self.log_message(f"生成报告失败: {e}")
    
    def reset_batch_ui(self):
        """重置批量处理UI状态"""
        self.is_processing = False
        self.batch_start_btn.config(state=tk.NORMAL)
        self.batch_stop_btn.config(state=tk.DISABLED)
        self.batch_status_var.set("处理完成")
        
    def stop_batch_processing(self):
        """停止批量处理"""
        self.is_processing = False
        self.log_message("正在停止批量处理...")
        self.batch_status_var.set("正在停止...")
        
    def import_results(self):
        """导入分析结果"""
        filetypes = [
            ('JSON文件', '*.json'),
            ('所有文件', '*.*')
        ]
        
        file_path = filedialog.askopenfilename(
            title="导入识别结果文件",
            filetypes=filetypes
        )
        
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    results = json.load(f)
                
                # 验证数据格式
                if isinstance(results, list) and results:
                    timestamp = datetime.now().isoformat()
                    self.batch_results[f"imported_{timestamp}"] = results
                    
                    self.update_analysis_chart()
                    messagebox.showinfo("成功", f"成功导入 {len(results)} 条识别结果")
                    self.logger.info(f"导入结果文件: {file_path}")
                else:
                    messagebox.showerror("错误", "文件格式不正确")
                    
            except Exception as e:
                error_msg = f"导入文件失败: {e}"
                messagebox.showerror("错误", error_msg)
                self.logger.error(error_msg)
        
    def generate_analysis_report(self):
        """生成分析报告"""
        if not self.batch_results and not self.recognition_results:
            messagebox.showwarning("警告", "没有可分析的数据")
            return
        
        try:
            # 合并所有结果
            all_results = []
            
            # 添加批量结果
            for batch_results in self.batch_results.values():
                all_results.extend(batch_results)
            
            # 添加单张识别结果
            all_results.extend(self.recognition_results)
            
            if not all_results:
                messagebox.showwarning("警告", "没有可分析的数据")
                return
            
            # 选择保存位置
            file_path = filedialog.asksaveasfilename(
                title="保存分析报告",
                defaultextension=".txt",
                filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")]
            )
            
            if not file_path:
                return
            
            # 生成详细分析报告
            self.create_detailed_report(all_results, file_path)
            messagebox.showinfo("成功", "分析报告已生成")
            
        except Exception as e:
            error_msg = f"生成报告失败: {e}"
            messagebox.showerror("错误", error_msg)
            self.logger.error(error_msg)
    
    def create_detailed_report(self, results, file_path):
        """创建详细分析报告"""
        # 统计分析
        total_count = len(results)
        class_counts = {}
        confidence_values = []
        
        for result in results:
            label = result['predicted_label']
            confidence = result['confidence']
            
            class_counts[label] = class_counts.get(label, 0) + 1
            confidence_values.append(confidence)
        
        # 计算统计指标
        avg_confidence = np.mean(confidence_values) if confidence_values else 0
        min_confidence = min(confidence_values) if confidence_values else 0
        max_confidence = max(confidence_values) if confidence_values else 0
        std_confidence = np.std(confidence_values) if confidence_values else 0
        
        # 置信度分布
        high_conf = sum(1 for c in confidence_values if c >= 0.9)
        medium_conf = sum(1 for c in confidence_values if 0.7 <= c < 0.9)
        low_conf = sum(1 for c in confidence_values if c < 0.7)
        
        # 生成报告
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("坍落度识别系统 - 详细分析报告\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析样本数: {total_count}\n")
            f.write(f"模型类型: {self.model_name.get().upper()}\n")
            f.write(f"分类标签: {', '.join(self.class_names)}\n\n")
            
            # 基本统计
            f.write("📊 基本统计信息\n")
            f.write("-" * 40 + "\n")
            f.write(f"平均置信度: {avg_confidence:.3f} ({avg_confidence:.1%})\n")
            f.write(f"最高置信度: {max_confidence:.3f} ({max_confidence:.1%})\n")
            f.write(f"最低置信度: {min_confidence:.3f} ({min_confidence:.1%})\n")
            f.write(f"置信度标准差: {std_confidence:.3f}\n\n")
            
            # 置信度分布
            f.write("🎯 置信度分布\n")
            f.write("-" * 40 + "\n")
            f.write(f"高置信度 (≥90%): {high_conf} 张 ({high_conf/total_count:.1%})\n")
            f.write(f"中等置信度 (70%-90%): {medium_conf} 张 ({medium_conf/total_count:.1%})\n")
            f.write(f"低置信度 (<70%): {low_conf} 张 ({low_conf/total_count:.1%})\n\n")
            
            # 分类统计
            f.write("📈 分类分布统计\n")
            f.write("-" * 40 + "\n")
            for label in sorted(class_counts.keys()):
                count = class_counts[label]
                percentage = (count / total_count) * 100
                f.write(f"{label:>12}: {count:>4} 张 ({percentage:>5.1f}%)\n")
            
            f.write("\n")
            
            # 质量评估
            f.write("🔍 质量评估\n")
            f.write("-" * 40 + "\n")
            
            if avg_confidence >= 0.9:
                quality = "优秀"
            elif avg_confidence >= 0.8:
                quality = "良好"
            elif avg_confidence >= 0.7:
                quality = "一般"
            else:
                quality = "需要改进"
            
            f.write(f"整体识别质量: {quality}\n")
            
            if std_confidence < 0.1:
                consistency = "非常稳定"
            elif std_confidence < 0.15:
                consistency = "稳定"
            elif std_confidence < 0.2:
                consistency = "较稳定"
            else:
                consistency = "不稳定"
            
            f.write(f"识别一致性: {consistency}\n\n")
            
            # 建议
            f.write("💡 改进建议\n")
            f.write("-" * 40 + "\n")
            
            if low_conf > total_count * 0.1:
                f.write("- 低置信度样本较多，建议检查图片质量或重新训练模型\n")
            
            if std_confidence > 0.2:
                f.write("- 识别结果不够稳定，建议增加训练数据\n")
            
            # 找出最不平衡的分类
            class_ratios = [count/total_count for count in class_counts.values()]
            if max(class_ratios) > 0.6:
                f.write("- 分类分布不平衡，建议平衡各类别的训练数据\n")
            
            if avg_confidence >= 0.9:
                f.write("- 模型表现优秀，可以考虑部署到生产环境\n")
            
            f.write("\n详细识别结果列表:\n")
            f.write("-" * 40 + "\n")
            
            # 按置信度排序显示
            sorted_results = sorted(results, key=lambda x: x['confidence'], reverse=True)
            for i, result in enumerate(sorted_results[:50]):  # 只显示前50个
                filename = os.path.basename(result['image_path']) if 'image_path' in result else result.get('filename', f'Image_{i+1}')
                f.write(f"{filename:>30}: {result['predicted_label']:>10} ({result['confidence']:>6.1%})\n")
            
            if len(sorted_results) > 50:
                f.write(f"\n... 还有 {len(sorted_results) - 50} 条结果未显示\n")
        
    def update_analysis_chart(self):
        """更新分析图表"""
        # 合并所有结果
        all_results = []
        
        # 添加批量结果
        for batch_results in self.batch_results.values():
            all_results.extend(batch_results)
        
        # 添加单张识别结果
        all_results.extend(self.recognition_results)
        
        if not all_results:
            # 显示空数据提示
            self.fig.clear()
            ax = self.fig.add_subplot(111)
            ax.text(0.5, 0.5, '暂无数据\n请先进行坍落度识别', 
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, fontsize=16, color='gray')
            ax.set_title('识别结果统计', fontsize=14, fontweight='bold')
            ax.axis('off')
            self.canvas_plot.draw()
            return
        
        # 统计数据
        class_counts = {}
        confidence_values = []
        
        for result in all_results:
            label = result['predicted_label']
            confidence = result['confidence']
            
            class_counts[label] = class_counts.get(label, 0) + 1
            confidence_values.append(confidence)
        
        # 清除之前的图表
        self.fig.clear()
        
        # 创建2x2子图
        gs = self.fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. 分类分布饼图
        ax1 = self.fig.add_subplot(gs[0, 0])
        if class_counts:
            labels = list(class_counts.keys())
            sizes = list(class_counts.values())
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
            
            wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax1.set_title('分类分布', fontweight='bold')
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        # 2. 分类分布柱状图
        ax2 = self.fig.add_subplot(gs[0, 1])
        if class_counts:
            labels = list(class_counts.keys())
            values = list(class_counts.values())
            
            bars = ax2.bar(labels, values, color=plt.cm.Set2(np.linspace(0, 1, len(labels))))
            ax2.set_title('各类别数量', fontweight='bold')
            ax2.set_ylabel('数量')
            
            # 在柱子上显示数值
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{value}', ha='center', va='bottom', fontweight='bold')
            
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 3. 置信度分布直方图
        ax3 = self.fig.add_subplot(gs[1, 0])
        if confidence_values:
            ax3.hist(confidence_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax3.axvline(np.mean(confidence_values), color='red', linestyle='--', 
                       label=f'平均值: {np.mean(confidence_values):.2f}')
            ax3.set_title('置信度分布', fontweight='bold')
            ax3.set_xlabel('置信度')
            ax3.set_ylabel('频次')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. 置信度等级分布
        ax4 = self.fig.add_subplot(gs[1, 1])
        if confidence_values:
            high_conf = sum(1 for c in confidence_values if c >= 0.9)
            medium_conf = sum(1 for c in confidence_values if 0.7 <= c < 0.9)
            low_conf = sum(1 for c in confidence_values if c < 0.7)
            
            categories = ['高\n(≥90%)', '中\n(70%-90%)', '低\n(<70%)']
            values = [high_conf, medium_conf, low_conf]
            colors = ['green', 'orange', 'red']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.7)
            ax4.set_title('置信度等级分布', fontweight='bold')
            ax4.set_ylabel('数量')
            
            # 在柱子上显示数值和百分比
            total = len(confidence_values)
            for bar, value in zip(bars, values):
                if value > 0:
                    height = bar.get_height()
                    percentage = (value / total) * 100
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                            f'{value}\n({percentage:.1f}%)', 
                            ha='center', va='bottom', fontweight='bold')
        
        # 更新图表
        self.canvas_plot.draw()

def main():
    """主函数"""
    root = tk.Tk()
    app = SlumpRecognitionGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
