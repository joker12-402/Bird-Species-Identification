由于.gitignore始终不能忽略掉data/和results/下的大文件，将代码和数据进行分离

├──Bird_Species_Identification/（当前上传的）
├── data/           # 空壳
├── src/            # 存放所有源代码
├── models/         # 存放训练好的模型权重
├── results/        # 空壳
└── docs/           # 存放论文、参考资料

├──results（和Bird_Species_Indentification同一目录下）  # 存放训练日志、实验结果、图表

├──data（和Bird_Species_Indentification同一目录下）   # 存放所有数据
│   ├── raw/        # 存放原始下载的音频文件
│   └── processed/  # 存放后续处理后的文件（特征、列表等
