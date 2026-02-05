# 网格模式（自适应，每个单元独立颜色范围）
python tests/analyze_recorded_data.py --max-display 1286

# 单独保存模式（每个单元独立自适应）
python tests/analyze_recorded_data.py --plot-mode individual

# 随机采样模式（自适应）
python tests/analyze_recorded_data.py --plot-mode sample --max-display 64