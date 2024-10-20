# 根据对应json 绘制 曲线图 
dir="work_dirs/c3d_sports1m-pretrained_8xb30-16x1x1-45e_ucf101-rgb/20241009_230103/vis_data/20241009_230103.json"


metric="top1_acc"
python tools/analysis_tools/analyze_logs.py plot_curve \
    $dir\
    --keys $metric\
    --out $dir'_'${metric:0:3}'.png'


metric="loss"
python tools/analysis_tools/analyze_logs.py plot_curve \
    $dir\
    --keys $metric\
    --out $dir'_'${metric:0:3}'.png'