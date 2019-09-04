# Visualization-for-Darknet-network-structure-
[2019/09/04] adding node [conv_lstm]

[2019/05/14] Modify the number of output channel in residual operation (ElewiseSUM)

For instance, layer -1 (128 * 32 * 32) and layer 66 (256 * 32 * 32) (ch * w * h)

residual (layer -1 and layer 66) = 128 * 32 * 32

[2019/03/20] adding node [crnn]

Visualizing the darknet network structure by .cfg file, and also provide the output featrue map size for each layer.

# How to use
Example: python3 main_DarknetVisualization.py 

"import .cfg path" and "Result save format" can be modified in the 'main_DarknetVisualization.py' directly.

Result save format: default: 'png'. However, 'pdf' is better choice.

<img src="https://github.com/TommyHuang821/Visualization-for-Darknet-network-structure-/blob/master/fig/de1.png" width="60%" height="60%">

# Output Example
YOLOv2

<img src="https://github.com/TommyHuang821/Visualization-for-Darknet-network-structure-/blob/master/yolov2.gv.png" width="10%" height="10%">

YOLOv3

<img src="https://github.com/TommyHuang821/Visualization-for-Darknet-network-structure-/blob/master/yolov3.gv.png" width="10%" height="10%">

# Requirements

python3, graphviz
