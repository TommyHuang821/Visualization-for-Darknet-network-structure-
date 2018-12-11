# -*- coding: utf-8 -*-
"""
Visualization for Darknet (.cfg format)
@author: Tommy Huang, chih.sheng.huang821@gmail.com
"""
from fun_plot_digraph import plot_graph

path_cfg='yolov3.cfg'
format_output_figure='png'


if __name__=='__main__':
    savefilename=path_cfg.split('.cfg')[0]
    grap_g=plot_graph(path_cfg,savefilename, format=format_output_figure)
    grap_g.view()





