# -*- coding: utf-8 -*-
"""
create YOLO model nodes and link edges
@author: Tommy Huang, chih.sheng.huang821@gmail.com
"""
import numpy as np   
from graphviz import Digraph
from  fun_parse_cfg import yolo_parse

def plot_graph(path_cfg,savefilename, format):
    # 1. parse yolo .cfg file
    model_layers=yolo_parse(path_cfg)
    # 2. create a digraph
    grap_g = Digraph(savefilename,format=format)
    create_node(grap_g,model_layers)
    link_edge(grap_g,model_layers) 
    return grap_g



def inputimage(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='yellow')
  
def conv(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='lightblue2')
   
def maxpool(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='grey')

def reorg(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='grey')
    
def upsample(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='blue')
    
def yolo_region(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='yellow')

def net(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='white')

def concatenate(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='red')
    
def residual(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='lightgrey')
    
def crnn(grap_g):
    grap_g.attr('node', shape='record', style='filled', color='lightslateblue')

def cal_conv_w_h_ch(h,w,ch,ks,pad,stride,n_filter):
    tmp_inf=[]
    if pad==1:
        pad+=1
    if ks==1:
        pad=0
    if isinstance(h,int):
        h=int((h+pad-(ks-1))/stride)
        tmp_inf.append(h)
    else:
        tmp_inf.append('?')
    if isinstance(w,int):
        w=int((w+pad-(ks-1))/stride) 
        tmp_inf.append(w)
    else:
        tmp_inf.append('?')
    if isinstance(ch,int):
        ch=int(n_filter) 
        tmp_inf.append(ch)
    else:
        tmp_inf.append('?')
    return tmp_inf

def cal_pool_w_h_ch(h,w,ch,stride):
    tmp_inf=[]
    if isinstance(h,int):
        h=int(h/stride)
        tmp_inf.append(h)
    else:
        tmp_inf.append('?')
    if isinstance(w,int):
        w=int(w/stride)
        tmp_inf.append(w)
    else:
        tmp_inf.append('?')
    if isinstance(ch,int):
        tmp_inf.append(ch)
    else:
        tmp_inf.append('?')
    return tmp_inf

def cal_upsample_w_h_ch(h,w,ch,stride):
    tmp_inf=[]
    if isinstance(h,int):
        h=int(h*stride)
        tmp_inf.append(h)
    else:
        tmp_inf.append('?')
    if isinstance(w,int):
        w=int(w*stride)
        tmp_inf.append(w)
    else:
        tmp_inf.append('?')
    if isinstance(ch,int):
        tmp_inf.append(ch)
    else:
        tmp_inf.append('?')
    return tmp_inf            
    
def cal_reorg_w_h_ch(h,w,ch,stride):
    tmp_inf=[]
    if isinstance(h,int):
        h=int(h/stride)
        tmp_inf.append(h)
    else:
        tmp_inf.append('?')
    if isinstance(w,int):
        w=int(w/stride)
        tmp_inf.append(w)
    else:
        tmp_inf.append('?')
    if isinstance(ch,int):
        tmp_inf.append(ch*stride*stride)
    else:
        tmp_inf.append('?')
    return tmp_inf

def cal_crnn_w_h_ch(h,w,ch,ks,pad,stride,output):
    tmp_inf=[]
    if pad==1:
        pad+=1
    if ks==1:
        pad=0
    if isinstance(h,int):
        h=int((h+pad-(ks-1))/stride)
        tmp_inf.append(h)
    else:
        tmp_inf.append('?')
    if isinstance(w,int):
        w=int((w+pad-(ks-1))/stride) 
        tmp_inf.append(w)
    else:
        tmp_inf.append('?')
    if isinstance(ch,int):
        ch=int(output) 
        tmp_inf.append(ch)
    else:
        tmp_inf.append('?')
    return tmp_inf

def create_node(grap_g,all_layers):
    imagesize=[]
    for tmp in all_layers:
        imagesize.append([])
        
    for i_L, layer in enumerate(all_layers):        
        if 'prev_layer' in layer:
            pos_previous=layer['prev_layer']
                    
        if layer['type']=='input':
            h=layer['image_structure']['image_height']
            w=layer['image_structure']['image_width']
            ch=layer['image_structure']['image_channel']
            inputimage(grap_g)
            grap_g.node('Input',r'{ Input |output: %d × %d × %d}' % (h,w,ch))  
            
            #imagesize.append([h,w,ch])
            imagesize[0]=[h,w,ch]
            
        elif layer['type']=='convolutional':              
            conv(grap_g)
            curr_layer=layer['layer']
            n_filter=layer['filters']
            ks=layer['size']
            stride=layer['stride']
            pad=layer['pad']
            act=layer['activation']
            bn=layer['batch_normalize']
            
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                h,w,ch=cal_conv_w_h_ch(h,w,ch,ks,pad,stride,n_filter)
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            if bn:
                grap_g.node('l_'+ str(curr_layer), r'{ Conv_%d (bn)|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (n_filter,ks,stride,pad,act,h,w,ch))
            else:
                grap_g.node('l_'+ str(curr_layer), r'{ Conv_%d |{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (n_filter,ks,stride,pad,act,h,w,ch))
        
        elif layer['type']=='maxpool':     
            maxpool(grap_g)
            curr_layer=layer['layer']
            ks=layer['size']
            stride=layer['stride']
            
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                h,w,ch=cal_pool_w_h_ch(h,w,ch,stride)
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            grap_g.node('l_'+ str(curr_layer), r'{ maxpool|{ks=%d, stri=%d}|output:%s × %s × %s}' % (ks,stride,h,w,ch))
       
        elif layer['type']=='reorg':   
            stride=2

            reorg(grap_g)
            curr_layer=layer['layer']
            stride=layer['stride']
            
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous] 
                
                h,w,ch=cal_reorg_w_h_ch(h,w,ch,stride)
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            grap_g.node('l_'+ str(curr_layer), r'{ reorg|{ks=%d, stri=%d}|output:%s × %s × %s}' % (ks,stride,h,w,ch))
       
        elif layer['type']=='concatenate':
             concatenate(grap_g)
             curr_layer=layer['layer']
             if len(imagesize)>0:
                ch=0
                for tmp_l in pos_previous:
                    tmp_h,tmp_w,tmp_ch=imagesize[tmp_l] 
                    if isinstance(tmp_ch,int):
                        ch+=tmp_ch
                    else:
                        h,w,ch='?','?','?'
                        break
                    if isinstance(tmp_h,int):
                        h=tmp_h
                    else:
                        h,w,ch='?','?','?'
                        break
                    if isinstance(tmp_w,int):
                        w=tmp_w
                    else:
                        h,w,ch='?','?','?' 
                        break
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
             else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
             grap_g.node('l_'+ str(curr_layer), r'{ concatenate|output:%s × %s × %s}' % (h,w,ch))
             
        elif layer['type']=='upsample':
            upsample(grap_g)
            curr_layer=layer['layer']
            stride=layer['stride']
            
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                h,w,ch=cal_upsample_w_h_ch(h,w,ch,stride)
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            grap_g.node('l_'+ str(curr_layer), r'{upsample|s=%d|output:%s × %s × %s}' % (stride,h,w,ch))
        elif (layer['type']=='yolo') :
            yolo_region(grap_g)
            curr_layer=layer['layer']
            classes=layer['classes']
            mask=layer['mask']
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            grap_g.node('l_'+ str(curr_layer), r'{yolo|classes=%d|mask:%s}' % (classes,mask))
        
        elif (layer['type']=='region') :
            yolo_region(grap_g)
            curr_layer=layer['layer']
            classes=layer['classes']
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
            grap_g.node('l_'+ str(curr_layer), r'{yolo|classes=%d}' % (classes))    
        elif (layer['type']=='shortcut') :
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]  
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
        elif (layer['type']=='residual') :
            residual(grap_g)
            curr_layer=layer['layer']
            if len(imagesize)>0:
                cap_ind = np.argmin(np.abs(np.array(pos_previous)-curr_layer))
                h,w,ch=imagesize[pos_previous[cap_ind]]  
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
            grap_g.node('l_'+ str(curr_layer), r'{ElewiseSum|output: %s × %s × %s}' % (h,w,ch))
        elif layer['type']=='crnn': 
            
            crnn(grap_g)
            curr_layer=layer['layer']
            output=layer['output']
            hidden=layer['hidden']
            #time_steps=layer['time_steps']
            ks=layer['size']
            pad=layer['pad']
            act=layer['activation']
            bn=layer['batch_normalize']
            if len(imagesize)>0:
                h,w,ch=imagesize[pos_previous]
                h,w,ch = cal_crnn_w_h_ch(h,w,ch,ks,pad,stride,output)
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                h,w,ch=str(h),str(w),str(ch)
            else:
                h,w,ch='?','?','?'
                #imagesize.append([h,w,ch])
                imagesize[layer['layer']]=[h,w,ch]
                
            if bn:
                #grap_g.node('l_'+ str(curr_layer), r'{crnn_%d (bn)|{ks=%d, s=%d, p=%d , %s}|hidden filter: %d, recurrent time: %d| output: %s × %s × %s}' % (output,ks,stride,pad,act,hidden,time_steps,h,w,ch))
                grap_g.node('l_'+ str(curr_layer)+ '_1', r'{CRNN|Conv_%d (bn)|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (hidden,ks,stride,pad,act,h,w,hidden))
                grap_g.node('l_'+ str(curr_layer)+ '_2', r'{CRNN|Conv_%d (bn)|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (hidden,ks,stride,pad,act,h,w,hidden))
                grap_g.node('l_'+ str(curr_layer)+ '_3', r'{CRNN|Conv_%d (bn)|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (output,ks,stride,pad,act,h,w,output))
            else:
                #grap_g.node('l_'+ str(curr_layer), r'{crnn_%d |{ks=%d, s=%d, p=%d , %s}|hidden filter: %d, recurrent time: %d| output: %s × %s × %s}' % (output,ks,stride,pad,act,hidden, time_steps,h,w,ch))
                grap_g.node('l_'+ str(curr_layer)+ '_1', r'{CRNN|Conv_%d|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (hidden,ks,stride,pad,act,h,w,hidden))
                grap_g.node('l_'+ str(curr_layer)+ '_2', r'{CRNN|Conv_%d|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (output,ks,stride,pad,act,h,w,output))
                grap_g.node('l_'+ str(curr_layer)+ '_3', r'{CRNN|Conv_%d|{ks=%d, s=%d, p=%d , %s}| output: %s × %s × %s}' % (output,ks,stride,pad,act,h,w,output))
            
        
        
def link_edge(grap_g,all_layers):
    for layer in all_layers:
        if 'prev_layer' in layer:
            pos_previous=layer['prev_layer']
            pos_current=layer['layer']
    
            if layer['prev_layer']==0:
                grap_g.edge("Input","l_"+str(pos_current))
            elif layer['type']=="shortcut":pass
            elif  layer['type']=='crnn':
                time_steps=layer['time_steps']
               #grap_g.edge("l_"+ str(pos_current),"l_"+str(pos_current)) 
                grap_g.edge("l_"+ str(pos_previous),"l_"+ str(pos_current)+ '_1')
                grap_g.edge("l_"+ str(pos_current)+ '_1',"l_"+str(pos_current) + '_2') 
                grap_g.edge("l_"+ str(pos_current)+ '_2',"l_"+str(pos_current) + '_3') 
                grap_g.edge("l_"+ str(pos_current)+ '_3',"l_"+str(pos_current) + '_1',label='recurrent: '+str(time_steps) + ' times') 
            else:                   
                if np.size(pos_previous)==1:
                    if all_layers[pos_previous]['type']=='crnn':
                        grap_g.edge("l_"+ str(pos_previous)+ '_3',"l_"+str(pos_current))    
                    else:
                        grap_g.edge("l_"+ str(pos_previous),"l_"+str(pos_current))
                elif np.size(pos_previous)>1:
                    for tmp_l in pos_previous:
                        grap_g.edge("l_"+ str(tmp_l),"l_"+str(pos_current))        


    
    
    
        

