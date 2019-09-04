# -*- coding: utf-8 -*-
"""
parse darknet .cfg file to a list

@author: Tommy Huang, chih.sheng.huang821@gmail.com
"""
import configparser
from collections import defaultdict
import io


def unique_config_sections(config_file):
    """Convert all config sections to have unique names.

    Adds unique suffixes to config sections for compability with configparser.
    """
    section_counters = defaultdict(int)
    output_stream = io.StringIO()
    with open(config_file) as fin:
        for line in fin:
            if line.startswith('['):
                section = line.strip().strip('[]')
                _section = section + '_' + str(section_counters[section])
                section_counters[section] += 1
                line = line.replace(section, _section)
            output_stream.write(line)
    output_stream.seek(0)
    return output_stream


def yolo_parse(path_cfg):  
    unique_config_file = unique_config_sections(path_cfg)
    cfg_parser = configparser.ConfigParser()
    cfg_parser.read_file(unique_config_file)
    
    all_layers=[]
    image_height = int(cfg_parser['net_0']['height'])
    image_width = int(cfg_parser['net_0']['width'])
    image_channel = int(cfg_parser['net_0']['channels'])
    image_structure={'image_height':image_height,'image_width':image_width,'image_channel':image_channel}

    # data augmentation, training parameter:
    angle = float(cfg_parser['net_0']['angle'])
    saturation = float(cfg_parser['net_0']['saturation']) # 飽和度
    exposure = float(cfg_parser['net_0']['exposure']) #曝光度
    hue = float(cfg_parser['net_0']['hue']) #色調
    training_DataAugmentation={'angle':angle,'saturation': saturation,
                        'exposure':exposure,'hue':hue}
    
    # rnn setting
    time_steps, track, augment_speed='','',''
    batch, subdivisions='',''
    momentum, weight_decay, learning_rate='','',''
    burn_in, max_batches, policy='','',''
    steps, scales='',''
    sgdr_cycle, sgdr_mult, seq_scales='','',''
    if 'batch' in cfg_parser['net_0']:
        batch = int(cfg_parser['net_0']['batch'].split('#')[0])
    if 'subdivisions' in cfg_parser['net_0']:
        subdivisions = int(cfg_parser['net_0']['subdivisions'].split('#')[0])

    if 'track' in cfg_parser['net_0']: track = int(cfg_parser['net_0']['track'].split('#')[0])
    if 'time_steps' in cfg_parser['net_0']: time_steps = int(cfg_parser['net_0']['time_steps'].split('#')[0])
    if 'augment_speed' in cfg_parser['net_0']:augment_speed = int(cfg_parser['net_0']['augment_speed'].split('#')[0])                             
    
    # training learning parameter:
    if 'momentum' in cfg_parser['net_0']:
        momentum = float(cfg_parser['net_0']['momentum'])
    if 'weight_decay' in cfg_parser['net_0']:
        weight_decay = float(cfg_parser['net_0']['decay']) if 'net_0' in cfg_parser.sections() else 5e-4
    if 'learning_rate' in cfg_parser['net_0']:
        learning_rate = float(cfg_parser['net_0']['learning_rate']) #學習率
    if 'burn_in' in cfg_parser['net_0']:    
        burn_in = int(cfg_parser['net_0']['burn_in']) #學習率控制的參數
    if 'max_batches' in cfg_parser['net_0']:    
        max_batches = int(cfg_parser['net_0']['max_batches']) #跌代次數
    
    if 'policy' in cfg_parser['net_0']:    
        policy = (cfg_parser['net_0']['policy']) #學習率策略 
    if 'steps' in cfg_parser['net_0']:
        steps = list(map(int,cfg_parser['net_0']['steps'].split(','))) #學習率變動步長
    if 'scales' in cfg_parser['net_0']:
        scales = list(map(float,cfg_parser['net_0']['scales'].split(','))) #學習率變動因子
    
    #
    if 'sgdr_cycle' in cfg_parser['net_0']:
        sgdr_cycle = list(map(float,cfg_parser['net_0']['sgdr_cycle'].split(','))) 
    if 'sgdr_mult' in cfg_parser['net_0']:
        sgdr_mult = list(map(float,cfg_parser['net_0']['sgdr_mult'].split(','))) 
    if 'seq_scales' in cfg_parser['net_0']:
        seq_scales = list(map(float,cfg_parser['net_0']['seq_scales'].split(',')))
        
    training_parameter={'batch':batch,
                        'subdivisions':subdivisions,
                        'momentum':momentum,
                        'weight_decay': weight_decay,
                        'learning_rate': learning_rate,
                        'burn_in':burn_in,
                        'max_batches':max_batches,
                        'augment_speed':augment_speed,
                        'policy':policy,
                        'steps':steps,
                        'track':track,
                        'scales':scales,
                        'sgdr_cycle':sgdr_cycle,
                        'sgdr_mult': sgdr_mult, 
                        'seq_scales': seq_scales}
    
    structure={'type':'input',
               'image_structure':image_structure,
               'training_DataAugmentation': training_DataAugmentation,
               'training_parameter': training_parameter }
    all_layers.append(structure)
    #all_layers.append([image_height,image_width,image_channel])
    count_layer = 0
    for section in cfg_parser.sections():
        print('Parsing section {}'.format(section)) 
        if section.startswith('convolutional'):
            filters = int(cfg_parser[section]['filters'])
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            
            if all_layers[-1]['type']!='input': 
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=0
    
            structure={'type':'convolutional','prev_layer': prev_layer,'layer': count_layer ,'filters':filters,'size':size,'stride':stride,'pad':pad,'activation':activation,'batch_normalize':batch_normalize}
            print('prev_layer:{}, layer:{}, conv., filter size={}, kernel size={}, stride={}, pad={}, activation={}, batch_normalize={}'.format(prev_layer,count_layer,filters,size,stride,pad,activation,batch_normalize))
            all_layers.append(structure)
            
        elif section.startswith('maxpool'):
            size = int(cfg_parser[section]['size'])
            stride = int(cfg_parser[section]['stride'])
            
            if all_layers[-1]['type']!='input':
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=''
                
            structure={'type':'maxpool','prev_layer': prev_layer,'layer': count_layer ,'size':size,'stride':stride}
            print('prev_layer:{}, layer:{}, maxpool, size={}, stride={}'.format(prev_layer,count_layer,size,stride))
            all_layers.append(structure)
            prev_layer = all_layers[-1]['layer']
            
        elif section.startswith('avgpool'):
    
            if all_layers[-1]['type']!='input':
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=''
                
            structure={'type':'avgpool', 'prev_layer': prev_layer,'layer': count_layer}
            print('prev_layer:{}, layer:{}, avgpool'.format(prev_layer, count_layer))
            all_layers.append(structure)
            prev_layer = all_layers[-1]['layer']
            
        elif section.startswith('route'):
            ids = [int(i) for i in cfg_parser[section]['layers'].split(',')]
            layers=[]
            for i in ids: 
                if i>0:
                    layers.append(all_layers[i+1])
                else:
                    layers.append(all_layers[i])
            #layers = [all_layers[i] for i in ids]
            if len(layers) > 1:
                prev_layer=[i['layer'] for i in layers]  
                structure={'type':'concatenate', 'prev_layer': prev_layer, 'layer': count_layer}
                print('prev_layer:{}, layer:{}, concatenate'.format(prev_layer, count_layer))
                all_layers.append(structure)  
            else:
                skip_layer = layers[0]  # only one layer to route
                skip_layer = skip_layer['layer']
                if skip_layer<0:
                    prev_layer=count_layer+skip_layer
                else: 
                    prev_layer=skip_layer
                structure={'type':'shortcut', 'prev_layer': prev_layer, 'layer': count_layer}
    
                print('prev_layer:{}, layer:{}, shortcut'.format(prev_layer, count_layer))
                all_layers.append(structure) 
                
    
        elif section.startswith('reorg'):
                # stride = int(cfg_parser[section]['stride'])
                block_size = int(cfg_parser[section]['stride'])
                assert block_size == 2, 'Only reorg with stride 2 supported.'
                
                if all_layers[-1]['type']!='input':
                    if all_layers[-1]['type']=='shortcut':
                        prev_layer=all_layers[-1]['prev_layer']
                    else:
                        prev_layer=all_layers[-1]['layer']  
                else: 
                    prev_layer=''
                
                structure={'type':'reorg', 'prev_layer': prev_layer, 'layer': count_layer,'stride': block_size}
                print('prev_layer:{}, layer:{}, reorg'.format(prev_layer, count_layer))
                all_layers.append(structure) 
        elif section.startswith('region'):
            if all_layers[-1]['type']!='input':
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=''
            anchors=cfg_parser[section]['anchors']
            classes = int(cfg_parser[section]['classes'])        
            structure={'type':'region', 'prev_layer': prev_layer, 'layer': count_layer, 'classes':classes, 'anchors':anchors }
            print('prev_layer:{}, layer:{}, classes:{}, anchors:{}'.format(prev_layer, count_layer,classes, anchors))
            all_layers.append(structure)
    
        elif section.startswith('shortcut'):
            
            ids = [int(i) for i in cfg_parser[section]['from'].split(',')][0]
            '''
            Becasue I add input image layer in first factor of all_layers list, 
            the corresponding layer code must +1 (if layer code is postive number)
            '''
            if ids>0: 
                ids+=1
             
            activation = cfg_parser[section]['activation']
            prev_layer=[ all_layers[ids]['layer'], all_layers[-1]['layer']]
    
            structure={'type':'residual', 'prev_layer': prev_layer, 'layer': count_layer, 'activation': activation}
            print('prev_layer:{}, layer:{}, residual, activation:{}'.format(prev_layer, count_layer,activation))
            all_layers.append(structure)
            
        elif section.startswith('upsample'):
            stride = int(cfg_parser[section]['stride'])
            if all_layers[-1]['type']!='input':
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=''
                    
            structure={'type':'upsample', 'prev_layer': prev_layer, 'layer': count_layer,'stride':stride}
            print('prev_layer:{}, layer:{}, upsample'.format(prev_layer, count_layer))
            all_layers.append(structure)
        elif section.startswith('yolo'):
            classes = int(cfg_parser[section]['classes']) 
            #mask = [int(i) for i in cfg_parser[section]['mask'].split(',')]
            mask = cfg_parser[section]['mask']   
            anchors=cfg_parser[section]['anchors']
            if all_layers[-1]['type']!='input':
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=''
            structure={'type':'yolo', 'prev_layer': prev_layer, 'layer': count_layer,'classes':classes,'mask': mask,'anchors':anchors}
            print('prev_layer:{}, layer:{}, yolo, classes:{}, mask:{}, anchors:{}'.format(prev_layer, count_layer,classes,mask,anchors))
            all_layers.append(structure)    
      
        elif section.startswith('crnn'):            
            hidden = int(cfg_parser[section]['hidden'])
            size = int(cfg_parser[section]['size'])
            output = int(cfg_parser[section]['output'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            
            if all_layers[-1]['type']!='input': 
                if all_layers[-1]['type']=='shortcut':
                    prev_layer=all_layers[-1]['prev_layer']
                else:
                    prev_layer=all_layers[-1]['layer']  
            else: 
                prev_layer=0  
            structure={'type':'crnn','prev_layer': prev_layer,'layer': count_layer,'output':output ,'hidden':hidden,'size':size,'pad':pad,'activation':activation,'batch_normalize':batch_normalize,'time_steps':time_steps}
            print('prev_layer:{}, layer:{}, crnn., output size={}, hidden size={}, time_steps={},size={}, pad={}, activation={}, batch_normalize={}'.format(prev_layer,count_layer,output, hidden,time_steps, size,pad,activation,batch_normalize))
            all_layers.append(structure)
        elif section.startswith('conv_lstm'):
            size = int(cfg_parser[section]['size'])
            output = int(cfg_parser[section]['output'])
            pad = int(cfg_parser[section]['pad'])
            activation = cfg_parser[section]['activation']
            batch_normalize = 'batch_normalize' in cfg_parser[section]
            peephole = 'peephole' in cfg_parser[section]
            
            prev_layer=all_layers[-1]['layer']  
            
            structure={'type':'conv_lstm','prev_layer': prev_layer,'layer': count_layer,'output':output,'size':size,'pad':pad,'activation':activation,'batch_normalize':batch_normalize,'time_steps':time_steps,'peephole':peephole}
            print('prev_layer:{}, layer:{}, conv_lstm., output size={}, time_steps={},size={}, pad={}, activation={}, batch_normalize={}'.format(prev_layer,count_layer,output,time_steps, size,pad,activation,batch_normalize))
            all_layers.append(structure)
        elif (section.startswith('net') or section.startswith('cost') or
                  section.startswith('softmax')):
                pass  # Configs not currently handled during model definition.    
        count_layer+=1
    return all_layers
