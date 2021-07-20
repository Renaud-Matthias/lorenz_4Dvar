#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 18 11:44:08 2021

@author: Matthias
"""

class Config :
    
    __init__(self,path_config_file) :
        '''
        path_config_file : path to the config file
        '''
        self.path = path_config_file
        
        self.config = {}
    
    def fill(self) :
        