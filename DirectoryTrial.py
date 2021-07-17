# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 16:43:17 2021

@author: AnnA
"""

import os
print(os.getcwd())



os.mkdir('Dataset')

os.chdir('H:\\FaceDetection\\Dataset')
os.mkdir('Train')
os.mkdir('Test')
print(os.listdir())
