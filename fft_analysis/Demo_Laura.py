"""
Created on Tue Dec 12 19:13:46 2023

@author: Laura Ibagon
"""

from cellbedformLaura import CellBedform

Q_try=[0.5]
Lo_try=[13]
for q_value in Q_try:
    for L_value in Lo_try:
        cb = CellBedform(grid=(100, 25), D=0.62, Q=q_value, L0=L_value, b=1.7, y_cut=10)
        cb.run(60)
        cb.save_images(f'test_demo_{q_value}_{L_value}','bed')
        cb._plot()
        cb.show()
