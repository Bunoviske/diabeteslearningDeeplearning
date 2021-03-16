# README #

Algoritmo que faz o ground truth de classes segmentadas a partir da técnica watershed




# Compilation

Install opencv and latest g++ and c++ 

- mkdir build  
- cd build  
- cmake ..  
- make  


# Execution

Two arguments:  

1- path to images folder  
2- path to classesNumber.txt 

Example  

- ./main ../bruno/ ../../classesConfiguration/classesNumber.txt 


# How to annotate data

1- Give hint marks for the Watershed algorithm  
2- Click 'w' to run watershed and check the segmentation result  
3- Click 'r' to reload and erase the marks  
4- Click 'c' to go to the classification step  
5- When classifying each segmentation region, first click on that region and then insert via terminal the class id  
6- Click 'n' to finalize this image and go to the next one  

AT ANY POINT OF THE PROGRAM, PRESS 'ESC' TO GO TO THE NEXT IMAGE