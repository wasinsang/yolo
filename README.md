1. install anaconda
2. open anaconda cmd and run
	conda create --name yolo-env1 python=3.12
3. after finish set up env then run command
	activate yolo-env1
4. run
	pip install label-studio
5. run label-studio by run this command
	label-studio start
6. open label-studio page and sign up for fake account 
7. create project
	- project name
		- name a project name
	- data import 
		- add image to do labeling (add less than 100 pics)
	- labeling setup
		- choese object detection with bounding boxes
		- add all class in add label names (the object you want to detect such as Honda,Toyota)
	- click add 
	- click save
8. statr labeling by click on the image
	- select class and then draw retagle around your object 
	- after finish click submit and go to next image 
	- do again until last image
9. after finsh , click project name above the window then click export on the right upper conner
10. select YOLO 
11. click export
12. after download rename file to data.zip
13. Open colab and upload data.zip into colab
14. change GPU to T4 GPU
15. train model ,test and deploy
16. after download my_model , back to conda cmd 
	- activate project 
	- pip install ultralytics
	- if laptop has CPU then 
		- pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
	- download code from efelectronic
		- curl -o yolo_detect.py https://www.ejtech.io/code/yolo_detect.py
17. run script by using
	- python yolo_detect.py --model my_model.pt --source use0 --resolution 1920*1080