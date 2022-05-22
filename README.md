# Natural Scene Classification
This is an image classification demo.

The original notebook was downloaded from: 
https://www.kaggle.com/code/pranjalsoni17/natural-scene-classification

## Download the app from github
The github repo for this project is `https://github.com/shawn-becker-angel/natural-scene-classification`  
```
cd to local parent directory of local copy of repo
Use `git clone <repo-url>` to create subfolder `natural-scene-classification`
cd to subfolder
verify that `git branch` returns `master`
```
### Create a virtual environment

```
python3 -m venv venv  
source venv/bin/activate  
python3 -m pip install --upgrade pip  
pip install -r requirements.txt  
```
## Setup vscode   
Use shift-command-p to open the Command Palett  
Select `Python: Select Interpreter` 
Select or enter `./venv/bin/python`  

## Download and configure the local dataset
Download the remote dataset from:  
https://www.kaggle.com/datasets/puneet6060/intel-image-classification  
to create this local folder structure:  
`./kaggle/input/<segment>/<category>/<jpg_files>`  
where segment is `(seg_train, seg_test and seg_pred)`  
category is `(buildings, forest, glacier, mountain, sea and street)`  

## Check app settings
verify the following:
```
data_root = "./kaggle/input"
data_dir = "./kaggle/input/seg_train"
test_data_dir = "./kaggle/input/seg_test"
pred_data_dir = "./kaggle/input/seg_pred"
```


open and run all cells of the jupyter notebook in vscode