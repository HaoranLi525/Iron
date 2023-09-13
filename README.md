## 1. Training
+ Creating a virtual environment in terminal: `conda create --name <env> --file requirement.txt`.
+ Prepare dataset:
  + move all images from training set into dataset/train/image, move all labels into dataset/train/label
  + move all images from testing set into dataset/test/image, move all labels into dataset/test/label
+ Run `python Train.py`
+ Modify the `--load` in `Test.py` with the best epoch obatained through training
+ Run `python Test.py --data_root root of the desire image` to test the model.
