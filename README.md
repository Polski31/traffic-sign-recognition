# traffic-sign-recognition

Simple school project.
Program recognizes traffic signs and adds labels to them.

Required libraries:
* opencv-contrib-python
* numpy
* scikit-learn
* scikit-image
* imutils
* matplotlib
* tensorflow==2.0.0

Usage
- training
  - python recognition/train.py --dataset <dataset with images> --model <CNN modeldestination/model.model> --plot <destination/plot.png>
- recognition
  - python GUI/main_app.py
    - 1st button - recognize signs:
      - 1st button - trained model directory
      - 2nd button - recognized images with labels destination
      - 3rd button - csv file with labels
      - 4th button - submit
    - 2nd button - show image
