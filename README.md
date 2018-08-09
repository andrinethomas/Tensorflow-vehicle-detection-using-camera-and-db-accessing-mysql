# Tensorflow-vehicle-detection-using-camera-and-db-accessing-mysql 
The project developed using TensorFlow to detect the License Plate from a car and uses the Tesseract Engine to recognize the charactes from the detected plate.

### Software Packs Needed

* <a href='https://www.anaconda.com/download/'>Anaconda 3</a> (**Tool comes with most of the required python packages along with python3 & spyder IDE**)<br>
* <a href='https://github.com/tesseract-ocr/tesseract'>Tesseract Engine</a> (**Doesn't need to be installed cause I included the trained OCR data along in the repository**)<br>

### Python Packages Needed

* <a href='https://github.com/tensorflow/tensorflow'>Tensorflow</a><br>
* <a href='https://github.com/skvark/opencv-python'>openCV</a><br>
* <a href='https://github.com/madmaze/pytesseract'>pytesseract</a><br>
* <a href='https://github.com/tzutalin/labelImg'>labelImg</a><br>

### ABOUT PROJECT

* TensorFlow is an open-source software library **(Deep learning)** for dataflow programming across a range of tasks. It is a symbolic math library, and also used for machine learning applications such as neural networks. So we have planned to use it for number plate detection.The images are been collected by the ip camera's that are been connected and it identify the scanned values to the mysql db.
#### TRAINING PHASE -- IMAGE LABELING

* Collected the set of 1200 images (Cars along with number plate) from the sources such as Google Images and Flickr. Then annotated the set of images by drawing the boundary box over the number plates to send it for the training phase.
  * The Annoation gives the co-ordinates of license plates such as **(xmin, ymin, xmax, ymax)**
  * Then the co-ordinates are saved into a **XML** file
  * All the XML files are grouped and the Co-ordinates are saved in **CSV** file.
  * Then the CSV file is converted into **TensorFlow record format**.
* The set of other separate 10 images also gone through the above steps and saved as **Test Record file** 
<p align="center">
  
</p>  

#### GPU TRAINING

* By using the **Tensorflow-gpu** version, the set of annotated images were sent into the Convolutional neural network called as **ssd-mobilenet** where the metrics such as model learning rate, batch of images sent into the network and evaluation configurations were set. The training phase of the model took several days. At last the model came around with the positive result and detected the number plate over the input images.
<p align="center">
  
</p> 

#### OCR PART

* Then the detected number plate is cropped using Tensorflow, By using the Google **Tesseract-OCR** (Package originally developed to scan hard copy documents to filter out the characters from it) the picture undergoes some coversions using **computer vision** package then the charcters are filtered out.

#### CROP
<p align="left">
   
</p>

#### CONVERSION
<p align="left">
   
</p> 
<p align="center">
  
</p> 

#### MOTION DETECTION PART

* The image capturing has been implemented to capture the picture of moving vehicle by using the **IP camera with motion detection** where the boundary frame of the camera is fixed (boundary frame fixing changes depending on the camera view). If the vehicle touches the boundary the picture is captured. The transfering part of the pictures by the camera is sent to the system using the FTP 

#### PUSH TO MYSQL PART

* The derived datas from each images are been inserted with the date and time stamp to the local dp of the configured system.

Connection with multiple ip cameras **(In progress)**

