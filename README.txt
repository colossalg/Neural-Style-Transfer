CITS4404 Group Project
======================

This folder contains all of the resources for the CITS4404 group project. The
group members for this team were:

+ Angus Wylie 		(21962246)
+ Lucie Cunningham 	(22260943)
+ Jehan Sappideen  	(19523162)


Contents
--------

The directory tree looks roughly as follows:


.
|- Code
| 	|- Content ...
|	|- Texture ...
| 	|- Results ...
| 	|- Content Reconstruction.ipynb
| 	|- Texture Reconstruction.ipynb
| 	|- Style Transfer.ipynb
|
|- Papers
| 	|- Gatys et al. Style Transfer.pdf
| 	|- Gatys et al. Texture Synthesis.pdf
|
|- Report
| 	|- Images ...
| 	|- CITS4404-Report.md
| 	|- CITS4404-Report.pdf


The Content, Texture and Results directories contain the corresponding images
used and generated in the content reconstruction, texture reconstruction and 
style transfer notebooks.

Content Reconstruction.ipynb is the notebook containing the content reconstruction implementation.
Texture Reconstruction.ipynb is the notebook containing the texture reconstruction implementation.
Style Transfer.ipynb is the noteook containing the style transfer implementation.

The Papers directory contains the two papers:
	"Style transfer using convolutional neural networks" and
	"Texture synthesis using convolutional neural networks"
by Leon A. Gatys and his colleagues that this work is based off.

The Report directory contains the report for this project both in the original
markdown and as a pdf.


Running the Code
----------------

The notebooks were developed using TensorFlow in the Google Colab environment to
make use of the GPU kernel. It is recomended that you run the code in Colab if
possible. 

You may need to change some variables within the notebooks regarding the paths 
of the images. In particular there is a variable called 'root' which must be
changed in each notebook.
