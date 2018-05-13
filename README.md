# OpenCV_OpponentSIFT
The Simple Project is using OpponentColorDescriptor to image matchimg.

Runtime Environment:
    Windows10 x64 + Visual Studio 2015 + OpenCV 2.4.11
    
OpenCV 2.x has the OpponentColorDescriptor class and I can not find the OpponentColorDescriptor class in OpenCV 3.x.

If you using the OpenCV 2.x lib to complete the image matching, maybe you can find the bug about the _block_type_is_valid(phead->nblockuse)
so you will copy the source code of opponentcolordescriptor and revise it. The bug you will revise the scope of data store structure. 
My code update is simple modification about the data structure in OpenCV.
