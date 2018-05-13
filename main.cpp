#include "colorDescriptor.h"

int main(void)
{
	initModule_nonfree();
	///////////////////////////////////Color Image Inputing///////////////////////////////////
	Mat img1 = imread("../opencv_ColorDescriptor/image/paris1.jpg", 1);
	Mat img2 = imread("../opencv_ColorDescriptor/image/paris2.jpg", 1);

	Mat img1GRAY, img2GRAY;
	cvtColor(img1, img1GRAY, CV_BGR2GRAY);
	cvtColor(img2, img2GRAY, CV_BGR2GRAY);

	//////////////////////////////////Features Extractor//////////////////////////////////
	vector<KeyPoint> kpts1, kpts2;
	Ptr<FeatureDetector> feature2D = FeatureDetector::create("SIFT");
	feature2D->detect(img1GRAY, kpts1);
	feature2D->detect(img2GRAY, kpts2);

	////////////////////////////Color Descriptor Extractor//////////////////////////////
	Mat desc1, desc2;
	Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create("SIFT");
	double t = (double)getTickCount();

	//vector<KeyPoint> outKpts1, outKpts2;
	//outKpts1.reserve(kpts1.size());
	//outKpts2.reserve(kpts2.size());
	computeOpponentColorDescriptorImpl(descriptor, img1, kpts1, desc1);
	computeOpponentColorDescriptorImpl(descriptor, img2, kpts2, desc2);

	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "OpponentColor Descriptor Times (s): " << t << endl;
	//////////////////////////////Rough Descriptor Match////////////////////////////////
	vector<DMatch> matches;
	FlannBasedMatcher matcher;
	matcher.match(desc1, desc2, matches);
	//////////////////////////////Delete Error Descriptor//////////////////////////////
	//-- Step 4: Remove mismatches by vector field consensus (VFC)
	vector<DMatch> correctMatches;
	removeMismatchesByVFC(matches, kpts1, kpts2, correctMatches);

	Mat img_correctMatches;
	drawMatches(img1, kpts1, img2, kpts2, correctMatches, img_correctMatches, Scalar::all(-1), \
		        Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

	namedWindow("PreciseMatchWithVFC");
	imshow("PreciseMatchWithVFC", img_correctMatches);
	imwrite("../opencv_ColorDescriptor/image/_match.jpg", img_correctMatches);
	waitKey(0);

	system("pause");

	return 0;
}


