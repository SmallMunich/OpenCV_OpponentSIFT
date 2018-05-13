#ifndef _COLOR_DESCRIPTOR_
#define _COLOR_DESCRIPTOR_
#include <iostream>
#include <opencv2\opencv.hpp>
#include <opencv2\features2d\features2d.hpp>
#include <opencv2\nonfree\features2d.hpp>
#include <opencv2\nonfree\nonfree.hpp>
#include "vfc.h"
using namespace std;
using namespace cv;

static void convertBGRImageToOpponentColorSpace(const Mat& bgrImage, vector<Mat>& opponentChannels);

struct KP_LessThan
{
	KP_LessThan(const vector<KeyPoint>& _kp) : kp(&_kp) {}
	bool operator()(int i, int j) const
	{
		return (*kp)[i].class_id < (*kp)[j].class_id;
	}
	const vector<KeyPoint>* kp;
};

void computeOpponentColorDescriptorImpl(Ptr<DescriptorExtractor>& descriptor, const Mat& bgrImage, \
	                                    vector<KeyPoint>& keypoints, Mat& descriptors);


void removeMismatchesByVFC(const vector<DMatch> matches, const vector<KeyPoint>& kpts1, \
	                       const vector<KeyPoint>& kpts2, vector<DMatch>& correctMatches);

#endif



