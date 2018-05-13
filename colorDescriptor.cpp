#include "colorDescriptor.h"


static void convertBGRImageToOpponentColorSpace(const Mat& bgrImage, vector<Mat>& opponentChannels)
{
	if (bgrImage.type() != CV_8UC3)
		CV_Error(CV_StsBadArg, "input image must be an BGR image of type CV_8UC3");

	// Prepare opponent color space storage matrices.
	opponentChannels.resize(3);
	opponentChannels[0] = cv::Mat(bgrImage.size(), CV_8UC1); // R-G RED-GREEN
	opponentChannels[1] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G-2B YELLOW-BLUE
	opponentChannels[2] = cv::Mat(bgrImage.size(), CV_8UC1); // R+G+B

	for (int y = 0; y < bgrImage.rows; ++y)
	{
		for (int x = 0; x < bgrImage.cols; ++x)
		{
			Vec3b v = bgrImage.at<Vec3b>(y, x);
			uchar& b = v[0];
			uchar& g = v[1];
			uchar& r = v[2];
			// (R - G)/sqrt(2), but converted to the destination data type
			opponentChannels[0].at<uchar>(y, x) = saturate_cast<uchar>(0.5f    * (255 + g - r));
			// (R + G - 2B)/sqrt(6), but converted to the destination data type
			opponentChannels[1].at<uchar>(y, x) = saturate_cast<uchar>(0.25f   * (510 + r + g - 2 * b));
			// (R + G + B)/sqrt(3), but converted to the destination data type
			opponentChannels[2].at<uchar>(y, x) = saturate_cast<uchar>(1.f / 3.f * (r + g + b));
		}
	}
	//namedWindow("opponentChannels[0]");
	//imshow("opponentChannels[0]", opponentChannels[0]);
	//namedWindow("opponentChannels[1]");
	//imshow("opponentChannels[1]", opponentChannels[1]);
	//namedWindow("opponentChannels[2]");
	//imshow("opponentChannels[2]", opponentChannels[2]);
	//imwrite("o1.jpg", opponentChannels[0]);
	//imwrite("o2.jpg", opponentChannels[1]);
	//imwrite("o3.jpg", opponentChannels[2]);
	//waitKey(0);
}


void computeOpponentColorDescriptorImpl(Ptr<DescriptorExtractor>& descriptor, const Mat& bgrImage, \
	                                    vector<KeyPoint>& keypoints, Mat& descriptors )
{
    vector<Mat> opponentChannels;
    convertBGRImageToOpponentColorSpace( bgrImage, opponentChannels );

    const int N = 3; // channels count
    vector<KeyPoint> channelKeypoints[N];
    Mat channelDescriptors[N];
    vector<int> idxs[N];

    // Compute descriptors three times, once for each Opponent channel to concatenate into a single color descriptor
    int maxKeypointsCount = 0;
    for( int ci = 0; ci < N; ci++ )
    {
        channelKeypoints[ci].insert( channelKeypoints[ci].begin(), keypoints.begin(), keypoints.end() );
        // Use class_id member to get indices into initial keypoints vector
		for (size_t ki = 0; ki < channelKeypoints[ci].size(); ki++)
		{
			channelKeypoints[ci][ki].class_id = (int)ki;
		}

		descriptor->compute( opponentChannels[ci], channelKeypoints[ci], channelDescriptors[ci] );
        idxs[ci].resize( channelKeypoints[ci].size() );
        for( size_t ki = 0; ki < channelKeypoints[ci].size(); ki++ )
        {
            idxs[ci][ki] = (int)ki;
        }
        std::sort( idxs[ci].begin(), idxs[ci].end(), KP_LessThan(channelKeypoints[ci]) );
        maxKeypointsCount = std::max( maxKeypointsCount, (int)channelKeypoints[ci].size());
    }

    vector<KeyPoint> outKeypoints;
    outKeypoints.reserve( keypoints.size() );

    int dSize = descriptor->descriptorSize();
    Mat mergedDescriptors( maxKeypointsCount, 3*dSize, descriptor->descriptorType() );
    int mergedCount = 0;
    // cp - current channel position
    size_t cp[] = {0, 0, 0};
    while( cp[0] < channelKeypoints[0].size() &&
           cp[1] < channelKeypoints[1].size() &&
           cp[2] < channelKeypoints[2].size() )
    {
        const int maxInitIdx = std::max( 0, std::max( channelKeypoints[0][idxs[0][cp[0]]].class_id,
                                                      std::max( channelKeypoints[1][idxs[1][cp[1]]].class_id,
                                                                channelKeypoints[2][idxs[2][cp[2]]].class_id ) ) );

        while( channelKeypoints[0][idxs[0][cp[0]]].class_id < maxInitIdx && 
			   cp[0] < channelKeypoints[0].size() ) 
		{ 
			cp[0]++;
		}
        while( channelKeypoints[1][idxs[1][cp[1]]].class_id < maxInitIdx && 
			   cp[1] < channelKeypoints[1].size() ) 
		{ 
			cp[1]++; 
		}
        while( channelKeypoints[2][idxs[2][cp[2]]].class_id < maxInitIdx && 
			   cp[2] < channelKeypoints[2].size() ) 
		{ 
			cp[2]++; 
		}
		if (cp[0] >= channelKeypoints[0].size() || cp[1] >= channelKeypoints[1].size() ||
			cp[2] >= channelKeypoints[2].size())
		{
			break;
		}

        if( channelKeypoints[0][idxs[0][cp[0]]].class_id == maxInitIdx &&
            channelKeypoints[1][idxs[1][cp[1]]].class_id == maxInitIdx &&
            channelKeypoints[2][idxs[2][cp[2]]].class_id == maxInitIdx )
        {
            outKeypoints.push_back( keypoints[maxInitIdx] );
            // merge descriptors
            for( int ci = 0; ci < N; ci++ )
            {
                Mat dst = mergedDescriptors(Range(mergedCount, mergedCount+1), Range(ci*dSize, (ci+1)*dSize));
                channelDescriptors[ci].row( idxs[ci][cp[ci]] ).copyTo( dst );
                cp[ci]++;
            }
            mergedCount++;
        }
    }
    mergedDescriptors.rowRange(0, mergedCount).copyTo( descriptors );

    //std::swap( outKeypoints, keypoints );

	for (int i = 0; i < N; ++i)
	{
		channelKeypoints[i].clear();
		channelKeypoints[i].shrink_to_fit();
		idxs[i].clear();
		idxs[i].shrink_to_fit();
	}
	opponentChannels.clear();
	opponentChannels.shrink_to_fit();
}


void removeMismatchesByVFC(const vector<DMatch> matches, const vector<KeyPoint>& kpts1, \
	                       const vector<KeyPoint>& kpts2, vector<DMatch>& correctMatches)
{
	vector<Point2f> X;  vector<Point2f> Y;
	X.clear();          Y.clear();
	for (unsigned int i = 0; i < matches.size(); i++) {
		int idx1 = matches[i].queryIdx;
		int idx2 = matches[i].trainIdx;
		X.push_back(kpts1[idx1].pt);
		Y.push_back(kpts2[idx2].pt);
	}
	//-- VFC 向量场稀疏剔除过程...
	double t = (double)getTickCount();
	VFC myvfc;
	myvfc.setData(X, Y);
	myvfc.optimize();
	vector<int> matchIdx = myvfc.obtainCorrectMatch();
	t = ((double)getTickCount() - t) / getTickFrequency();
	cout << "VFC Times (s): " << t << endl;

	//vector< DMatch > correctMatches;
	correctMatches.clear();
	for (unsigned int i = 0; i < matchIdx.size(); i++) {
		int idx = matchIdx[i];
		correctMatches.push_back(matches[idx]);
	}
}
