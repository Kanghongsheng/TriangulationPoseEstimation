#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
  const Mat& image_1,const Mat& image_2,
  vector<KeyPoint>& kp1,
  vector<KeyPoint>& kp2,
  vector<DMatch>& matches,
  vector<DMatch>& good_matches
 			);
  
void pose_estimation(
  vector<KeyPoint>& kp1,
  vector<KeyPoint>& kp2,
  vector<DMatch>& matches,
  Mat& R,Mat& t
);

Point2d pixel2cam(
  const Mat& K,const Point2d& p
);

int main(int argc,char** argv)
{
  Mat image_1=imread(argv[1],CV_LOAD_IMAGE_COLOR);
  Mat image_2=imread(argv[2],CV_LOAD_IMAGE_COLOR);
  vector<KeyPoint> kp1,kp2;
  vector<DMatch> matches;
  vector<DMatch> good_matches;
  Mat outimg;
  Mat R,t;
  
  find_feature_matches(image_1,image_2,kp1,kp2,matches,good_matches);
  drawMatches(image_1,kp1,image_2,kp2,good_matches,outimg);
  imshow("BruteForce-Hamming",outimg);
  pose_estimation(kp1,kp2,good_matches,R,t);
  waitKey(0);
  
  //T=t^
  Mat T=(Mat_<double>(3,3)<<0,                                -t.at<double>(2,0),              t.at<double>(1,0),
                                                 t.at<double>(2.0),       0,                                        -t.at<double>(0,0),
                                                 -t.at<double>(1,0),      t.at<double>(0,0),             0);
  //验证对极几何约束
  Mat K=(Mat_<double> (3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  for(DMatch m:good_matches)
  {
    Point2d p1=pixel2cam(K,kp1[m.queryIdx].pt);
    Mat y1=(Mat_<double>(3,1)<<p1.x,p1.y,1);
    Point2d p2=pixel2cam(K,kp2[m.trainIdx].pt);
    Mat y2=(Mat_<double>(3,1)<<p2.x,p2.y,1);
    Mat epipolar_constrain;
    epipolar_constrain=y2.t()*T*R*y1;
    cout<<"epipolar_constrain= \n"<<epipolar_constrain<<endl;
  };
  return 0;
}

void find_feature_matches(const Mat& image_1,const Mat& image_2,
  vector<KeyPoint>& kp1,vector<KeyPoint>& kp2,
  vector<DMatch>& matches,vector<DMatch>& good_matches){
    //初始化
    Mat descriptors_1,descriptors_2;
    Ptr<FeatureDetector> detectors=ORB::create();
    Ptr<DescriptorExtractor> descriptors=ORB::create();
    Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
    good_matches.clear();
    
    //特征点检测及匹配
    detectors->detect(image_1,kp1);
    detectors->detect(image_2,kp2);
    descriptors->compute(image_1,kp1,descriptors_1);
    descriptors->compute(image_2,kp2,descriptors_2);
    matcher->match(descriptors_1,descriptors_2,matches);
    
    //特征匹配的筛选
    double minDist=10000;
    double maxDist=0;
    for(int i=0;i<descriptors_1.rows;i++)
    {
      int dist=matches[i].distance;
      if(dist<=minDist){minDist=dist;}
      if(dist>maxDist){maxDist=dist;}
    }
    printf("minDist=%f\n",minDist);
    printf("maxDist=%f\n",maxDist);
    
   // vector<DMatch> good_matches;
    for(int i=0;i<descriptors_1.rows;i++)
    {
      if(matches[i].distance<=max(3*minDist,30.0)){
	good_matches.push_back(matches[i]);
      }
    }
  }

void pose_estimation(
   vector<KeyPoint>& kp1,
   vector<KeyPoint>& kp2,
   vector<DMatch>& good_matches,
  Mat& R,Mat& t
){
  //相机内参
  Mat K=(Mat_<double> (3,3)<<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  vector<Point2f> points1,points2;
  for(int i=0;i<(int)good_matches.size();i++)
  {
    points1.push_back(kp1[good_matches[i].queryIdx].pt);
    points2.push_back(kp2[good_matches[i].trainIdx].pt);
  }
 Mat fundamental_matrix;
 fundamental_matrix=findFundamentalMat(points1,points2,CV_FM_8POINT);
 cout<<"fundamental_matrix= \n"<<endl<<fundamental_matrix<<endl;
 
 Point2d princepal_point(325.1,249.7);
 double focal_length=521; 
 Mat essential_matrix;
 essential_matrix=findEssentialMat(points1,points2,K);
 cout<<"essential_matrix= \n"<<endl<<essential_matrix<<endl;
  
 recoverPose(essential_matrix,points1,points2,K,R,t);
 cout<<"Rotating matrix= \n"<<endl<<R<<endl;
 cout<<"translation matrix= \n"<<endl<<t<<endl;
}

Point2d pixel2cam(const Mat& K,const Point2d& p)
{
  return Point2d
  (
    (p.x-K.at<double>(0,2))/(K.at<double>(0,0)),
    (p.y-K.at<double>(1,2))/(K.at<double>(1,1))
  );    
}
