#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void findFeatureMatches(const Mat& img1,const Mat& img2,
		vector<KeyPoint>& kp1,vector<KeyPoint>& kp2,vector<DMatch>& matches);

void poseEstimation(const Mat& cameraMatrix, const vector<KeyPoint>& kp1,const vector<KeyPoint>& kp2,
	     const vector<DMatch>& matches,Mat& R,Mat& T);

Point2f pixel2cam(const KeyPoint& _p,const Mat& cameraMatrix);

void triangulation(const Mat& R, const Mat& T, const vector< KeyPoint >& kp1, const vector< KeyPoint >& kp2, const vector< DMatch >& matches, vector< Point3d >& P);

int main(int argc,char** argv)
{
  //读取原始图片
  Mat img1,img2;
  img1=imread(argv[1],1);
  img2=imread(argv[2],1);
  if (img1.empty()||img2.empty()){cerr<<"imread failed!"<<endl;return -1;}
  //找对应特征点
  vector<KeyPoint> kp1;
  vector<KeyPoint> kp2;
  vector<DMatch> matches;
  findFeatureMatches(img1,img2,kp1,kp2,matches);
  if(matches.empty()){cerr<<"matches are empty!"<<endl;return -1;}
  
  //显示匹配结果
  //Mat outImage;
  //drawMatches(img1,kp1,img2,kp2,matches,outImage);
  //imshow("outImage",outImage);
  //waitKey(0);
  
  //估计姿态
  Mat R,T;
  Mat cameraMatrix=(Mat_<double>(3,3) <<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
  poseEstimation(cameraMatrix,kp1,kp2,matches,R,T);
  //三角化
  vector<Point3d> P;
  triangulation(R,T,kp1,kp2,matches,P);
  
  //计算重投影误差
  //将三维点重新投影到cam1和cam2上，比较误差
  vector<Point2d> Points_cam1,Points_cam2,Points_cam1_3d,Points_cam2_3d;
    if(matches.size()==P.size())
    cout<<"continue ..."<<endl;
  else cerr<<"something wrong ..."<<endl; 
  for(int i=0;i<matches.size();i++){
	   Points_cam1.push_back(          pixel2cam(kp1[matches[i].queryIdx],cameraMatrix)            );
	   Points_cam2.push_back(           pixel2cam(kp2[matches[i].trainIdx],cameraMatrix)             );
           Points_cam1_3d.push_back(Point2f(      (P[i].x  /  P[i].z),
                                                                              (P[i].y /   P[i].z)   ));
	   Mat transPoint=R*(Mat_<double>(3,1)<< P[i].x,P[i].y,P[i].z )+T;
           Points_cam2_3d.push_back(Point2f(      (transPoint.at<double>(0,0)/transPoint.at<double>(2,0)),
	                                                                      (transPoint.at<double>(1,0)/transPoint.at<double>(2,0))  ));
           //transPoint/=transPoint.at<double>(2,0);
	   cout<<"Points in cam1 is "<<Points_cam1[i]<<"\n"
	   <<"Points in cam1 from 3d is "<<Points_cam1_3d[i]<<"\n"
	   <<"the depth of Points in cam1 is"<<(P[i].z)<<"\n"
	   <<"Points in cam2 is "<<Points_cam2[i]<<" \n"
	   <<"Points in cam2 from 3d is "<<Points_cam2_3d[i]<< endl;
	   
 
  }
  return 0;
}

void findFeatureMatches(const Mat& img1, const Mat& img2, vector< KeyPoint >& kp1, vector< KeyPoint >& kp2,
			vector< DMatch >& matches)
{
       //初始化
       Ptr<FeatureDetector> detector=ORB::create();
       Ptr<DescriptorExtractor> descriptor=ORB::create();
       Ptr<DescriptorMatcher> matcher=DescriptorMatcher::create("BruteForce-Hamming");
       detector->detect(img1,kp1);
       detector->detect(img2,kp2);
       Mat descriptor1,descriptor2;
       descriptor->compute(img1,kp1,descriptor1);
       descriptor->compute(img2,kp2,descriptor2);
       vector<DMatch> _matches;
       matcher->match(descriptor1,descriptor2,_matches);
       //筛选匹配
       double minDist=10000;
       double maxDist=0;
       for(int i=0;i<_matches.size();i++)
       {
	 double dist=_matches[i].distance;
	 if(dist<minDist){minDist=dist;}
	 if(dist>maxDist){maxDist=dist;}
      }
      for(int i=0;i<_matches.size();i++)
      {
	if(_matches[i].distance<max(2*minDist,30.0)){matches.push_back(_matches[i]);}
      }
}

void poseEstimation(const Mat& cameraMatrix,const vector< KeyPoint >& kp1, const vector< KeyPoint >& kp2, 
		    const vector< DMatch >& matches, Mat& R, Mat& T)
{
         vector<Point2d> points1,points2;
	 for(int i=0;i<matches.size();i++){
	   points1.push_back(kp1[matches[i].queryIdx].pt);
	   points2.push_back(kp2[matches[i].trainIdx].pt);
	 }
	 if(points1.empty()||points2.empty())
	 {cerr<<"vector<Point2f>.push_back faied!"<<endl;}
         Mat essentialMatrix;
         essentialMatrix=findEssentialMat(points1,points2,cameraMatrix);
	 recoverPose(essentialMatrix,points1,points2,cameraMatrix,R,T);
	 cout<<"Rotation Matrix is  \n"<<R<<endl;
	 cout<<"Translation Matrix is  \n"<<T<<endl;
}

Point2f pixel2cam(const KeyPoint& _p,const Mat& cameraMatrix)
{
      //(x/d)=(u=u0)/fu
      double x=(_p.pt.x-cameraMatrix.at<double>(0,2))/cameraMatrix.at<double>(0,0);  
      double y=(_p.pt.y-cameraMatrix.at<double>(1,2))/cameraMatrix.at<double>(1,1);
      return Point2f(x,y);
}

void triangulation(const Mat& R, const Mat& T, const vector< KeyPoint >& kp1, const vector< KeyPoint >& kp2, 
		   const vector< DMatch >& matches, vector<Point3d>& P)
{
      Mat T1=(Mat_<float>(3,4)<<    1,0,0,0,
	                                                 0,1,0,0,
	                                                 0,0,1,0);
      Mat T2=(Mat_<float>(3,4)<<    R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),T.at<double>(0,0),
	                                                 R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),T.at<double>(1,0),
	                                                 R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),T.at<double>(2,0));
      Mat cameraMatrix=(Mat_<double>(3,3) <<520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
      vector<Point2f> _points1,_points2;
      for(DMatch m:matches)
      {
	_points1.push_back(pixel2cam(kp1[m.queryIdx],cameraMatrix));
	_points2.push_back(pixel2cam(kp2[m.trainIdx],cameraMatrix));
      }
      Mat _P;
         triangulatePoints(T1,T2,_points1,_points2,_P);
	 for(int i=0;i<_P.cols;i++)
	 {Mat x=_P.col(i);
	  x /=x.at<float>(3,0);
	  P.push_back(Point3d(x.at<float>(0,0),
	                                       x.at<float>(1,0),
			                       x.at<float>(2.0)  ));
	 }
}
