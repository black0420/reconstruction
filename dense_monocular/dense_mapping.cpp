#include <iostream>
#include <vector>
#include <fstream>
using namespace std; 
#include <boost/timer.hpp>
#include <boost/concept_check.hpp>
#include <pangolin/pangolin.h>
#include <cstdio>
#include <cstdlib>
// for sophus 
#include <sophus/se3.h>
using Sophus::SE3;
#include <opencv2/core/eigen.hpp>
// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/eigen.hpp>
// #include <Eigen/Matrix>
#include <opencv2/opencv.hpp>

using namespace Eigen;
#include <highgui.h>
#include <cv.h>
#include <cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
using namespace cv;

/**********************************************
* 本程序实现单目相机在已知轨迹下的三维重建
* 
* 
* 
* 传入两张单目图像
* 
* 传入已知pose求出相对RT
* 
* 传入相机内参
* 
* 
* 
* 
* 
    需要rectify，然后用SGBM求出视差，三角化得到三维坐标。
***********************************************/
int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)  ;

void DrawMapPoints (
    vector<Vec3f>& points);

// ------------------------------------------------------------------
// 重要的函数 
// 读取数据 -单目图像和对应的pose 
bool readDatasetFiles( 
    const string& path, 
    vector<string>& color_image_files, 
    vector<SE3>& poses 
);
   static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
        static cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t);

int main( int argc, char** argv )
{
    
//     从数据集读取数据
    vector<string> color_image_files; 
    vector<SE3> poses_TWC;
    bool ret = readDatasetFiles( argv[1], color_image_files, poses_TWC );
    if ( ret==false )
    {
        cout<<"Reading image files failed!"<<endl;
        return -1; 
    }
    cout<<"read total "<<color_image_files.size()<<" files."<<endl;
    
//     第一张图
//     Mat ref = imread( color_image_files[0], CV_LOAD_IMAGE_UNCHANGED);                // CV_LOAD_IMAGE_COLOR
        Mat ref = imread( "/home/fubo/reconstruction/dataset/jiao/0055.png", CV_LOAD_IMAGE_UNCHANGED);                // CV_LOAD_IMAGE_COLOR
    
    //第二张图
            Mat curr = imread("/home/fubo/reconstruction/dataset/jiao/0056.png", CV_LOAD_IMAGE_UNCHANGED );      
     
        //获得图片和位姿之后 读内参
        FileStorage fs("/home/fubo/reconstruction/dataset/jiao/intrinsics.yml",CV_STORAGE_READ);

		Mat  _M1, _D1, _M2, _D2;
		fs["M1"] >> _M1;
		fs["D1"] >> _D1;
		fs["M2"] >> _M2;
		fs["D2"] >> _D2;
		
		 //读外参
  Matrix3d Rcw1; // Current Camera Rotation
			Vector3d tcw1; // Current Camera Translation
			  Rcw1 << 0.04802580, -0.96432000, -0.26034800, 0.99529700, 0.02424890, 0.09378320, -0.08412380, -0.26362800, 0.96094900;  
			  tcw1<<177.08100000, 187.89500000, 621.80700000;			  
    SE3 pose_ref_TWC = SE3(Rcw1,tcw1);
    
         Matrix3d Rcw2; // Current Camera Rotation
			Vector3d tcw2; // Current Camera Translation
			  Rcw2 << 0.04802580, -0.96432000, -0.26034800, 0.99529700, 0.02424890, 0.09378320, -0.08412380, -0.26362800, 0.96094900; 
			  tcw2<<77.10210000, 186.45800000, 620.24200000;
    SE3 pose_curr_TWC = SE3(Rcw2,tcw2);
    
    
           SE3 pose_T_C_R = pose_curr_TWC * pose_ref_TWC.inverse(); // 坐标转换关系： T_C_W * T_W_R = T_C_R		
				 Vector3d _T = pose_T_C_R.translation();
		 Matrix3d _R = pose_T_C_R.rotation_matrix();		
		Mat R,T;		
		    Mat R1;
		    Mat R2;
		    Mat P1;
                    Mat abb3;
		R=toCvMat(_R);
		T=toCvMat(_T);
	int rows_l=ref.rows;
	int cols_l=ref.cols;
	Rect roi1, roi2;
	  Mat Q;
	  cv::Mat r;
	  R.convertTo(r, CV_64F);
	  cv::Mat t;
	  T.convertTo(t, CV_64F);
cout <<r<<endl;
cout<<t<<endl;

cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, -t, R1, R2, P1, abb3, Q);
// 	  cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, t, R1, R2, P1, abb3, Q,CALIB_ZERO_DISPARITY, 1,  cv::Size(cols_l,rows_l), &roi1, &roi2);


    Mat map11;  
    Mat map12;  
    Mat map21;  
    Mat map22;  

		cv::initUndistortRectifyMap(_M1,_D1, R1, P1.rowRange(0,3).colRange(0,3), cv::Size(cols_l,rows_l),CV_32F,map11, map12);
		cv::initUndistortRectifyMap(_M2,_D2, R2, abb3.rowRange(0,3).colRange(0,3), cv::Size(cols_l,rows_l),CV_32F,map21, map22);
		
// 		Mat left;
// 		Mat right;
// // 
// 		  cv::remap(ref, left, map11, map12,cv::INTER_LINEAR);
// 		  cv::remap(curr, right, map21, map22,cv::INTER_LINEAR);

		   Mat img1 = imread("/home/fubo/reconstruction/dataset/jiao/0055.png"), left;
		    Mat img2 = imread("/home/fubo/reconstruction/dataset/jiao/0056.png"), right;

		Mat img( rows_l,cols_l * 2, CV_8UC3);//高度一样，宽度双倍
		imshow("rectified", img);
		
		remap(img1, left, map11, map12, cv::INTER_LINEAR);//左校正
    remap(img2, right, map21,map22, cv::INTER_LINEAR);//右校正

			      Mat imgPart1 = img( Rect(0, 0,cols_l,rows_l) );//浅拷贝
		Mat imgPart2 = img( Rect(cols_l, 0, cols_l,rows_l) );//浅拷贝
		resize(left, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
		resize(right, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);

    //画横线
    for( int i = 0; i < img.rows; i += 32 )
        line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);

    //显示行对准的图形
    Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
    resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
    imshow("rectified", smallImg);
		  
/*    
    Mat left = imread("/home/fubo/reconstruction/dataset/first",  IMREAD_GRAYSCALE );  
    Mat right = imread("/home/fubo/reconstruction/dataset/second",  IMREAD_GRAYSCALE );  */

//sgbm 算法
    Mat disp;  

    int mindisparity = -15;  
//     int ndisparities = 64;  
        int ndisparities = 144;    

//     int SADWindowSize = 11;   
        int SADWindowSize = 11;   

//     SGBM  
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);  
//     int P11 = 8 * left.channels() * SADWindowSize* SADWindowSize;  
        int P11 = 4 * left.channels() * SADWindowSize* SADWindowSize;  

    int P22 = 32 * left.channels() * SADWindowSize* SADWindowSize;  
    sgbm->setP1(P11);  
    sgbm->setP2(P22);  
  
//     sgbm->setPreFilterCap(15);  
        sgbm->setPreFilterCap(30);  

    sgbm->setUniquenessRatio(2);  
//         sgbm->setUniquenessRatio(15);  

        sgbm->setSpeckleRange(2);  

//     sgbm->setSpeckleWindowSize(100);  
        sgbm->setSpeckleWindowSize(10);  

    sgbm->setDisp12MaxDiff(1);  
    
//     sgbm->setMode(cv::StereoSGBM::MODE_HH);  
    sgbm->compute(left, right, disp);  
  
    disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值  
    
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示  
        Mat disp8U2 = Mat(disp.rows, disp.cols, CV_8UC1);       //显示  

    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);  
    getDisparityImage(disp8U,disp8U2,1);
    
//      imshow("left", left);  
//     imshow("right", right);  
   imshow("disparity", disp8U2);  
waitKey();
    imwrite("/home/fubo/reconstruction/SGBM.jpeg", disp8U2);  

//  Mat newMat;
//         cv::reprojectImageTo3D( disp8U, newMat, Q,true,-1 );
 
     const string& filename =  "reproject_pcd.txt";
     std::FILE* fp = std::fopen(filename.c_str(), "wt");

     
     cv::Mat_ <float> Q_=Q;
   cout<<"q"<<Q<<endl;
     
      const double max_z = 10000;
     cv::Mat_<cv::Vec3f> XYZ(disp.rows,disp.cols);   // Output point cloud
cv::Mat_<float> vec_tmp(4,1);
for(int y=0; y<disp.rows; ++y) {
    for(int x=0; x<disp.cols; ++x) {
        vec_tmp(0)=x; 
	vec_tmp(1)=y; 
	vec_tmp(2)=disp.at<float>(y,x); 
	vec_tmp(3)=1;
        vec_tmp = Q_*vec_tmp;
        vec_tmp /= vec_tmp(3);
        cv::Vec3f &point = XYZ.at<cv::Vec3f>(y,x);
        point[0] = vec_tmp(0);
        point[1] = vec_tmp(1);
        point[2] = vec_tmp(2);
	  if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
	   if(point[2]>300)continue;
	              fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
    }
}
     fclose(fp);
    return 0;
}

bool readDatasetFiles(
    const string& path, 
    vector< string >& color_image_files, 
    std::vector<SE3>& poses
)
{
    ifstream fin( path+"/first_200_frames_traj_over_table_input_sequence.txt");
    if ( !fin ) return false;
    
    while ( !fin.eof() )
    {
		// 数据格式：图像文件名 tx, ty, tz, qx, qy, qz, qw ，注意是 TWC 而非 TCW
        string image; 
        fin>>image; 
        double data[7];
        for ( double& d:data ) fin>>d;
        
        color_image_files.push_back( path+string("/images/")+image );
        poses.push_back(
            SE3( Quaterniond(data[6], data[3], data[4], data[5]), 
                 Vector3d(data[0], data[1], data[2]))
        );
        if ( !fin.good() ) break;
    }
    return true;
}


void DrawMapPoints (
 
    vector<Vec3f>& points
		      )
{
    pangolin::CreateWindowAndBind("单目三维重建",1024,768);
  // 3D Mouse handler requires depth testing to be enabled
    // 启动深度测试，OpenGL只绘制最前面的一层，绘制时检查当前像素前面是否有别的像素，如果别的像素挡住了它，那它就不会绘制
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    // 在OpenGL中使用颜色混合
    glEnable(GL_BLEND);
    // 选择混合选项
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      // 新建按钮和选择框，第一个参数为按钮的名字，第二个为默认状态，第三个为是否有选择框
    pangolin::CreatePanel("menu").SetBounds(0.0,1.0,0.0,pangolin::Attach::Pix(175));
    pangolin::Var<bool> menuFollowCamera("menu.Follow Camera",true,true);
    pangolin::Var<bool> menuShowPoints("menu.Show Points",true,true);
    pangolin::Var<bool> menuReset("menu.Reset",false,false);
  
       // Define Camera Render Object (for view / scene browsing)
    // 定义相机投影模型：ProjectionMatrix(w, h, fu, fv, u0, v0, zNear, zFar)
    // 定义观测方位向量：观测点位置：(mViewpointX mViewpointY mViewpointZ)
    //                观测目标位置：(0, 0, 0)
    //                观测的方位向量：(0.0,-1.0, 0.0)
    pangolin::OpenGlRenderState s_cam(
                pangolin::ProjectionMatrix(1024,768,500,500,512,389,0.1,1000),
                pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0)
                );
    
      // Add named OpenGL viewport to window and provide 3D Handler
    // 定义显示面板大小，orbslam中有左右两个面板，昨天显示一些按钮，右边显示图形
    // 前两个参数（0.0, 1.0）表明宽度和面板纵向宽度和窗口大小相同
    // 中间两个参数（pangolin::Attach::Pix(175), 1.0）表明右边所有部分用于显示图形
    // 最后一个参数（-1024.0f/768.0f）为显示长宽比
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f/768.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));
  
	        pangolin::OpenGlMatrix M;
    M.SetIdentity();
  while(1)
    {
        // 清除缓冲区中的当前可写的颜色缓冲 和 深度缓冲
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
//  Vector3d twc = T_C_R.translation();
//     Matrix3d Rwc = T_C_R.rotation_matrix() ; 
//        M.m[0] = Rwc(0,0);
//         M.m[1] =Rwc(1,0);
//         M.m[2] = Rwc(2,0);
//         M.m[3]  = 0.0;
// 
//         M.m[4] = Rwc(0,1);
//         M.m[5] =Rwc(1,1);
//         M.m[6] = Rwc(2,1);
//         M.m[7]  = 0.0;
// 
//         M.m[8] = Rwc(0,2);
//         M.m[9] = Rwc(1,2);
//         M.m[10] = Rwc(2,2);
//         M.m[11]  = 0.0;
// 
//         M.m[12] = twc[0];
//         M.m[13] = twc[1];
//         M.m[14] = twc[2];
//         M.m[15]  = 1.0;
// 
// 
//         // 步骤2：根据相机的位姿调整视角
//         // menuFollowCamera为按钮的状态，bFollow为真实的状态
//         if(menuFollowCamera )
//         {
//             s_cam.Follow(M);
//         }
//         else if(menuFollowCamera )
//         {
            s_cam.SetModelViewMatrix(pangolin::ModelViewLookAt(0,-0.7,-1.8, 0,0,0,0.0,-1.0, 0.0));
//             s_cam.Follow(M);
//         }
//      
// 
        d_cam.Activate(s_cam);
        // 步骤3：绘制地图和图像
	
	
        // 设置为白色，glClearColor(red, green, blue, alpha），数值范围(0, 1)
        glClearColor(1.0f,1.0f,1.0f,1.0f);
      glPointSize(1);
    glBegin(GL_POINTS);
    glColor3d(0.0,0.0,0.0);
 for ( int i=0; i<points.size(); i=i+50 )
    {Vec3f pointxyz=points[i];
         glVertex3d( pointxyz[0], pointxyz[1],pointxyz[2]);
          cout<< pointxyz[0]<< pointxyz[1]<< pointxyz[2]<<endl;

    }
        glEnd();

        pangolin::FinishFrame();

     
    
    }
  
  
}
cv::Mat toCvMat(const Eigen::Matrix3d &m)
{
    cv::Mat cvMat(3,3,CV_32F);
    for(int i=0;i<3;i++)
        for(int j=0; j<3; j++)
            cvMat.at<float>(i,j)=m(i,j);

    return cvMat.clone();
}

cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m)
{
    cv::Mat cvMat(3,1,CV_32F);
    for(int i=0;i<3;i++)
            cvMat.at<float>(i)=m(i);

    return cvMat.clone();
}
 cv::Mat toCvSE3(const Eigen::Matrix<double, 3, 3> &R, const Eigen::Matrix<double, 3, 1> &t) {
        cv::Mat cvMat = cv::Mat::eye(4, 4, CV_32F);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                cvMat.at<float>(i, j) = R(i, j);
            }
        }
        for (int i = 0; i < 3; i++) {
            cvMat.at<float>(i, 3) = t(i);
        }

        return cvMat.clone();
    }

int getDisparityImage(cv::Mat& disparity, cv::Mat& disparityImage, bool isColor)  
{  
    // 将原始视差数据的位深转换为 8 位  
    cv::Mat disp8u;  
    if (disparity.depth() != CV_8U)  
    {  
        disparity.convertTo(disp8u, CV_8U, 255/(64*16.));  
    }   
    else  
    {  
        disp8u = disparity;  
    }  
  
  
    // 转换为伪彩色图像 或 灰度图像  
    if (isColor)  
    {  
        if (disparityImage.empty() || disparityImage.type() != CV_8UC3 )  
        {  
            disparityImage = cv::Mat::zeros(disparity.rows, disparity.cols, CV_8UC3);  
        }  
  
  
        for (int y=0;y<disparity.rows;y++)  
        {  
            for (int x=0;x<disparity.cols;x++)  
            {  
                uchar val = disp8u.at<uchar>(y,x);  
                uchar r,g,b;  
  
  
                if (val==0)   
                    r = g = b = 0;  
                else  
                {  
                    r = 255-val;  
                    g = val < 128 ? val*2 : (uchar)((255 - val)*2);  
                    b = val;  
                }  
                disparityImage.at<cv::Vec3b>(y,x) = cv::Vec3b(r,g,b);  
            }  
        }  
    }   
    else  
    {  
        disp8u.copyTo(disparityImage);  
    }  
  
    return 1;  
}  