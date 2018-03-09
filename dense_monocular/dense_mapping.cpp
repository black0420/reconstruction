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
    Mat ref = imread( color_image_files[0], CV_LOAD_IMAGE_UNCHANGED);                // CV_LOAD_IMAGE_COLOR
    SE3 pose_ref_TWC = poses_TWC[0];
    
    //第二张图
        Mat curr = imread( color_image_files[57], CV_LOAD_IMAGE_UNCHANGED );      
        SE3 pose_curr_TWC = poses_TWC[57];
	
	
//         SE3 pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // 坐标转换关系： T_C_W * T_W_R = T_C_R
	        SE3 pose_T_C_R = pose_ref_TWC.inverse() * pose_curr_TWC; // 坐标转换关系： T_C_W * T_W_R = T_C_R

        
        //获得图片和位姿之后 读内参
        FileStorage fs("/home/fubo/reconstruction/dataset/intrinsics.yml",CV_STORAGE_READ);

		Mat  _M1, _D1, _M2, _D2;
		fs["M1"] >> _M1;
		fs["D1"] >> _D1;
		fs["M2"] >> _M2;
		fs["D2"] >> _D2;
		
// 		cout<<_D1<<endl;
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
cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, t, R1, R2, P1, abb3, Q);
// 	  cv::stereoRectify( _M1, _D1,_M2, _D2, cv::Size(cols_l,rows_l), r, t, R1, R2, P1, abb3, Q,CALIB_ZERO_DISPARITY, 0,  cv::Size(cols_l,rows_l), &roi1, &roi2);


    Mat map11;  
    Mat map12;  
    Mat map21;  
    Mat map22;  

		cv::initUndistortRectifyMap(_M1,_D1, R1, P1.rowRange(0,3).colRange(0,3), cv::Size(cols_l,rows_l),CV_32F,map11, map12);
		cv::initUndistortRectifyMap(_M2,_D2, R2, abb3.rowRange(0,3).colRange(0,3), cv::Size(cols_l,rows_l),CV_32F,map21, map22);
		
		Mat left;
		Mat right;
		
		  cv::remap(ref, left, map11, map12,cv::INTER_LINEAR);
		  cv::remap(curr, right, map21, map22,cv::INTER_LINEAR);
		  
// 		   Mat img1 = imread(color_image_files[0]), img1r;
// 		    Mat img2 = imread(color_image_files[55]), img2r;
// 
// 		Mat img( rows_l,cols_l * 2, CV_8UC3);//高度一样，宽度双倍
// 		imshow("rectified", img);
// 		
// 		remap(img1, img1r, map11, map12, cv::INTER_LINEAR);//左校正
//     remap(img2, img2r, map21,map22, cv::INTER_LINEAR);//右校正
//     
//     
//     
// 			      Mat imgPart1 = img( Rect(0, 0,cols_l,rows_l) );//浅拷贝
// 		Mat imgPart2 = img( Rect(cols_l, 0, cols_l,rows_l) );//浅拷贝
// 		resize(img1r, imgPart1, imgPart1.size(), 0, 0, CV_INTER_AREA);
// 		resize(img2r, imgPart2, imgPart2.size(), 0, 0, CV_INTER_AREA);
// 
//     //画横线
//     for( int i = 0; i < img.rows; i += 32 )
//         line(img, Point(0, i), Point(img.cols, i), Scalar(0, 255, 0), 1, 8);
// 
//     //显示行对准的图形
//     Mat smallImg;//由于我的分辨率1:1显示太大，所以缩小显示
//     resize(img, smallImg, Size(), 0.8, 0.8, CV_INTER_AREA);
//     imshow("rectified", smallImg);
		  
/*    
    Mat left = imread("/home/fubo/reconstruction/dataset/first",  IMREAD_GRAYSCALE );  
    Mat right = imread("/home/fubo/reconstruction/dataset/second",  IMREAD_GRAYSCALE );  */
    Mat disp;  
 
    
    int mindisparity = 0;  
    int ndisparities = 64;    
    int SADWindowSize = 11;   
  
//     SGBM  
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);  
    int P11 = 8 * left.channels() * SADWindowSize* SADWindowSize;  
    int P22 = 32 * left.channels() * SADWindowSize* SADWindowSize;  
    sgbm->setP1(P11);  
    sgbm->setP2(P22);  
  
    sgbm->setPreFilterCap(15);  
    sgbm->setUniquenessRatio(10);  
    sgbm->setSpeckleRange(2);  
    sgbm->setSpeckleWindowSize(100);  
    sgbm->setDisp12MaxDiff(1);  
//     sgbm->setMode(cv::StereoSGBM::MODE_HH);  
  
    sgbm->compute(left, right, disp);  
  
    disp.convertTo(disp, CV_32F, 1.0 / 16);                //除以16得到真实视差值  
    Mat disp8U = Mat(disp.rows, disp.cols, CV_8UC1);       //显示  
    normalize(disp, disp8U, 0, 255, NORM_MINMAX, CV_8UC1);  
     imshow("left", left);  
    imshow("right", right);  
   imshow("disparity", disp8U);  
waitKey();
    imwrite("/home/fubo/reconstruction/SGBM.jpeg", disp8U);  
  

 Mat newMat;
        cv::reprojectImageTo3D( disp, newMat, Q,true,-1 );

    Point3d  Point;
    vector<Vec3f> pointArray;
int y;

 const double max_z = 1.0e4;
 
      const string& filename =  "reproject_pcd.txt";
     std::FILE* fp = std::fopen(filename.c_str(), "wt");

 
     for(int y = 0; y < newMat.rows; y++){
        for(int x = 0; x < newMat.cols; x++){
           Vec3f point = newMat.at<Vec3f>(y, x);
           if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
	   if(point[2]>300)continue;
	              fprintf(fp, "%f %f %f\n", point[0], -point[1], point[2]);

//         cout<< point[0]<< point[1]<< point[2]<<endl;
	pointArray.push_back(point);

        }
     }
             cout<<pointArray.size()<<endl;

//         DrawMapPoints(pointArray);

    
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