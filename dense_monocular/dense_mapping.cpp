#include <iostream>
#include <vector>
#include <fstream>
using namespace std; 
#include <boost/timer.hpp>
#include <pangolin/pangolin.h>
#include <cstdio>
#include <cstdlib>
// for sophus 
#include <sophus/se3.h>
using Sophus::SE3;

// for eigen 
#include <Eigen/Core>
#include <Eigen/Geometry>
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
***********************************************/



// ------------------------------------------------------------------
// parameters 
const int boarder = 20; 	// 边缘宽度
const int width = 640;  	// 宽度 
const int height = 480;  	// 高度
const double fx = 481.2f;	// 相机内参
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 2;	// NCC 取的窗口半宽度
const int ncc_area = (2*ncc_window_size+1)*(2*ncc_window_size+1); // NCC窗口面积
const double min_cov = 0.1;	// 收敛判定：最小方差
const double max_cov = 10;	// 发散判定：最大方差


    static cv::Mat toCvMat(const SE3 &SE3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);

void DrawMapPoints (
    vector<Vec3f>& points);

void find_feature_matches (
    const Mat& img_1, const Mat& img_2,
    std::vector<KeyPoint>& keypoints_1,
    std::vector<KeyPoint>& keypoints_2,
    std::vector< DMatch >& matches );

void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t );

void triangulation (
    const vector<KeyPoint>& keypoint_1,
    const vector<KeyPoint>& keypoint_2,
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t,
    vector<Point3d>& points
);

// 像素坐标转相机归一化坐标
Point2f pixel2cam( const Point2d& p, const Mat& K );
// ------------------------------------------------------------------
// 重要的函数 
// 读取数据 -单目图像和对应的pose 
bool readDatasetFiles( 
    const string& path, 
    vector<string>& color_image_files, 
    vector<SE3>& poses 
);

// 根据新的图像更新深度估计
bool update( 
    const Mat& ref, 
    const Mat& curr, 
    const SE3& T_C_R, 
    Mat& depth, 
    Mat& depth_cov 
);

// 极线搜索 
bool epipolarSearch( 
    const Mat& ref, 
    const Mat& curr, 
    const SE3& T_C_R, 
    const Vector2d& pt_ref, 
    const double& depth_mu, 
    const double& depth_cov,
    Vector2d& pt_curr
);

// 更新深度滤波器 
bool updateDepthFilter( 
    const Vector2d& pt_ref, 
    const Vector2d& pt_curr, 
    const SE3& T_C_R, 
    Mat& depth, 
    Mat& depth_cov
);

// 计算 NCC 评分 
double NCC( const Mat& ref, const Mat& curr, const Vector2d& pt_ref, const Vector2d& pt_curr );

// 双线性灰度插值 
inline double getBilinearInterpolatedValue( const Mat& img, const Vector2d& pt ) {
    uchar* d = & img.data[ int(pt(1,0))*img.step+int(pt(0,0)) ];
    double xx = pt(0,0) - floor(pt(0,0)); 
    double yy = pt(1,0) - floor(pt(1,0));
    return  (( 1-xx ) * ( 1-yy ) * double(d[0]) +
            xx* ( 1-yy ) * double(d[1]) +
            ( 1-xx ) *yy* double(d[img.step]) +
            xx*yy*double(d[img.step+1]))/255.0;
}

// ------------------------------------------------------------------
// 一些小工具 
// 显示估计的深度图 
bool plotDepth( const Mat& depth );

// 像素到相机坐标系 
inline Vector3d px2cam ( const Vector2d px ) {
    return Vector3d ( 
        (px(0,0) - cx)/fx,
        (px(1,0) - cy)/fy, 
        1
    );
}

// 相机坐标系到像素 
inline Vector2d cam2px ( const Vector3d p_cam ) {
    return Vector2d (
        p_cam(0,0)*fx/p_cam(2,0) + cx, 
        p_cam(1,0)*fy/p_cam(2,0) + cy 
    );
}

// 检测一个点是否在图像边框内
inline bool inside( const Vector2d& pt ) {
    return pt(0,0) >= boarder && pt(1,0)>=boarder 
        && pt(0,0)+boarder<width && pt(1,0)+boarder<=height;
}

// 显示极线匹配 
void showEpipolarMatch( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr );

// 显示极线 
void showEpipolarLine( const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr );
// ------------------------------------------------------------------


int main(  )
{
    /*
    // 从数据集读取数据
    vector<string> color_image_files; 
    vector<SE3> poses_TWC;
    bool ret = readDatasetFiles( argv[1], color_image_files, poses_TWC );
    if ( ret==false )
    {
        cout<<"Reading image files failed!"<<endl;
        return -1; 
    }
    cout<<"read total "<<color_image_files.size()<<" files."<<endl;
    
    // 第一张图
    Mat ref = imread( color_image_files[0], 1 );                // CV_LOAD_IMAGE_COLOR
    SE3 pose_ref_TWC = poses_TWC[0];
    
    
    for ( int index=1; index<color_image_files.size(); index++ )
    {
//         cout<<"*** loop "<<index<<" ***"<<endl;
        Mat curr = imread( color_image_files[index], 1 );       
        if (curr.data == nullptr) continue;
        SE3 pose_curr_TWC = poses_TWC[index];
        SE3 pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; // 坐标转换关系： T_C_W * T_W_R = T_C_R
        
        
        
        vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches ( ref, curr, keypoints_1, keypoints_2, matches );
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;

    //-- 估计两张图像间运动
    Mat R,t;
    pose_estimation_2d2d ( keypoints_1, keypoints_2, matches, R, t );

    //-- 三角化
    vector<Point3d> points;
    triangulation( keypoints_1, keypoints_2, matches, R, t, points );
    
    DrawMapPoints(pose_curr_TWC,points,matches);
      Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    for ( int i=0; i<matches.size(); i++ )
    {
        Point2d pt1_cam = pixel2cam( keypoints_1[ matches[i].queryIdx ].pt, K );
        Point2d pt1_cam_3d(
            points[i].x/points[i].z, 
            points[i].y/points[i].z 
        );
        
        cout<<"point in the first camera frame: "<<pt1_cam<<endl;
        cout<<"point projected from 3D "<<pt1_cam_3d<<", d="<<points[i].z<<endl;
        
        // 第二个图
        Point2f pt2_cam = pixel2cam( keypoints_2[ matches[i].trainIdx ].pt, K );
        Mat pt2_trans = R*( Mat_<double>(3,1) << points[i].x, points[i].y, points[i].z ) + t;
        pt2_trans /= pt2_trans.at<double>(2,0);
        cout<<"point in the second camera frame: "<<pt2_cam<<endl;
        cout<<"point reprojected from second frame: "<<pt2_trans.t()<<endl;
        cout<<endl;
    }
    
    }*/

    Mat left = imread("/home/fubo/reconstruction/first",  IMREAD_GRAYSCALE );  
    Mat right = imread("/home/fubo/reconstruction/second",  IMREAD_GRAYSCALE );  
    Mat disp;  
 
 

//     Mat R1, R2, P1, P2, Q;
// Rect roi1, roi2;
// stereoRectify(M1, D1, M2, D2, imageSize, R, T, R1, R2, P1, P2, Q,
// CALIB_ZERO_DISPARITY, 0, imageSize, &roi1, &roi2);
//     stereoRectify
//     
//     cv::reprojectImageTo3D(displf, img3d, Q, true);  
    
    int mindisparity = 0;  
    int ndisparities = 64;    
    int SADWindowSize = 11;   
  
//     SGBM  
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(mindisparity, ndisparities, SADWindowSize);  
    int P1 = 8 * left.channels() * SADWindowSize* SADWindowSize;  
    int P2 = 32 * left.channels() * SADWindowSize* SADWindowSize;  
    sgbm->setP1(P1);  
    sgbm->setP2(P2);  
  
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
//      imshow("left", left);  
//     imshow("right", right);  
//    imshow("disparity", disp8U);  
// waitKey();
    imwrite("/home/fubo/reconstruction/SGBM.jpeg", disp8U);  
  


// double _Q[] = { 1., 0., 0., -3.2554532241821289e+002, 0., 1., 0., -2.4126087760925293e+002, 0., 0., 0., 4.2440051858596559e+002, 0., 0., -2.9937622423123672e-001, 0. }; 

Mat Q = (Mat_<double>(4, 4) << 1., 0., 0., -3.2554532241821289e+002, 0., 1., 0., -2.4126087760925293e+002, 0., 0., 0., 4.2440051858596559e+002, 0., 0., -2.9937622423123672e-001, 0. ); 

// CvMat Q=cvMat(4, 4, CV_64FC1, _Q);  




 Mat_<double> _3dImg(disp8U.size());
 Mat newMat;

        CvMat cvdisp = disp8U; 
	CvMat cv_3dImg = _3dImg; 
	CvMat cvQ = Q;
        cv::reprojectImageTo3D( disp8U, newMat, Q,true,-1 );

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
	              fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);

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

// 对整个深度图进行更新
bool update(const Mat& ref, const Mat& curr, const SE3& T_C_R, Mat& depth, Mat& depth_cov )
{
#pragma omp parallel for
    for ( int x=boarder; x<width-boarder; x++ )
#pragma omp parallel for
        for ( int y=boarder; y<height-boarder; y++ )
        {
			// 遍历每个像素
            if ( depth_cov.ptr<double>(y)[x] < min_cov || depth_cov.ptr<double>(y)[x] > max_cov ) // 深度已收敛或发散
                continue;
            // 在极线上搜索 (x,y) 的匹配 
            Vector2d pt_curr; 
            bool ret = epipolarSearch ( 
                ref, 
                curr, 
                T_C_R, 
                Vector2d(x,y), 
                depth.ptr<double>(y)[x], 
                sqrt(depth_cov.ptr<double>(y)[x]),
                pt_curr
            );
            
            if ( ret == false ) // 匹配失败
                continue; 
            
			// 取消该注释以显示匹配
            // showEpipolarMatch( ref, curr, Vector2d(x,y), pt_curr );
            
            // 匹配成功，更新深度图 
            updateDepthFilter( Vector2d(x,y), pt_curr, T_C_R, depth, depth_cov );
        }
}

// 极线搜索
// 方法见书 13.2 13.3 两节
bool epipolarSearch(
    const Mat& ref, const Mat& curr, 
    const SE3& T_C_R, const Vector2d& pt_ref, 
    const double& depth_mu, const double& depth_cov, 
    Vector2d& pt_curr )
{
    Vector3d f_ref = px2cam( pt_ref );
    f_ref.normalize();
    Vector3d P_ref = f_ref*depth_mu;	// 参考帧的 P 向量
    
    Vector2d px_mean_curr = cam2px( T_C_R*P_ref ); // 按深度均值投影的像素
    double d_min = depth_mu-3*depth_cov, d_max = depth_mu+3*depth_cov;
    if ( d_min<0.1 ) d_min = 0.1;
    Vector2d px_min_curr = cam2px( T_C_R*(f_ref*d_min) );	// 按最小深度投影的像素
    Vector2d px_max_curr = cam2px( T_C_R*(f_ref*d_max) );	// 按最大深度投影的像素
    
    Vector2d epipolar_line = px_max_curr - px_min_curr;	// 极线（线段形式）
    Vector2d epipolar_direction = epipolar_line;		// 极线方向 
    epipolar_direction.normalize();
    double half_length = 0.5*epipolar_line.norm();	// 极线线段的半长度
    if ( half_length>100 ) half_length = 100;   // 我们不希望搜索太多东西 
    
	// 取消此句注释以显示极线（线段）
    // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );
    
    // 在极线上搜索，以深度均值点为中心，左右各取半长度
    double best_ncc = -1.0;
    Vector2d best_px_curr; 
    for ( double l=-half_length; l<=half_length; l+=0.7 )  // l+=sqrt(2) 
    {
        Vector2d px_curr = px_mean_curr + l*epipolar_direction;  // 待匹配点
        if ( !inside(px_curr) )
            continue; 
        // 计算待匹配点与参考帧的 NCC
        double ncc = NCC( ref, curr, pt_ref, px_curr );
        if ( ncc>best_ncc )
        {
            best_ncc = ncc; 
            best_px_curr = px_curr;
        }
    }
    if ( best_ncc < 0.85f )      // 只相信 NCC 很高的匹配
        return false; 
    pt_curr = best_px_curr;
    return true;
}

double NCC (
    const Mat& ref, const Mat& curr, 
    const Vector2d& pt_ref, const Vector2d& pt_curr
)
{
    // 零均值-归一化互相关
    // 先算均值
    double mean_ref = 0, mean_curr = 0;
    vector<double> values_ref, values_curr; // 参考帧和当前帧的均值
    for ( int x=-ncc_window_size; x<=ncc_window_size; x++ )
        for ( int y=-ncc_window_size; y<=ncc_window_size; y++ )
        {
            double value_ref = double(ref.ptr<uchar>( int(y+pt_ref(1,0)) )[ int(x+pt_ref(0,0)) ])/255.0;
            mean_ref += value_ref;
            
            double value_curr = getBilinearInterpolatedValue( curr, pt_curr+Vector2d(x,y) );
            mean_curr += value_curr;
            
            values_ref.push_back(value_ref);
            values_curr.push_back(value_curr);
        }
        
    mean_ref /= ncc_area;
    mean_curr /= ncc_area;
    
	// 计算 Zero mean NCC
    double numerator = 0, demoniator1 = 0, demoniator2 = 0;
    for ( int i=0; i<values_ref.size(); i++ )
    {
        double n = (values_ref[i]-mean_ref) * (values_curr[i]-mean_curr);
        numerator += n;
        demoniator1 += (values_ref[i]-mean_ref)*(values_ref[i]-mean_ref);
        demoniator2 += (values_curr[i]-mean_curr)*(values_curr[i]-mean_curr);
    }
    return numerator / sqrt( demoniator1*demoniator2+1e-10 );   // 防止分母出现零
}

bool updateDepthFilter(
    const Vector2d& pt_ref, 
    const Vector2d& pt_curr, 
    const SE3& T_C_R,
    Mat& depth, 
    Mat& depth_cov
)
{
    // 我是一只喵
    // 不知道这段还有没有人看
    // 用三角化计算深度
    SE3 T_R_C = T_C_R.inverse();
    Vector3d f_ref = px2cam( pt_ref );
    f_ref.normalize();
    Vector3d f_curr = px2cam( pt_curr );
    f_curr.normalize();
    
    // 方程
    // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
    // => [ f_ref^T f_ref, -f_ref^T f_cur ] [d_ref] = [f_ref^T t]
    //    [ f_cur^T f_ref, -f_cur^T f_cur ] [d_cur] = [f_cur^T t]
    // 二阶方程用克莱默法则求解并解之
    Vector3d t = T_R_C.translation();
    Vector3d f2 = T_R_C.rotation_matrix() * f_curr; 
    Vector2d b = Vector2d ( t.dot ( f_ref ), t.dot ( f2 ) );
    double A[4];
    A[0] = f_ref.dot ( f_ref );
    A[2] = f_ref.dot ( f2 );
    A[1] = -A[2];
    A[3] = - f2.dot ( f2 );
    double d = A[0]*A[3]-A[1]*A[2];
    Vector2d lambdavec = 
        Vector2d (  A[3] * b ( 0,0 ) - A[1] * b ( 1,0 ),
                    -A[2] * b ( 0,0 ) + A[0] * b ( 1,0 )) /d;
    Vector3d xm = lambdavec ( 0,0 ) * f_ref;
    Vector3d xn = t + lambdavec ( 1,0 ) * f2;
    Vector3d d_esti = ( xm+xn ) / 2.0;  // 三角化算得的深度向量
    double depth_estimation = d_esti.norm();   // 深度值
    
    // 计算不确定性（以一个像素为误差）
    Vector3d p = f_ref*depth_estimation;
    Vector3d a = p - t; 
    double t_norm = t.norm();
    double a_norm = a.norm();
    double alpha = acos( f_ref.dot(t)/t_norm );
    double beta = acos( -a.dot(t)/(a_norm*t_norm));
    double beta_prime = beta + atan(1/fx);
    double gamma = M_PI - alpha - beta_prime;
    double p_prime = t_norm * sin(beta_prime) / sin(gamma);
    double d_cov = p_prime - depth_estimation; 
    double d_cov2 = d_cov*d_cov;
    
    // 高斯融合
    double mu = depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    double sigma2 = depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ];
    
    double mu_fuse = (d_cov2*mu+sigma2*depth_estimation) / ( sigma2+d_cov2);
    double sigma_fuse2 = ( sigma2 * d_cov2 ) / ( sigma2 + d_cov2 );
    
    depth.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = mu_fuse; 
    depth_cov.ptr<double>( int(pt_ref(1,0)) )[ int(pt_ref(0,0)) ] = sigma_fuse2;
    
    return true;
}

// 后面这些太简单我就不注释了（其实是因为懒）
bool plotDepth(const Mat& depth)
{
    imshow( "depth", depth*0.4 );
    waitKey(1);
}

void showEpipolarMatch(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_curr)
{
    Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );
    
    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,0,250), 2);
    cv::circle( curr_show, cv::Point2f(px_curr(0,0), px_curr(1,0)), 5, cv::Scalar(0,0,250), 2);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(1);
}

void showEpipolarLine(const Mat& ref, const Mat& curr, const Vector2d& px_ref, const Vector2d& px_min_curr, const Vector2d& px_max_curr)
{

    Mat ref_show, curr_show;
    cv::cvtColor( ref, ref_show, CV_GRAY2BGR );
    cv::cvtColor( curr, curr_show, CV_GRAY2BGR );
    
    cv::circle( ref_show, cv::Point2f(px_ref(0,0), px_ref(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_min_curr(0,0), px_min_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::circle( curr_show, cv::Point2f(px_max_curr(0,0), px_max_curr(1,0)), 5, cv::Scalar(0,255,0), 2);
    cv::line( curr_show, Point2f(px_min_curr(0,0), px_min_curr(1,0)), Point2f(px_max_curr(0,0), px_max_curr(1,0)), Scalar(0,255,0), 1);
    
    imshow("ref", ref_show );
    imshow("curr", curr_show );
    waitKey(1);
}
void find_feature_matches ( const Mat& img_1, const Mat& img_2,
                            std::vector<KeyPoint>& keypoints_1,
                            std::vector<KeyPoint>& keypoints_2,
                            std::vector< DMatch >& matches )
{
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3 
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2 
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher  = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( match[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            matches.push_back ( match[i] );
        }
    }
}

void pose_estimation_2d2d (
    const std::vector<KeyPoint>& keypoints_1,
    const std::vector<KeyPoint>& keypoints_2,
    const std::vector< DMatch >& matches,
    Mat& R, Mat& t )
{
    // 相机内参,TUM Freiburg2
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for ( int i = 0; i < ( int ) matches.size(); i++ )
    {
        points1.push_back ( keypoints_1[matches[i].queryIdx].pt );
        points2.push_back ( keypoints_2[matches[i].trainIdx].pt );
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat ( points1, points2, CV_FM_8POINT );
    cout<<"fundamental_matrix is "<<endl<< fundamental_matrix<<endl;

    //-- 计算本质矩阵
    Point2d principal_point ( 325.1, 249.7 );				//相机主点, TUM dataset标定值
    int focal_length = 521;						//相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat ( points1, points2, focal_length, principal_point );
    cout<<"essential_matrix is "<<endl<< essential_matrix<<endl;

    //-- 计算单应矩阵
    Mat homography_matrix;
    homography_matrix = findHomography ( points1, points2, RANSAC, 3 );
    cout<<"homography_matrix is "<<endl<<homography_matrix<<endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    recoverPose ( essential_matrix, points1, points2, R, t, focal_length, principal_point );
    cout<<"R is "<<endl<<R<<endl;
    cout<<"t is "<<endl<<t<<endl;
}

void triangulation ( 
    const vector< KeyPoint >& keypoint_1, 
    const vector< KeyPoint >& keypoint_2, 
    const std::vector< DMatch >& matches,
    const Mat& R, const Mat& t, 
    vector< Point3d >& points )
{
    Mat T1 = (Mat_<float> (3,4) <<
        1,0,0,0,
        0,1,0,0,
        0,0,1,0);
    Mat T2 = (Mat_<float> (3,4) <<
        R.at<double>(0,0), R.at<double>(0,1), R.at<double>(0,2), t.at<double>(0,0),
        R.at<double>(1,0), R.at<double>(1,1), R.at<double>(1,2), t.at<double>(1,0),
        R.at<double>(2,0), R.at<double>(2,1), R.at<double>(2,2), t.at<double>(2,0)
    );
    
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point2f> pts_1, pts_2;
    for ( DMatch m:matches )
    {
        // 将像素坐标转换至相机坐标
        pts_1.push_back ( pixel2cam( keypoint_1[m.queryIdx].pt, K) );
        pts_2.push_back ( pixel2cam( keypoint_2[m.trainIdx].pt, K) );
    }
    
    Mat pts_4d;
    cv::triangulatePoints( T1, T2, pts_1, pts_2, pts_4d );
    
    // 转换成非齐次坐标
    for ( int i=0; i<pts_4d.cols; i++ )
    {
        Mat x = pts_4d.col(i);
        x /= x.at<float>(3,0); // 归一化
        Point3d p (
            x.at<float>(0,0), 
            x.at<float>(1,0), 
            x.at<float>(2,0) 
        );
        points.push_back( p );
    }
}

Point2f pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2f
    (
        ( p.x - K.at<double>(0,2) ) / K.at<double>(0,0), 
        ( p.y - K.at<double>(1,2) ) / K.at<double>(1,1) 
    );
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























