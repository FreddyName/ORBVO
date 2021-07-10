// The member function `addFrame` of class VisualOdometry is defined here.

#include "my_slam/vo/vo.h"

namespace my_slam
{
namespace vo
{

void VisualOdometry::addFrame(Frame::Ptr frame)
{
    // Settings
    pushFrameToBuff_(frame);   //把新输入的帧放入buffer中

    // Renamed vars
    curr_ = frame;
    const int img_id = curr_->id_;    //得到当前帧的id
    const cv::Mat &K = curr_->camera_->K_;   //得到相机内参K

    // Start
    printf("\n\n=============================================\n");
    printf("Start processing the %dth image.\n", img_id);

    curr_->calcKeyPoints(); //计算关键点
    curr_->calcDescriptors();  //计算描述子
    cout << "Number of keypoints: " << curr_->keypoints_.size() << endl;
    prev_ref_ = ref_;   // 关键帧转移为上一关键帧，因为下面会更新新的参考帧

    // vo_state_: BLANK -> DOING_INITIALIZATION
    // 下面就根据4个状态进行了
    // 刚开始---初始化---追踪---lost
    if (vo_state_ == BLANK) // 如果是第一帧，那么就把位姿初始化单位矩阵，并且把此帧当做关键帧
    {
        curr_->T_w_c_ = cv::Mat::eye(4, 4, CV_64F);
        vo_state_ = DOING_INITIALIZATION;
        addKeyFrame_(curr_); // curr_ becomes the ref_
    }
    else if (vo_state_ == DOING_INITIALIZATION)
    {
        // Match features
        static const float max_matching_pixel_dist_in_initialization =
            basics::Config::get<float>("max_matching_pixel_dist_in_initialization");
        static const int method_index = basics::Config::get<float>("feature_match_method_index_initialization");
        // 这里进行当前帧和参考帧的匹配，结果保存在当前帧的matches_with_ref_中
        geometry::matchFeatures(
            ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_, method_index,
            false,
            ref_->keypoints_, curr_->keypoints_,
            max_matching_pixel_dist_in_initialization);

        printf("Number of matches with the 1st frame: %d\n", (int)curr_->matches_with_ref_.size());

        // Estimae motion and triangulate points
        // 然后在这里进行位姿估计和三角化
        estimateMotionAnd3DPoints_();
        printf("Number of inlier matches: %d\n", (int)curr_->inliers_matches_for_3d_.size());

        // Check initialization condition:
        printf("\nCheck VO init conditions: \n");
        // 判断是否初始化成功
        if (isVoGoodToInit_())
        {
            cout << "Large movement detected at frame " << img_id << ". Start initialization" << endl;
            pushCurrPointsToMap_();  // 初始化得到的3D点要放入地图
            addKeyFrame_(curr_);  // 初始化成功的这一帧也要做为关键帧
            vo_state_ = DOING_TRACKING;
            cout << "Inilialiation success !!!" << endl;
            cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
        }
        else // skip this frame
        {
            // 如果初始化不成功，认为刚才得到的位姿是不准确的，所以使用参考帧的位姿当做这一帧的位姿
            curr_->T_w_c_ = ref_->T_w_c_;
            cout << "Not initialize VO ..." << endl;
        }
    }
    else if (vo_state_ == DOING_TRACKING)
    {
        printf("\nDoing tracking\n");
        // 初始化的时候为什么不给当前帧赋初始值？？？
        curr_->T_w_c_ = ref_->T_w_c_.clone(); // Initial estimation of the current pose

        // 这里估计位姿不就使用PnP就可以了
        // PnP是3D地图点和当前帧的特征点进行匹配的
        bool is_pnp_good = poseEstimationPnP_();

        // 如果PnP失败
        if (!is_pnp_good) // pnp failed. Print log.
        {
            int num_matches = curr_->matches_with_map_.size();
            constexpr int kMinPtsForPnP = 5;
            printf("PnP failed.\n");
            printf("    Num inlier matches: %d.\n", num_matches);
            if (num_matches >= kMinPtsForPnP)
            {
                printf("    Computed world to camera transformation:\n");
                std::cout << curr_->T_w_c_ << std::endl;
            }
            // 啥逻辑？？？？
            printf("PnP result has been reset as R=identity, t=zero.\n");
        }
        else // pnp good 
        {
            // 这里优化的是frame_buf_中的固定数量的帧（后几帧），也可以选择优化单帧
            callBundleAdjustment_();

            // -- Insert a keyframe is motion is large. Then, triangulate more points
            // 通过比较当前帧和参考帧来决定是否是关键帧
            // 只有在是关键帧的时候才会和参考帧进行再一次的特征匹配，三角化来生成新的地图点
            if (checkLargeMoveForAddKeyFrame_(curr_, ref_))
            {
                // Feature matching
                static const float max_matching_pixel_dist_in_triangulation =
                    basics::Config::get<float>("max_matching_pixel_dist_in_triangulation");
                static const int method_index = basics::Config::get<float>("feature_match_method_index_pnp");
                geometry::matchFeatures(
                    ref_->descriptors_, curr_->descriptors_, curr_->matches_with_ref_, method_index,
                    false,
                    ref_->keypoints_, curr_->keypoints_,
                    max_matching_pixel_dist_in_triangulation);

                // Find inliers by epipolar constraint
                // 这里只是帮助找出内点的
                // 不过使用了本质矩阵估计方法，而前面的初始化部分是综合了本质矩阵和H矩阵的结果
                curr_->inliers_matches_with_ref_ = geometry::helperFindInlierMatchesByEpipolarCons(
                    ref_->keypoints_, curr_->keypoints_, curr_->matches_with_ref_, K);

                // Print
                printf("For triangulation: Matches with prev keyframe: %d; Num inliers: %d \n",
                       (int)curr_->matches_with_ref_.size(), (int)curr_->inliers_matches_with_ref_.size());

                // Triangulate points
                // 使用内点进行三角化，位姿已知
                curr_->inliers_pts3d_ = geometry::helperTriangulatePoints(
                    ref_->keypoints_, curr_->keypoints_,
                    curr_->inliers_matches_with_ref_, getMotionFromFrame1to2(curr_, ref_), K);

                // 这个函数是干啥的？？？？
                // 计算每一个点的三角化角度进行统计，移除角度偏小和偏大的点
                retainGoodTriangulationResult_();

                // -- Update state
                pushCurrPointsToMap_(); // 把地图点加到地图里
                optimizeMap_(); // 这里是根据一些策略来删除地图中的一些点
                addKeyFrame_(curr_);  // 将当前关键帧加入map，同时ref设置为当前关键帧
            }

            // 不是关键帧就只是用来优化，其他的什么都没干
        }
    }

    // Print relative motion
    if (vo_state_ == DOING_TRACKING)
    {
        static cv::Mat T_w_to_prev = cv::Mat::eye(4, 4, CV_64F); // 这里有问题把？？？
        const cv::Mat &T_w_to_curr = curr_->T_w_c_;
        cv::Mat T_prev_to_curr = T_w_to_prev.inv() * T_w_to_curr;
        cv::Mat R, t;
        basics::getRtFromT(T_prev_to_curr, R, t);
        cout << "\nCamera motion:" << endl;
        cout << "R_prev_to_curr: " << R << endl;
        cout << "t_prev_to_curr: " << t.t() << endl;
    }
    prev_ = curr_;
    cout << "\nEnd of a frame" << endl;
}

} // namespace vo
} // namespace my_slam
