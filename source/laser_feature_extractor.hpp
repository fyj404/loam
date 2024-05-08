
#ifndef LASER_FEATURE_EXTRACTION_H
#define LASER_FEATURE_EXTRACTION_H

#include <cmath>
#include <opencv/cv.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <string>
#include <vector>
#include "livox_feature_extractor.hpp"
#include "tools/common.h"
#include "tools/tools_logger.hpp"
#include <iostream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>

using std::atan2;
using std::cos;
using std::sin;
using namespace Common_tools;
class Param
{
public:
    std::string pcd_file_folder;
    int m_laser_scan_number;
    int m_lidar_type;
    int m_piecewise_number;
    int m_if_motion_deblur;
    int m_maximum_input_lidar_pointcloud;
    ~Param() {}
    Param()
    {
        pcd_file_folder = "/home/f404/pcd_file_folder";
        m_laser_scan_number = 64;
        m_lidar_type = 1;
        m_piecewise_number = 1;
        m_if_motion_deblur = 1;
        m_maximum_input_lidar_pointcloud=1;
    }
};
class Laser_feature
{
public:
    const double m_para_scanPeriod = 0.1;

    int m_if_pub_debug_feature = 1;

    const int m_para_system_delay = 5;
    int m_para_system_init_count = 0;
    bool m_para_systemInited = false;
    float m_pc_curvature[400000];
    int m_pc_sort_idx[400000];
    int m_pc_neighbor_picked[400000];
    int m_pc_cloud_label[400000];
    int m_if_motion_deblur = 1;
    int m_odom_mode = 0; // 0 = for odom, 1 = for mapping
    float m_plane_resolution;
    float m_line_resolution;
    int m_piecewise_number = 1;


    PCL_point_cloud_to_pcd m_pcl_tools_feature,m_pcl_tools_corner,m_pcl_tools_surf;
    int m_maximum_input_lidar_pointcloud = 1;
    File_logger m_file_logger;

    bool m_if_pub_each_line = false;
    int m_lidar_type = 1; // 0 is velodyne, 1 is livox
    int m_laser_scan_number = 64;
    std::mutex m_mutex_lock_handler;
    Param param;
    Livox_laser m_livox;

    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> m_map_pointcloud_full_vec_vec;
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> m_map_pointcloud_surface_vec_vec;
    std::vector<std::vector<pcl::PointCloud<pcl::PointXYZI>>> m_map_pointcloud_corner_vec_vec;
    ADD_SCREEN_PRINTF_OUT_METHOD;

    bool comp(int i, int j)
    {
        return (m_pc_curvature[i] < m_pc_curvature[j]);
    }
    ~Laser_feature() {}
    Laser_feature(Param _param)
    {
        param = _param;
        m_laser_scan_number = param.m_laser_scan_number;
        m_lidar_type = param.m_lidar_type;
        m_piecewise_number = param.m_piecewise_number;
        m_if_motion_deblur = param.m_if_motion_deblur;
        m_maximum_input_lidar_pointcloud=param.m_maximum_input_lidar_pointcloud;
        m_map_pointcloud_corner_vec_vec.resize(3);
        m_map_pointcloud_surface_vec_vec.resize(3);
        m_map_pointcloud_full_vec_vec.resize(3);
        for (int i = 0; i < 3; i++)
        {
            m_map_pointcloud_corner_vec_vec[i].resize(m_piecewise_number);
            m_map_pointcloud_surface_vec_vec[i].resize(m_piecewise_number);
            m_map_pointcloud_full_vec_vec[i].resize(m_piecewise_number);
        }
        std::string m_pcd_save_dir_name="/home/f404/loam_no_ros/feature/";
        m_pcl_tools_feature.set_save_dir_name( m_pcd_save_dir_name );
        m_pcl_tools_corner.set_save_dir_name( m_pcd_save_dir_name );
        m_pcl_tools_surf.set_save_dir_name( m_pcd_save_dir_name );
    }
    void laserCloudHandler(const pcl::PointCloud<pcl::PointXYZI>::Ptr &laserCloudMsg,std::string idx)
    {
        int current_lidar_index = 0;
        std::unique_lock<std::mutex> lock(m_mutex_lock_handler);
        std::vector<pcl::PointCloud<PointType>> laserCloudScans(m_laser_scan_number);
        if (!m_para_systemInited)
        {
            m_para_system_init_count++;

            if (m_para_system_init_count >= m_para_system_delay)
            {
                m_para_systemInited = true;
            }
            else
                return;
        }
        std::vector<int> scanStartInd(1000, 0);
        std::vector<int> scanEndInd(1000, 0);
        pcl::PointCloud<pcl::PointXYZI> laserCloudIn;
        laserCloudIn = *laserCloudMsg;
        int raw_pts_num = laserCloudIn.size();

        // m_file_logger.printf( " Time: %.5f, num_raw: %d, num_filted: %d\r\n",
        // laserCloudMsg->header.stamp.toSec(), raw_pts_num, laserCloudIn.size() );

        size_t cloudSize = laserCloudIn.points.size();

        if (m_lidar_type) // Livox scans
        {   
            //std::cout<<"begin m_livox.extract_laser_features"<<std::endl;
            laserCloudScans = m_livox.extract_laser_features(laserCloudIn, 19260817.00);
            //std::cout<<"laserCloudScans.size"<<laserCloudScans.size()<<std::endl;
            if (laserCloudScans.size() <= 5) // less than 5 scan
            {
                return;
            }
            m_laser_scan_number = laserCloudScans.size() * 1;
            scanStartInd.resize(m_laser_scan_number);
            scanEndInd.resize(m_laser_scan_number);
            std::fill(scanStartInd.begin(), scanStartInd.end(), 0);
            std::fill(scanEndInd.begin(), scanEndInd.end(), 0);
            /********************************************
             *    Feature extraction for livox lidar     *
             ********************************************/
            int piece_wise = m_piecewise_number;
            if (m_if_motion_deblur)
            {
                piece_wise = 1;
            }
            vector<float> piece_wise_start(piece_wise);
            vector<float> piece_wise_end(piece_wise);

            std::cout<<"begin calc.start_scan and end_scan"<<piece_wise <<std::endl;
            for (int i = 0; i < piece_wise; i++)
            {
                int start_scans, end_scans;

                start_scans = int((m_laser_scan_number * (i)) / piece_wise);
                end_scans = int((m_laser_scan_number * (i + 1)) / piece_wise) - 1;

                //std::cout<<"start_scans and end_scans"<<start_scans<<"  "<<end_scans<<std::endl;
                int end_idx = laserCloudScans[end_scans].size() - 1;
                piece_wise_start[i] = ((float)m_livox.find_pt_info(laserCloudScans[start_scans].points[0])->idx) / m_livox.m_pts_info_vec.size();
                piece_wise_end[i] = ((float)m_livox.find_pt_info(laserCloudScans[end_scans].points[end_idx])->idx) / m_livox.m_pts_info_vec.size();
                //std::cout<<"piece_start "<<piece_wise_start[i]<<" piece_wise_end"<<piece_wise_end[i]<<std::endl;
            }
            for (int i = 0; i < piece_wise; i++)
            {
                pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>()),
                    livox_surface(new pcl::PointCloud<PointType>()),
                    livox_full(new pcl::PointCloud<PointType>());
                m_livox.get_features(*livox_corners, *livox_surface, *livox_full, piece_wise_start[i], piece_wise_end[i]);
                m_map_pointcloud_corner_vec_vec[current_lidar_index][i] = *livox_corners;
                m_map_pointcloud_surface_vec_vec[current_lidar_index][i] = *livox_surface;
                m_map_pointcloud_full_vec_vec[current_lidar_index][i] = *livox_full;
            }
            for (int i = 0; i < piece_wise; i++)
            {

                //ros::Time current_time = ros::Time::now();
                pcl::PointCloud<PointType>::Ptr livox_corners(new pcl::PointCloud<PointType>()),
                    livox_surface(new pcl::PointCloud<PointType>()),
                    livox_full(new pcl::PointCloud<PointType>()),livox_feature(new pcl::PointCloud<PointType>());
                if (1)
                {
                    if (current_lidar_index != 0)
                    {
                        return;
                    }

                    for (int ii = 0; ii < m_maximum_input_lidar_pointcloud; ii++)
                    {
                        *livox_full += m_map_pointcloud_full_vec_vec[ii][i];
                        *livox_surface += m_map_pointcloud_surface_vec_vec[ii][i];
                        *livox_corners += m_map_pointcloud_corner_vec_vec[ii][i];
                    }
                }
                else
                {
                    *livox_full = m_map_pointcloud_full_vec_vec[current_lidar_index][i];
                    *livox_surface = m_map_pointcloud_surface_vec_vec[current_lidar_index][i];
                    *livox_corners = m_map_pointcloud_corner_vec_vec[current_lidar_index][i];
                }
                *livox_feature+=*livox_surface;
                *livox_feature+=*livox_corners;
                int frame_index=1;
                frame_index=std::stoi(idx.substr(0, idx.length() - 4));
                std::cout<<"save feature cloud"<<frame_index<<" "<<livox_feature->size()<<std::endl;
                m_pcl_tools_feature.save_to_pcd_files( "feature", *livox_feature, frame_index );
                m_pcl_tools_corner.save_to_pcd_files( "corner", *livox_corners, frame_index );
                m_pcl_tools_surf.save_to_pcd_files( "surf", *livox_surface, frame_index );
            }
            

        }
    }
    void run()
    {
        std::string directory_path = param.pcd_file_folder;
        if (!boost::filesystem::exists(directory_path) || !boost::filesystem::is_directory(directory_path))
        {
            std::cerr << "Invalid directory: " << directory_path << std::endl;
            return;
        }
        for (boost::filesystem::directory_entry &entry : boost::filesystem::directory_iterator(directory_path))
        {
            if (entry.path().extension() == ".pcd")
            {
                std::string pcd_file_path = entry.path().string();
                std::cout << "Loading file: " << pcd_file_path << std::endl;

                // Load the PCD file using PCL
                pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
                if (pcl::io::loadPCDFile<pcl::PointXYZI>(pcd_file_path, *cloud) == -1)
                {
                    std::cerr << "Failed to load file: " << pcd_file_path << std::endl;
                    continue;
                }
                std::string filename = entry.path().filename().stem().string();
                std::string idx = filename.substr(4);
                // Process the point cloud data as needed
                // For example, you can perform various operations on the loaded point cloud here

                std::cout << "Loaded " << cloud->points.size() << " points from " << pcd_file_path << std::endl;
                laserCloudHandler(cloud,idx);
            }
        }
    }
};

#endif // LASER_FEATURE_EXTRACTION_H
