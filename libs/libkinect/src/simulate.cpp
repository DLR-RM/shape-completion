#include <render_kinect/kinectSimulator.h>
#include <csetjmp>
#include <csignal>

static std::jmp_buf jump_buffer;

extern "C" void segfault_handler(int signal)
{
  std::longjmp(jump_buffer, signal);
}

extern "C"
{
  int simulate(float *vertices,
               int num_verts,
               int *faces,
               int num_faces,
               float *out_depth,
               int width,
               int height,
               float fx,
               float fy,
               float cx,
               float cy,
               float z_near,
               float z_far,
               float baseline,
               int noise_type,  // 0=NONE, 1=GAUSSIAN, 2=PERLIN, 3=SIMPLEX
               const char *dot_pattern_path,
               bool debug)
  {
    if (setjmp(jump_buffer) == 0)
    {
      std::signal(SIGSEGV, segfault_handler);

      render_kinect::CameraInfo cam_info;

      cam_info.width = width;
      cam_info.height = height;
      cam_info.cx_ = cx;
      cam_info.cy_ = cy;
      cam_info.z_near = z_near;
      cam_info.z_far = z_far;
      cam_info.fx_ = fx;
      cam_info.fy_ = fy;
      cam_info.tx_ = baseline;

      // Set noise type
      switch (noise_type) {
        case 0: cam_info.noise_ = render_kinect::NONE; break;
        case 1: cam_info.noise_ = render_kinect::GAUSSIAN; break;
        case 2: cam_info.noise_ = render_kinect::PERLIN; break;
        case 3: cam_info.noise_ = render_kinect::SIMPLEX; break;
        default: cam_info.noise_ = render_kinect::PERLIN; break;
      }

      if (debug)
      {
        std::cout << "Instantiating KinectSimulator" << std::endl;
      }
      std::string dot_path(dot_pattern_path);
      render_kinect::KinectSimulator object_model(cam_info, vertices, num_verts, faces, num_faces, dot_path);

      Eigen::Affine3d current_tf = Eigen::Affine3d::Identity();
      cv::Mat point_cloud_, labels_;
      cv::Mat depth_im_(cam_info.height, cam_info.width, CV_32FC1, out_depth);
      depth_im_.setTo(0);  // Initialize depth buffer to 0
      cv::Mat scaled_im_(cam_info.height, cam_info.width, CV_32FC1);

      if (debug)
      {
        std::cout << "Simulating measurement" << std::endl;
      }
      object_model.intersect(current_tf, point_cloud_, depth_im_, labels_, debug);
      if (debug)
      {
        std::cout << "Done." << std::endl;
      }
      return 0;
    }
    else
    {
      std::cout << "Caught segfault" << std::endl;
      return 1;
    }
  }
}
