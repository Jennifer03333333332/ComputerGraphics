#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

float convertDtoRad(float degree) {
    float rad = degree / 180.0 * MY_PI;//不可以是180     
    return rad;
}
//mvp's v:placing camera to origin??
Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0], 
        0, 1, 0, -eye_pos[1], 
        0, 0, 1,-eye_pos[2], 
        0, 0, 0, 1;

    view = translate * view;

    return view;
}
//rotation_angle感觉是角度不是弧度
Eigen::Matrix4f get_model_matrix(Vector3f axis, float rotation_angle)
{
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    //Eigen::Matrix4f rotationMatrix;
    //rotationMatrix << std::cos(convertDtoRad(rotation_angle)), 0 - std::sin(convertDtoRad(rotation_angle)), 0, 0,
    //    std::sin(convertDtoRad(rotation_angle)), std::cos(convertDtoRad(rotation_angle)), 0, 0,
    //    0, 0, 1, 0,
    //    0, 0, 0, 1;
    //model = rotationMatrix;
    model = get_rotation(Vector3f axis, float angle) * model;
    return model;
}
//45,1,0.1,50
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
    float zNear, float zFar)
{
    // Students will implement this function

    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();

    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
    float top = std::tan(convertDtoRad(eye_fov / 2)) * abs(zNear);
    float right = top * aspect_ratio;

    Eigen::Matrix4f perspToOrthoMatrix, orthoMatrixScale, orthoMatrixTranslate;
    perspToOrthoMatrix << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, (-1)* zNear* zFar,
        0, 0, 1, 0;

    orthoMatrixTranslate << 
        1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0 - (zNear + zFar) / 2,
        0, 0, 0, 1;
    orthoMatrixScale << 
        1 / right, 0, 0, 0,
        0, 1 / top, 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;
    projection = orthoMatrixScale * orthoMatrixTranslate * perspToOrthoMatrix;
    return projection;
}
//绕任意经过原点的轴 旋转
//axis = n
Eigen::Matrix4f get_rotation(Vector3f axis, float angle) {
    //结果
    Eigen::Matrix4f rotate_martix = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f RodriguesMatrix;

    Eigen::Matrix3f I = Eigen::Matrix3f::Identity();
    //向量叉乘的对偶矩阵
    Eigen::Matrix3f nXMatrix;
    nXMatrix << 0, 0 - axis(2), axis(1),
        axis(2), 0, 0 - axis(0),
        0 - axis(1), axis(0), 0;
    //三维的
    RodriguesMatrix = std::cos(convertDtoRad(angle)) * I + (1 - std::cos(convertDtoRad(angle))) * axis * axis.transpose() + std::sin(convertDtoRad(angle)) * nXMatrix;

    //Eigen有自带的轴角旋转矩阵
    Eigen::AngleAxisf rotation_vector(convertDtoRad(angle), axis);
    Eigen::Matrix3f Eigen_rotation_matrix = rotation_vector.toRotationMatrix();

    //从[0,0]开始每行取3每列取3//三维转齐次
    rotate_martix.block(0, 0, 3, 3) = Eigen_rotation_matrix;//RodriguesMatrix;

    return rotate_martix;
}

//argc (argument count) 命令行有多少个参数
//argv (argument vector)指向一个数组，保存命令行参数
int main(int argc, const char** argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3) {
        command_line = true;
        //std::stof 将string parse为floating-point number
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4) {
            filename = std::string(argv[3]);
        }
        else
            return 0;
    }
    //define 700*700 screen
    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = { 0, 0, 5 };
    //target三角形三个顶点
    std::vector<Eigen::Vector3f> pos{ {2, 0, -2}, {0, 2, -2}, {-2, 0, -2} };
    //? index value,start from 0
    std::vector<Eigen::Vector3i> ind{ {0, 1, 2} };

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line) {
        //fill framebuf by vector3f(0,0,0)
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));
        //draw
        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27) {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        //mvp
        //轴角旋转替换掉z轴旋转
        //r.set_model(get_model_matrix(angle));
        r.set_model(get_model_matrix(Vector3f(0, 0, 1), angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));



        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a') {
            angle += 10;
        }
        else if (key == 'd') {
            angle -= 10;
        }
    }

    return 0;
}