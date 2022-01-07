#include<cmath>
#include<eigen3/Eigen/Core>
#include<eigen3/Eigen/Dense>
#include<iostream>

#define PI std::acos(-1)
//Degree to Rad 角度转弧度
float convertDtoRad(float degree) {
    float rad = degree / 180.0 * PI;//不可以是180
    return rad;
}

//Rad to Degree 弧度转角度
float convertRadtoD(float rad) {
    float degree = rad / 180.0 * PI;
    return degree;
}

int main() {
    //点P = (2,1)，先绕原点逆时针旋转45°，再平移(1,2)，计算结果。（用齐次坐标）
    //先旋转后平移，可以写成一个矩阵，顺序是对的
    Eigen::Vector3f p(2.0f, 1.0f, 1.0f);//error 2d point.z = 1
    float degree = 45;
    float sinA = std::sin(convertDtoRad(degree));
    float cosA = std::cos(convertDtoRad(degree));
    float tx = 1.0f;
    float ty = 2.0f;
    //Transformation matrix
    Eigen::Matrix3f t;//error no ()
    t << cosA, 0 - sinA, tx,
        sinA, cosA, ty,
        0.0f, 0.0f, 1.0f;

    std::cout << t << std::endl;
    Eigen::Vector3f result = t * p;
    std::cout << "Homogeneous:" << std::endl;

    std::cout << result << std::endl;
    //如果要变回笛卡尔坐标,这里不需要
    std::cout << "Cartesian:" << std::endl;
    std::cout << result / result(2) << std::endl;

    return 0;
}