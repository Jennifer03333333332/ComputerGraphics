// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>


rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f>& positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return { id };
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i>& indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return { id };
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f>& cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return { id };
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}

//计算该点在不在三角形内
static bool insideTriangle(float x, float y, const Vector3f* _v)//_v是个存储Vector3f的数组
{
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
    //？三角形的三维点怎么和二维比？这里z无意义去掉
    Eigen::Vector2f p((float)x, (float)y);
    Vector2f v0v1 = _v[1].head(2) - _v[0].head(2);
    Vector2f v1v2 = _v[2].head(2) - _v[1].head(2);
    Vector2f v2v0 = _v[0].head(2) - _v[2].head(2);
    // 
    Vector2f v0p = p - _v[0].head(2);
    Vector2f v1p = p - _v[1].head(2);
    Vector2f v2p = p - _v[2].head(2);

    //如果叉乘是0，说明点在三角形边上，暂时不管
    float cross1 = v0v1[0] * v0p[1] - v0v1[1] * v0p[0];
    float cross2 = v1v2[0] * v1p[1] - v1v2[1] * v1p[0];
    float cross3 = v2v0[0] * v2p[1] - v2v0[1] * v2p[0];
    //三叉乘同号
    return (cross1 * cross2 > 0) && (cross2 * cross3 > 0) && (cross1 * cross3 > 0);
}


//计算??
static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * y + v[1].x() * v[2].y() - v[2].x() * v[1].y()) / (v[0].x() * (v[1].y() - v[2].y()) + (v[2].x() - v[1].x()) * v[0].y() + v[1].x() * v[2].y() - v[2].x() * v[1].y());
    float c2 = (x * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * y + v[2].x() * v[0].y() - v[0].x() * v[2].y()) / (v[1].x() * (v[2].y() - v[0].y()) + (v[0].x() - v[2].x()) * v[1].y() + v[2].x() * v[0].y() - v[0].x() * v[2].y());
    float c3 = (x * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * y + v[0].x() * v[1].y() - v[1].x() * v[0].y()) / (v[2].x() * (v[0].y() - v[1].y()) + (v[1].x() - v[0].x()) * v[2].y() + v[0].x() * v[1].y() - v[1].x() * v[0].y());
    return { c1,c2,c3 };
}

// error: ‘calculateZinterpolatedZ’ was not declared in this scope
// solution: add 'rst::rasterizer::' 把函数写进类里，要在头文件里加入声明
// 不能不按序调用为声明的函数？？除非在类里
float rst::rasterizer::getMSAAInsideTriangleValue(float x, float y, const Triangle& t, float &minZ) {
    float insideTValue = 0;//颜色占比
    //2*2 row行col列
    float row = 2, col = 2;
    float ex = x, ey = y;
    //each sample
    for (int i = 0; i < row; i++) {
        for (int o = 0; o < col; o++) {
            //该采样点的中心
            if (insideTriangle(ex + 1 / (2 * col), ey + 1 / (2 * row), t.v)) {
                insideTValue += 1 / (col * row);
                minZ = std::min(minZ, calculateZinterpolatedZ(ex + 1 / (2 * col), ey + 1 / (2 * row), t));//取所有采样点中，z最小的
            }
            ex += 1 / col;
        }
        ex = x;
        ey += 1 / row;
    }

    return insideTValue;
}
//计算z插值
float rst::rasterizer::calculateZinterpolatedZ(float x, float y, const Triangle& t){
    auto v = t.toVector4();
    auto tup = computeBarycentric2D(x, y, t.v);
    float alpha, beta, gamma;
    std::tie(alpha, beta, gamma) = tup;
    float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    z_interpolated *= w_reciprocal;
    return z_interpolated;
}


void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];//position
    auto& ind = ind_buf[ind_buffer.ind_id];//indices 0123456...
    auto& col = col_buf[col_buffer.col_id];//color

    float f1 = (50 - 0.1) / 2.0;//(far-near)/2
    float f2 = (50 + 0.1) / 2.0;//(far+near)/2

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)//对每一个index
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto& vert : v)
        {
            vert.x() = 0.5 * width * (vert.x() + 1.0);//-1,1 -> 0,width
            vert.y() = 0.5 * height * (vert.y() + 1.0);
            vert.z() = vert.z() * f1 + f2;//z?
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();//获得Triangle的三个顶点的齐次坐标形式,v是std::array<Eigen::Vector4f, 3> 

    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle
    // 1 创建2维bounding box
    int xmin = (int)std::min(v[0][0], std::min(v[1][0], v[2][0]));
    int ymin = (int)std::min(v[0][1], std::min(v[1][1], v[2][1]));
    int xmax = (int)std::max(v[0][0], std::max(v[1][0], v[2][0])) + 1;
    int ymax = (int)std::max(v[0][1], std::max(v[1][1], v[2][1])) + 1;

    bool MSAA = true;

    if (MSAA) {
        //锯齿原因分析：像素点内有小采样点应该被涂色和记录却没有
        //每个像素划成2*2，对4个采样求解在不在三角形内,对每个采样记录深度值
        //如果一个子采样点在三角形内，那么该子采样点所代表的像素的颜色值就加上这个三角形颜色的四分之一
        for (float x = xmin; x <= xmax; x++) {
            for (float y = ymin; y <= ymax; y ++) {
                float minZforThisPixel = FLT_MAX;
                //这个值究竟与z_interpolated有什么关系？目前无关 先往后看看学习一下z_interpolated
                int MSAAValue = getMSAAInsideTriangleValue(x, y, t, minZforThisPixel);

                //当这个像素点在当前三角形内时
                if (MSAAValue>0) {
                    //framework:get the interpolated z value. 计算当前三角形在这一像素上的深度

                    //MSAA下，depth_buf和framebuffer是否都要增大相应倍数？先不用
                    if (minZforThisPixel < depth_buf[get_index(x, y)]) {//当前三角形在这一像素的深度<记录的深度
                        Eigen::Vector3f point((float)x, (float)y, minZforThisPixel);
                        set_pixel(point, t.getColor()* MSAAValue);//update pixel为这个三角形的颜色
                        //depth_buf是被FLX_MAX填满的数组
                        depth_buf[get_index(x, y)] = minZforThisPixel;//update deep buffer
                    }

                }
            }
        }
        return;
    }
    // Not MSAA
    // 2 遍历boundingbox内所有像素，检查像素中心是否在三角形内
    for (int x = xmin; x <= xmax; x++) {
        for (int y = ymin; y <= ymax; y++) {
            //error 检查的是像素中心点 x+0.5,y+0.5
            if (insideTriangle((float)(x+0.5), (float)(y+0.5), t.v)) {//在三角形内
                //get the interpolated z value.
                auto tup = computeBarycentric2D(x + 0.5, y+0.5, t.v);
                float alpha, beta, gamma;
                std::tie(alpha, beta, gamma) = tup;
                float w_reciprocal = 1.0 / (alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                z_interpolated *= w_reciprocal;
                /////
                if (z_interpolated < depth_buf[get_index(x, y)]) {//depth_buf 是 std::vector<float>
                    //error point不是(x,y,1)？？？是 z_interpolated 但是set_pixel的时候没有用到z值
                    Eigen::Vector3f point((float)x, (float)y, z_interpolated);//
                    set_pixel(point, t.getColor());//update pixel
                    depth_buf[get_index(x, y)] = z_interpolated;//update deep buffer
                }

            }
        }
    }


    // If so, use the following code to 
    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{ 0, 0, 0 });
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        //初始深度为无穷大
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    depth_buf.resize(w * h);
}

//Int or float type
template <class myType>
float rst::rasterizer::get_index(myType x, myType y)
{
    return (height - 1 - y) * width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height - 1 - point.y()) * width + point.x();
    frame_buf[ind] = color;

}

// clang-format on