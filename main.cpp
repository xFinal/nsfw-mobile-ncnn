#include <omp.h>
#include <opencv2/opencv.hpp>
#include <ncnn/net.h>

#include <vector>
using namespace std;


const int INPUT_SIZE = 224;
const float NORM[3] = { 1.0f / 255.f, 1.0f / 255.f, 1.0f / 255.f };
const float MEAN[3] = { 0, 0, 0 };
const vector<string> labels = {"drawings", "hentai", "neutral", "porn", "sexy"};


int main()
{
    ncnn::Net m_net;
    int ret = m_net.load_param("./nsfw-fp16.param");
    if (ret != 0) {
        return ret;
    }
    ret = m_net.load_model("./nsfw-fp16.bin");
    if (ret != 0) {
        return ret;
    }

    cv::Mat src_mat = cv::imread("./porn.jpg");
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src_mat.clone().data, ncnn::Mat::PIXEL_BGR2RGB,
        src_mat.cols, src_mat.rows, INPUT_SIZE, INPUT_SIZE);
    input.substract_mean_normalize(MEAN, NORM);

    // Inference
    ncnn::Extractor extractor = m_net.create_extractor();
    ncnn::Mat output;
    ret = extractor.input("130:12", input);
    if (ret != 0) {
        return ret;
    }
    ret = extractor.extract("304:12", output);
    if (ret != 0) {
        return ret;
    }

    for (int i = 0; i < output.w; ++i) {
        cout << labels[i] << ": " << output.channel(0)[i] << endl;
    }

    return ret;
}