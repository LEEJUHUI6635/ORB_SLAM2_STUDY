/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

// prototype
// image를 load하는 함수
// const string &strPathToSequence : 어떠한 sequence의 dataset을 쓸 것인지에 대한 string input -> const + reference = input
// vector<string> &vstrImageLeft : vector(동적 할당 배열) 자료형인 left image 이름을 output -> reference = output
// vector<string> &vstrImageRight : vector(동적 할당 배열) 자료형인 right image 이름을 output -> reference = output
// vector<double> &vTimestamps : double 자료형인 timestamps를 output -> reference = output
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps);

// int argc : main 함수에 전달되는 데이터의 갯수
// char **argv : main 함수에 전달되는 실제적인 데이터로 char형 포인터 배열로 구성되어 있다. 첫 번째 문자열은 프로그램의 실행경로이다. 
int main(int argc, char **argv) // terminal에서 main 함수를 시작하기 위해서, -> Q. 이중 포인터를 왜 쓰지? A. 여러 개의 string을 쓰기 위해
{   // main 함수에 전달되는 데이터의 갯수가 4개가 아닌 경우
    if(argc != 4) // char **argv = ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence
    {
        // cerr = cout error
        cerr << endl << "Usage: ./stereo_kitti path_to_vocabulary path_to_settings path_to_sequence" << endl;
        return 1; // 1 이상의 숫자 : 정상 종료 되었으나, 다른 무엇인가 있음을 나타낸다.
    }

    // Retrieve paths to images
    vector<string> vstrImageLeft; // left image path에 해당하는 string type의 vector(동적 할당 배열) 
    vector<string> vstrImageRight; // right image path에 해당하는 string type의 vector(동적 할당 배열)
    vector<double> vTimestamps;

    // 해당하는 sequence의 path로 가서, 폴더의 각 이미지명을 추출
    LoadImages(string(argv[3]), vstrImageLeft, vstrImageRight, vTimestamps); // string(argv[3]) = /dataset/sequences/00

    const int nImages = vstrImageLeft.size(); // left image의 개수 = nImages -> const : 코드의 뒷 부분에서 바꾸지 않는다.

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::STEREO,true); // System class instanciation -> 생성자를 통해 초기화
    // argv[1] = path_to_vocabulary, argv[2] = path_to_settings
    // System(const string &strVocFile, const string &strSettingsFile, const eSensor sensor, const bool bUseViewer = true)

    // Vector for tracking time statistics
    // tracking time check를 위한 vTimesTrack vector 선언
    vector<float> vTimesTrack; // float 자료형의 vector(동적 할당 배열) vTimesTrack 정의
    vTimesTrack.resize(nImages); // 동적 할당 배열이기 때문에 얼마만큼 메모리를 할당 받을 지 정해지지 않았기 때문에,
    // nImages(left image의 개수) x 4 byte 만큼 메모리를 할당 받은 것을 알 수 있다.
    // resize() -> 초기화 O(capacity와 size가 할당), reserve() -> 초기화 X(capacity만 할당)

    // image processing
    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imLeft, imRight; // left image와 right image에 대한 matrix 정의
    for(int ni=0; ni<nImages; ni++) // vstrImageLeft, vstrImageRight vector 안에는 image의 path가 담겨있다.
    {
        // Read left and right images from file
        imLeft = cv::imread(vstrImageLeft[ni],CV_LOAD_IMAGE_UNCHANGED); // left image path로부터 left image를 읽어온다.
        imRight = cv::imread(vstrImageRight[ni],CV_LOAD_IMAGE_UNCHANGED); // right image path로부터 right image를 읽어온다.
        double tframe = vTimestamps[ni]; // 해당하는 image의 timestamp = tframe

        if(imLeft.empty()) // left image가 비어있을 경우,
        {
            // std::cout은 buffered로 동작하기 때문에 원하는 위치에서 정상적인 출력이 되지 않을 수 있다.
            // 따라서 std::cout으로 로그를 찍기 위해서는 버퍼를 비우는 동작을 추가로 진행해야 하기 때문에 std::endl이 수행된다.
            // std::cerr는 unitbuffered로 동작한다. unitbuffered란 이 연산이 buffered이긴 하지만, 출력 이후 바로 버퍼를 비운다는 것을 의미한다.
            // cerr = cout error
            cerr << endl << "Failed to load image at: "
                 << string(vstrImageLeft[ni]) << endl;
            return 1; // 1 이상의 숫자 : 정상 종료 되었으나, 다른 무엇인가 있음을 나타낸다.
        }

// Q. COMPILEDWITHC11 정의가 어디서 이루어지는가? A. CMakeLists.txt
// #if와 #ifdef의 차이 : #if는 만약 ~라면, #ifdef는 만약 ~가 정의되어 있다면을 의미한다.
#ifdef COMPILEDWITHC11 // 만일 COMPILEDWITHC11이 정의가 되어 있다면,
        // std::chrono::steady_clock -> 물리적으로 역행하지 않는 시간, 시간의 흐름이 바뀌지 않는 시간
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now(); // 지금 시간을 time_point t1으로 나타낸다.
#else // 만일 COMPILEDWITHC11이 정의가 되어있지 않다면,
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now(); // tracking의 소요 시간을 측정하기 위함
#endif // image가 SLAM system에 들어가는 시간 -> t1

        // Pass the images to the SLAM system
        SLAM.TrackStereo(imLeft,imRight,tframe); // left image, right image, timestamp를 input으로 갖는 함수 -> const + reference = input
        // cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
        // Q. return 값이 없다.
        
#ifdef COMPILEDWITHC11 // 만일 COMPILEDWITHC11이 정의가 되어 있다면,
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif
        // ttrack : 시간의 기간을 std::chrono::duration<double> 단위로 casting한다.
        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack; // image 하나 처리할 때의 시간 -> vTimesTrack vector

        // Q. Wait to load the next frame
        double T=0;
        if(ni<nImages-1) // 마지막 image를 제외한 나머지 image라면,
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6); // (T-ttrack)*1e6 시간 동안 대기
    }

    // Stop all threads
    SLAM.Shutdown(); // 모든 image에 대한 처리가 끝나면,

    // Tracking time statistics
    // vTimesTrack.begin() : vector의 data가 있는 list의 시작 주소를 return, 첫 번째 값의 위치를 반환
    // vTimesTrack.end() : vector의 data가 있는 list의 끝 주소를 return, 마지막 값의 위치를 반환
    // 배열 자체를 print -> 주소
    sort(vTimesTrack.begin(),vTimesTrack.end()); // sort(배열의 시작점 주소, 마지막 주소 + 1) -> 배열 원소들을 sorting
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni]; // 모든 image들을 처리하는데 걸리는 시간 = totaltime
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl; // 정렬된 vTimesTrack vector의 중간 값
    cout << "mean tracking time: " << totaltime/nImages << endl; // 하나의 image를 처리하는데 걸리는 평균 시간

    // Save camera trajectory
    SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");
    // void System::SaveTrajectoryKITTI(const string &filename) -> filename을 input으로 받아, camera odometry 정보를 txt 파일에 저장한다.
    // const string & = input
    return 0;
}

// const string &strPathToSequence -> input, vector<string> &vstrImageLeft -> output, 
// vector<string> &vstrImageRight -> output, vector<double> &vTimestamps -> output
// vstrImageLeft -> left image folder 안의 각 image명, vstrImageRight -> right image folder 안의 각 image명, vTimestamps -> times.txt에 있는 timestamp
void LoadImages(const string &strPathToSequence, vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight, vector<double> &vTimestamps)
{
    // ifstream -> 파일 입출력 클래스
    ifstream fTimes; // 파일 입출력 클래스 instanciation
    string strPathTimeFile = strPathToSequence + "/times.txt"; // /dataset/sequences/00/times.txt
    fTimes.open(strPathTimeFile.c_str()); // Q. C string 형식으로 변환한다. String class에서 담고 있는 문자열을 C에서의
    // const char* 타입으로 변환하여 반환해주는 편리한 멤버 함수이다. -> char 형의 pointer로 변환
    // A. ifstream class의 객체인 fTimes의 멤버 함수인 open()은 const char*의 형식으로 input을 받는다.

    // 파일의 끝에 도달할 때까지 계속 반복
    while(!fTimes.eof()) // eof() = End of File, 파일의 끝에 도달하면 eof가 반환된다.
    {
        string s; // 비어있는 문자열 s 선언
        // std::getline() : C++에서 한 줄씩 파일을 읽는 데 선호되는 방법으로, 구분 기호 문자가 나타날 때까지 입력 스트림에서 문자를 읽은 다음, 문자열에 저장한다.
        // 파일을 한 줄씩 읽어, 기본 구분 기호 문자인 \n이 나타날 때까지 s에 저장한다.
        getline(fTimes,s);

        if(!s.empty()) // 문자열 s가 비어있지 않다면,
        {
            // stringstream : 문자열에서 내가 원하는 자료형의 데이터를 추출할 때 사용한다.
            stringstream ss;
            ss << s; // ss에 s를 저장
            double t;
            ss >> t; // double 형의 t에 ss를 저장
            vTimestamps.push_back(t); // vector vTimestamps의 요소로 t를 추가한다.
        }
    }
    
    // left image folder, right image folder
    string strPrefixLeft = strPathToSequence + "/image_0/"; // /dataset/sequences/00/image_0/
    string strPrefixRight = strPathToSequence + "/image_1/"; // /dataset/sequences/00/image_1/

    const int nTimes = vTimestamps.size(); // timestamp의 개수 = nTimes -> image의 개수

    vstrImageLeft.resize(nTimes); // vector vstrImageLeft의 크기를 nTimes로 resize -> 동적 할당 배열이기 때문에 메모리 절약을 위해,
    vstrImageRight.resize(nTimes); // vector vstrImageRight의 크기를 nTimes로 resize -> 동적 할당 배열이기 때문에 메모리 절약을 위해,

    for(int i=0; i<nTimes; i++) // image의 개수만큼 반복
    {
        stringstream ss;
        ss << setfill('0') << setw(6) << i; // setfill 함수를 이용하여 0으로, setw 함수를 이용하여 6만큼 채워준다.
        // ss를 0 6개로 채운 뒤, i를 넣어준다.
        vstrImageLeft[i] = strPrefixLeft + ss.str() + ".png"; // ./dataset/sequences/00/image_0/004540.png
        vstrImageRight[i] = strPrefixRight + ss.str() + ".png"; // ./dataset/sequences/00/image_1/004540.png
    }
}