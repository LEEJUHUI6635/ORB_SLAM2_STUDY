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


#include "Tracking.h"

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include"ORBmatcher.h"
#include"FrameDrawer.h"
#include"Converter.h"
#include"Map.h"
#include"Initializer.h"

#include"Optimizer.h"
#include"PnPsolver.h"

#include<iostream>

#include<mutex>


using namespace std;

namespace ORB_SLAM2
{

Tracking::Tracking(System *pSys, ORBVocabulary* pVoc, FrameDrawer *pFrameDrawer, MapDrawer *pMapDrawer, Map *pMap, KeyFrameDatabase* pKFDB, const string &strSettingPath, const int sensor):
    mState(NO_IMAGES_YET), mSensor(sensor), mbOnlyTracking(false), mbVO(false), mpORBVocabulary(pVoc),
    mpKeyFrameDB(pKFDB), mpInitializer(static_cast<Initializer*>(NULL)), mpSystem(pSys), mpViewer(NULL),
    mpFrameDrawer(pFrameDrawer), mpMapDrawer(pMapDrawer), mpMap(pMap), mnLastRelocFrameId(0)
{
    // Load camera parameters from settings file

    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    float fps = fSettings["Camera.fps"];
    if(fps==0)
        fps=30;

    // Max/Min Frames to insert keyframes and to check relocalisation
    mMinFrames = 0;
    mMaxFrames = fps;

    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if(DistCoef.rows==5)
        cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;


    int nRGB = fSettings["Camera.RGB"];
    mbRGB = nRGB;

    if(mbRGB)
        cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
        cout << "- color order: BGR (ignored if grayscale)" << endl;

    // Load ORB parameters

    int nFeatures = fSettings["ORBextractor.nFeatures"];
    float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
    int nLevels = fSettings["ORBextractor.nLevels"];
    int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
    int fMinThFAST = fSettings["ORBextractor.minThFAST"];

    mpORBextractorLeft = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::STEREO)
        mpORBextractorRight = new ORBextractor(nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    if(sensor==System::MONOCULAR)
        mpIniORBextractor = new ORBextractor(2*nFeatures,fScaleFactor,nLevels,fIniThFAST,fMinThFAST);

    cout << endl  << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;

    if(sensor==System::STEREO || sensor==System::RGBD)
    {
        mThDepth = mbf*(float)fSettings["ThDepth"]/fx;
        cout << endl << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }

    if(sensor==System::RGBD)
    {
        mDepthMapFactor = fSettings["DepthMapFactor"];
        if(fabs(mDepthMapFactor)<1e-5)
            mDepthMapFactor=1;
        else
            mDepthMapFactor = 1.0f/mDepthMapFactor;
    }

}

void Tracking::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing *pLoopClosing)
{
    mpLoopClosing=pLoopClosing;
}

void Tracking::SetViewer(Viewer *pViewer)
{
    mpViewer=pViewer;
}

// input : left image, right image, timestamp, output : 현재 frame의 world to camera coordinate의 pose
cv::Mat Tracking::GrabImageStereo(const cv::Mat &imRectLeft, const cv::Mat &imRectRight, const double &timestamp)
{
    mImGray = imRectLeft; // left image -> this->mImGray
    cv::Mat imGrayRight = imRectRight; // right image -> imGrayRight

    if(mImGray.channels()==3) // left image의 channel이 3인 경우
    {
        // mbRGB -> color order, true RGB / false BGR
        if(mbRGB) // RGB
        {
            cvtColor(mImGray,mImGray,CV_RGB2GRAY); // left image -> gray
            cvtColor(imGrayRight,imGrayRight,CV_RGB2GRAY); // right image -> gray
        }
        else // BGR
        {
            cvtColor(mImGray,mImGray,CV_BGR2GRAY); // left image -> gray
            cvtColor(imGrayRight,imGrayRight,CV_BGR2GRAY); // right image -> gray
        }
    }
    
    else if(mImGray.channels()==4) // left image의 channel이 4인 경우 -> mask channel(투명도)
    {
        if(mbRGB) // RGB
        {
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY); // left image -> gray
            cvtColor(imGrayRight,imGrayRight,CV_RGBA2GRAY); // right image -> gray
        }
        else // BGR
        {
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY); // left image -> gray
            cvtColor(imGrayRight,imGrayRight,CV_BGRA2GRAY); // right image -> gray
        }
    }

    // mImGray -> left image, imGrayRight -> right image, timestamp -> timestamp,
    // mpORBextractorLeft -> left image에서 추출한 ORB feature, mpORBextractorRight -> right image에서 추출한 ORB feature
    // mpORBVocabulary -> bag of words, mK -> intrinsic parameter, mDistCoef -> distortion coefficients
    // mbf -> Q. baseline, mThDepth -> close point와 far point를 결정짓는 depth threshold
    mCurrentFrame = Frame(mImGray,imGrayRight,timestamp,mpORBextractorLeft,mpORBextractorRight,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    //Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)

    Track();

    return mCurrentFrame.mTcw.clone(); // Tracking -> world to camera coordinate의 pose
    // clone() : 깊은 복사 -> 새롭게 메모리를 할당 받아 같은 값을 넣는다.
    // 얕은 복사 -> 포인터만 복사한다. 결국, 메모리 주소는 같다.
}


cv::Mat Tracking::GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp)
{
    mImGray = imRGB;
    cv::Mat imDepth = imD;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if((fabs(mDepthMapFactor-1.0f)>1e-5) || imDepth.type()!=CV_32F)
        imDepth.convertTo(imDepth,CV_32F,mDepthMapFactor);

    mCurrentFrame = Frame(mImGray,imDepth,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}


cv::Mat Tracking::GrabImageMonocular(const cv::Mat &im, const double &timestamp)
{
    mImGray = im;

    if(mImGray.channels()==3)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGB2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGR2GRAY);
    }
    else if(mImGray.channels()==4)
    {
        if(mbRGB)
            cvtColor(mImGray,mImGray,CV_RGBA2GRAY);
        else
            cvtColor(mImGray,mImGray,CV_BGRA2GRAY);
    }

    if(mState==NOT_INITIALIZED || mState==NO_IMAGES_YET)
        mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);
    else
        mCurrentFrame = Frame(mImGray,timestamp,mpORBextractorLeft,mpORBVocabulary,mK,mDistCoef,mbf,mThDepth);

    Track();

    return mCurrentFrame.mTcw.clone();
}

void Tracking::Track()
{
    if(mState==NO_IMAGES_YET) // reset 된 후,
    {
        mState = NOT_INITIALIZED; // eTrackingState : NO_IMAGES_YET -> NOT_INITIALIZED
    }

    mLastProcessedState=mState; // mState -> mLastProcessedState, Q. 깊은 복사
    // mLastProcessedState = NOT_INITIALIZED

    // Get Map Mutex -> Map cannot be changed
    unique_lock<mutex> lock(mpMap->mMutexMapUpdate); // unique_lock 객체인 lock은 (Map class의 포인터 객체)mpMap의 mutex 객체인 mMutexMapUdate를 소유한다.

    // Initialization
    if(mState==NOT_INITIALIZED) // mState가 eTrackingState : NOT_INITIALIZED 라면,
    {
        if(mSensor==System::STEREO || mSensor==System::RGBD) // System::eSensor : STEREO or RGBD
            StereoInitialization(); // mState=OK
        else // eSensor : MONOCULAR
            MonocularInitialization();

        // initialization을 통해 얻은 map points
        mpFrameDrawer->Update(this); // Tracking class 자신 -> this pointer, mState = OK

        if(mState!=OK) // initialization이 제대로 수행되지 않았을 경우
            return; // 해당 함수를 빠져나가라.
    }
    // Pose estimation
    else // 초기화를 마친 경우 or 초기화를 하지 않아도 되는 경우
    {
        // System is initialized. Track Frame.
        bool bOK;

        // Initial camera pose estimation using motion model or relocalization (if tracking is lost)
        if(!mbOnlyTracking) // mbOnlyTracking = false -> Localization mode가 아닌 경우
        {
            // Local Mapping is activated. This is the normal behaviour, unless
            // you explicitly activate the "only tracking" mode. -> Localization mode

            if(mState==OK) // 이전 frame에서의 tracking이 성공적인 경우
            {
                // Local Mapping might have changed some MapPoints tracked in last frame -> map point, keyframe의 culling 등의 상황
                // Local mapping thread에서 변화한 map point가 있다면, 해당 map point update
                CheckReplacedInLastFrame(); // 지난 frame에서 replaced map points가 존재한다면, 지난 frame의 map points를 replaced map points로 치환한다.

                // mVelocity가 비어있거나, 현재 frame이 마지막으로 relocalization을 수행한 frame에서 3 frame 이상을 지나지 않았을 때,
                // Q. relocalization에서 멀리 떨어지지 않은 시점에서의 mVelocity가 부정확하다고 판단
                if(mVelocity.empty() || mCurrentFrame.mnId<mnLastRelocFrameId+2) // mVelocity.empty() -> initialization 직후
                {
                    // 현재 frame의 keypoint에 해당하는 map point는 알 수 없기 때문에, reference keyframe와의 keypoint association을 이용하여 map point를 구한다.
                    bOK = TrackReferenceKeyFrame(); // 지난 frame에서의 tracking이 성공적이었기 때문에, initial pose estimation(지난 frame의 pose) + map points에 대한 motion-only BA를 통해 현재 frame의 pose를 최적화 한다.
                }
                else // 나머지 경우에는 등속도 운동 model을 따라, 현재 frame의 pose를 측정한다.
                {
                    bOK = TrackWithMotionModel(); // 등속도(relative pose : camera(t-1) to camera(t)) x 지난 frame의 world to camera coordinate의 pose
                    if(!bOK) // bOK = false
                        bOK = TrackReferenceKeyFrame();
                }
            }
            else // mState != OK -> 이전 frame에서의 tracking이 실패한 경우
            {
                bOK = Relocalization();
            }
        }
        else // mbOnlyTracking = true -> Localization mode인 경우
        {
            // Localization Mode: Local Mapping is deactivated

            if(mState==LOST) // 이전 frame에서의 tracking이 실패한 경우
            {
                bOK = Relocalization();
            }
            else // mState != LOST
            {
                if(!mbVO) // mbVO = false -> Localization mode -> 기존의 map points tracking
                {
                    // In last frame we tracked enough MapPoints in the map

                    if(!mVelocity.empty()) // mVelocity.empty() = false
                    {
                        bOK = TrackWithMotionModel();
                    }
                    else // mVelocity.empty() = true
                    {
                        bOK = TrackReferenceKeyFrame();
                    }
                }
                else // mbVO = true -> Visual Odometry mode -> temporal points tracking
                {
                    // In last frame we tracked mainly "visual odometry" points.

                    // We compute two camera poses, one from motion model and one doing relocalization.
                    // If relocalization is sucessfull we choose that solution, otherwise we retain
                    // the "visual odometry" solution.

                    bool bOKMM = false; // motion model
                    bool bOKReloc = false; // relocalization
                    vector<MapPoint*> vpMPsMM; // motion model로 얻은 map points
                    vector<bool> vbOutMM; // motion model로 얻은 outlier 여부
                    cv::Mat TcwMM; // motion model로 얻은 world to camera coordinate의 pose
                    if(!mVelocity.empty()) // mVelocity.empty() = false
                    {
                        bOKMM = TrackWithMotionModel(); // motion model로 camera pose를 구하는 것의 성공 여부
                        vpMPsMM = mCurrentFrame.mvpMapPoints; // 현재 frame의 keypoints에 해당하는 map points -> vpMPsMM
                        vbOutMM = mCurrentFrame.mvbOutlier; // 현재 frame의 keypoints에 해당하는 map points의 outlier 여부 -> vbOutMM
                        TcwMM = mCurrentFrame.mTcw.clone(); // 현재 frame의 world to camera coordinate의 pose에 대한 깊은 복사 -> TcwMM
                    }
                    bOKReloc = Relocalization(); // relocalization으로 camera pose를 구하는 것의 성공 여부

                    if(bOKMM && !bOKReloc) // bOKMM = true and bOKReloc = false -> motion model로 camera pose 계산
                    {
                        mCurrentFrame.SetPose(TcwMM); // input : world to camera coordinate의 pose, output : camera to world coordinate의 pose
                        mCurrentFrame.mvpMapPoints = vpMPsMM;
                        mCurrentFrame.mvbOutlier = vbOutMM;

                        // Visual Odometry mode
                        if(mbVO) // mbVO = true -> map point를 tracking x, temporal points를 tracking
                        {
                            for(int i =0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
                            {
                                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i]) // 현재 frame 상의 i번째 map point가 존재하고 해당 map point가 outlier가 아니라면,
                                {
                                    mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // mnFound += 1 -> Q. Tracking counters
                                }
                            }
                        }
                    }
                    else if(bOKReloc) // bOKReloc = true -> relocalization으로 camera pose 계산
                    {
                        // Visual Odometry mode x
                        mbVO = false; // map point를 tracking -> localization
                    }

                    bOK = bOKReloc || bOKMM; // bOKReloc = true or bOKMM = true -> bOK = true
                }
            }
        }

        mCurrentFrame.mpReferenceKF = mpReferenceKF;

        // If we have an initial estimation of the camera pose and matching. Track the local map.
        if(!mbOnlyTracking) // mbOnlyTracking = false
        {
            if(bOK) // bOK = true
                bOK = TrackLocalMap();
        }
        else // mbOnlyTracking = true -> Localization mode
        {
            // mbVO true means that there are few matches to MapPoints in the map. We cannot retrieve
            // a local map and therefore we do not perform TrackLocalMap(). Once the system relocalizes
            // the camera we will use the local map again.
            if(bOK && !mbVO) // bOK = true and mbVO = false -> map points를 tracking하는 localization mode
                bOK = TrackLocalMap();
        }

        if(bOK) // bOK = true
            mState = OK;
        else // bOK = false
            mState=LOST;

        // Update drawer
        // Pose estimation + Track local map -> 새로 map point가 갱신
        mpFrameDrawer->Update(this);

        // If tracking were good, check if we insert a keyframe -> New Keyframe Decision
        if(bOK) // bOK = true -> track local map 단계까지 성공적인 경우
        {
            // Q. empty() -> 메모리 주소는 할당 받아 있지만, 값이 비어있는 경우?
            // Update motion model
            if(!mLastFrame.mTcw.empty()) // mLastFrame.mTcw.empty() = false -> 지난 frame에서의 tracking이 성공적인 경우
            {
                // last frame의 camera to world coordinate의 pose
                cv::Mat LastTwc = cv::Mat::eye(4,4,CV_32F); // identity matrix
                // Mat::eye(int rows, int cols, int type) -> rows : 새로 만들 행렬의 행 개수, cols : 새로 만들 행렬의 열 개수,
                // type : 새로 만들 행렬의 타입
                mLastFrame.GetRotationInverse().copyTo(LastTwc.rowRange(0,3).colRange(0,3));
                // mLastFrame.GetRotationInverse() -> camera to world coordinate의 rotation
                // src.copyTo(dst) : src 이미지를 dst에 복사, 깊은 복사
                mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0,3).col(3));
                // mLastFrame.GetCameraCenter() -> camera to world coordinate의 translation
                // src.copyTo(dst) : src 이미지를 dst에 복사, 깊은 복사
                mVelocity = mCurrentFrame.mTcw*LastTwc; // relative pose
                // 현재 frame의 world to camera coordinate x 지난 frame의 camera to world coordinate = 지난 frame의 camera to 현재 frame의 camera coordinate
            }
            else // mLastFrame.mTcw.empty() = true -> 지난 frame에서의 tracking이 성공적이지 않은 경우
                mVelocity = cv::Mat(); // mVelocity.empty() = true

            mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw); // 현재 frame의 world to camera coordinate의 pose

            // Clean VO matches
            // Temporal points에 해당하는 mvpMapPoints 삭제
            for(int i=0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 현재 frame 상의 i번째 keypoint에 해당하는 map points -> pMP
                if(pMP) // pMP가 할당되어 있다면,
                    if(pMP->Observations()<1) // 해당 map point가 관측되는 keyframe이 한 개도 없다면,
                    {
                        mCurrentFrame.mvbOutlier[i] = false;
                        mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // Null 값으로 초기화 -> outlier 제거
                    }
            }

            // Delete temporal MapPoints
            for(list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(), lend =  mlpTemporalPoints.end(); lit!=lend; lit++)
            {
                MapPoint* pMP = *lit; // de-reference -> temporal point
                delete pMP; // temporal point 삭제 -> Q. memory 삭제?
            }
            mlpTemporalPoints.clear(); // clear() : list의 모든 값을 삭제 -> Q. 값을 삭제?

            // Check if we need to insert a new keyframe
            if(NeedNewKeyFrame()) // NeedNewKeyFrame() = true -> 현재 frame을 keyframe으로 만들 것인가.
                CreateNewKeyFrame(); // 현재 frame을 새로운 keyframe으로 만들고 local mapping thread에 넘긴다.
                // 현재 frame의 unmatched keypoint에 대하여, depth 정보를 이용하여 새로운 map point를 만들어 보강한다.
            
            // Q. 
            // We allow points with high innovation (considererd outliers by the Huber Function)
            // pass to the new keyframe, so that bundle adjustment will finally decide
            // if they are outliers or not. We don't want next frame to estimate its position
            // with those points so we discard them in the frame.
            for(int i=0; i<mCurrentFrame.N;i++) // 현재 frame의 keypoint 개수만큼 반복
            {
                if(mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
                // 현재 frame의 i번째 keypoint에 해당하는 map point가 존재하고, 해당 map point가 outlier라면,
                    mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 해당 map point는 Null 값으로 초기화 -> outlier 제거
            }
        }

        // Reset if the camera get lost soon after initialization
        // track local map 단계에서 실패한 경우
        if(mState==LOST) // 현재 tracking 상태가 LOST라면,
        {
            if(mpMap->KeyFramesInMap()<=5) // 전체 map에 포함되는 keyframe의 개수가 5개 이하라면,
            {
                cout << "Track lost soon after initialisation, reseting..." << endl;
                mpSystem->Reset(); // mbReset = true
                return; // 해당 함수를 빠져나와라.
            }
        }

        if(!mCurrentFrame.mpReferenceKF)
            mCurrentFrame.mpReferenceKF = mpReferenceKF;

        mLastFrame = Frame(mCurrentFrame);
    } // Tracking의 끝

    // 현재 frame에 대한 tracking 상태 update
    // Store frame pose information to retrieve the complete camera trajectory afterwards.
    if(!mCurrentFrame.mTcw.empty()) // mCurrentFrame.mTcw.empty() = false -> 현재 frame에서의 tracking이 성공적인 경우
    {
        cv::Mat Tcr = mCurrentFrame.mTcw*mCurrentFrame.mpReferenceKF->GetPoseInverse(); // from reference to camera coordinate
        // GetPoseInverse() : camera to world coordinate의 pose
        // 현재 frame의 world to camera coordinate의 pose x 현재 frame에 대한 reference keyframe의 camera to world coordinate의 pose
        // 현재 frame에 대한 reference keyframe의 camera to 현재 frame의 camera coordinate
        mlRelativeFramePoses.push_back(Tcr);
        // mlRelativeFramePoses -> 각 frame과 각 frame에 대한 reference keyframe과의 relative pose를 저장하는 list
        mlpReferences.push_back(mpReferenceKF);
        mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
        mlbLost.push_back(mState==LOST); // mState != LOST -> false
    }
    else // mCurrentFrame.mTcw.empty() = true -> 현재 frame에서의 tracking이 성공적이지 않은 경우
    {
        // This can happen if tracking is lost
        // back() -> 벡터의 마지막 요소를 반환한다.
        mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
        mlpReferences.push_back(mlpReferences.back());
        mlFrameTimes.push_back(mlFrameTimes.back());
        mlbLost.push_back(mState==LOST); // mState = LOST -> true
    }

}


void Tracking::StereoInitialization()
{
    if(mCurrentFrame.N>500) // keypoint의 개수 > 500 -> Q. 현재 frame의 keypoint의 개수가 500을 초과해야 map initialization 수행?
    {
        // Set Frame pose to the origin
        // 현재의 frame을 world origin으로 설정한다.
        mCurrentFrame.SetPose(cv::Mat::eye(4,4,CV_32F)); // cv::Mat::eye() : 대각선을 1로 저장한다. size=(4, 4) -> world origin
        // SetPose의 input : world to camera coordinate의 transformation matrix -> camera to world coordinate의 transformation matrix를 계산한다.

        // Create KeyFrame
        // 현재의 frame을 initialization을 위해서 keyframe으로 설정한다.
        KeyFrame* pKFini = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB); // 동적 할당
        // KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)

        // Insert KeyFrame in the map
        // 현재 keyframe을 전체 map의 Map::mspKeyFrames에 삽입한다.
        mpMap->AddKeyFrame(pKFini);
        
        // 현재 keyframe으로 map point initialization -> stereo camera이기 때문에 triangulation으로 depth 정보를 알 수 있다.
        // Create MapPoints and asscoiate to KeyFrame
        for(int i=0; i<mCurrentFrame.N;i++) // 현재 frame을 keyframe으로 설정하고, keyframe이 가지고 있는 keypoint의 개수만큼 반복
        {
            float z = mCurrentFrame.mvDepth[i]; // 현재 frame에 대한 depth 정보를 vector의 형태로 가지고 있다.
            if(z>0) // monocular keyframe의 경우 depth < 0
            {   
                // 현재 frame이 가지고 있는 keypoint의 pixel 좌표계 상의 값에 camera to world coordinate transformation matrix를 곱하여, world 좌표계 상의 3D point로 변환한다.
                // left image에서 추출한 keypoint + depth 정보 -> map point creation
                cv::Mat x3D = mCurrentFrame.UnprojectStereo(i); // x3D -> 현재 frame의 keypoint(pixel coordinate)에 해당하는 3D map point(world coordinate)
                MapPoint* pNewMP = new MapPoint(x3D,pKFini,mpMap); // 동적 할당, 새로운 map point -> 새로운 객체 생성
                // map point는 3D 좌표도 가지고 있지만, 어떠한 keyframe에서 만들어졌는지에 대한 정보도 가지고 있다.
                // 반복문 내에서 새로운 keypoint가 들어옴에 따라 새로운 map point를 만들어 낸다.
                pNewMP->AddObservation(pKFini,i); // Data Association -> 새로운 map point가 어느 keyframe에서 발견된 몇 번째 keypoint인지 저장한다.
                pKFini->AddMapPoint(pNewMP,i); // 해당 keyframe의 mvpMapPoints vector에 새로운 map point를 포함시킨다.
                pNewMP->ComputeDistinctiveDescriptors(); // 해당 map point가 representative descriptor를 저장한다.
                pNewMP->UpdateNormalAndDepth(); // map point는 max distance와 min distance, mean normal vector를 가지고 있다.
                mpMap->AddMapPoint(pNewMP); // 새로운 map point인 pNewMP를 전체 map의 mspMapPoints에 집어 넣는다.

                mCurrentFrame.mvpMapPoints[i]=pNewMP; // 현재 frame이 가지고 있는 map points <- 새로운 map point인 pNewMP
            }
        }

        // mpMap->MapPointsInMap() : map(Map::mspMapPoints)에 저장되어 있는 map point의 개수 -> Q. 전체 map? local map? A. 전체 map
        // 초기화의 과정이기 때문에, 전체 map은 현재 keyframe으로만 이루어져 있을 것이기 때문
        cout << "New map created with " << mpMap->MapPointsInMap() << " points" << endl;

        // Tracking thread에서 새로운 keyframe을 형성하면, local mapping thread에 넘기는 과정이 수행된다.
        mpLocalMapper->InsertKeyFrame(pKFini); // 현재 keyframe을 local mapping thread에 전달한다.
        
        // Tracking thread에서 처리한 마지막 frame, keyframe
        mLastFrame = Frame(mCurrentFrame);
        mnLastKeyFrameId=mCurrentFrame.mnId; // mnId -> current frame ID
        mpLastKeyFrame = pKFini;

        // for track local map thread
        // Local map
        mvpLocalKeyFrames.push_back(pKFini); // tracking thread의 local keyframes vector에 현재 keyframe을 넣는다.
        mvpLocalMapPoints=mpMap->GetAllMapPoints(); // tracking thread의 mpMap에서 모든 map point를 vector 형태로 받아온다.
        mpReferenceKF = pKFini;
        mCurrentFrame.mpReferenceKF = pKFini;

        mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

        mpMap->mvpKeyFrameOrigins.push_back(pKFini); // 모든 keyframe의 원점

        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw); // 현재 frame의 world to camera coordinate의 pose -> identity matrix

        mState=OK; // eTrackingState : mState
    }
}

void Tracking::MonocularInitialization()
{

    if(!mpInitializer)
    {
        // Set Reference Frame
        if(mCurrentFrame.mvKeys.size()>100)
        {
            mInitialFrame = Frame(mCurrentFrame);
            mLastFrame = Frame(mCurrentFrame);
            mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
            for(size_t i=0; i<mCurrentFrame.mvKeysUn.size(); i++)
                mvbPrevMatched[i]=mCurrentFrame.mvKeysUn[i].pt;

            if(mpInitializer)
                delete mpInitializer;

            mpInitializer =  new Initializer(mCurrentFrame,1.0,200);

            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);

            return;
        }
    }
    else
    {
        // Try to initialize
        if((int)mCurrentFrame.mvKeys.size()<=100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            fill(mvIniMatches.begin(),mvIniMatches.end(),-1);
            return;
        }

        // Find correspondences
        ORBmatcher matcher(0.9,true);
        int nmatches = matcher.SearchForInitialization(mInitialFrame,mCurrentFrame,mvbPrevMatched,mvIniMatches,100);

        // Check if there are enough correspondences
        if(nmatches<100)
        {
            delete mpInitializer;
            mpInitializer = static_cast<Initializer*>(NULL);
            return;
        }

        cv::Mat Rcw; // Current Camera Rotation
        cv::Mat tcw; // Current Camera Translation
        vector<bool> vbTriangulated; // Triangulated Correspondences (mvIniMatches)

        if(mpInitializer->Initialize(mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated))
        {
            for(size_t i=0, iend=mvIniMatches.size(); i<iend;i++)
            {
                if(mvIniMatches[i]>=0 && !vbTriangulated[i])
                {
                    mvIniMatches[i]=-1;
                    nmatches--;
                }
            }

            // Set Frame Poses
            mInitialFrame.SetPose(cv::Mat::eye(4,4,CV_32F));
            cv::Mat Tcw = cv::Mat::eye(4,4,CV_32F);
            Rcw.copyTo(Tcw.rowRange(0,3).colRange(0,3));
            tcw.copyTo(Tcw.rowRange(0,3).col(3));
            mCurrentFrame.SetPose(Tcw);

            CreateInitialMapMonocular();
        }
    }
}

void Tracking::CreateInitialMapMonocular()
{
    // Create KeyFrames
    KeyFrame* pKFini = new KeyFrame(mInitialFrame,mpMap,mpKeyFrameDB);
    KeyFrame* pKFcur = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB);


    pKFini->ComputeBoW();
    pKFcur->ComputeBoW();

    // Insert KFs in the map
    mpMap->AddKeyFrame(pKFini);
    mpMap->AddKeyFrame(pKFcur);

    // Create MapPoints and asscoiate to keyframes
    for(size_t i=0; i<mvIniMatches.size();i++)
    {
        if(mvIniMatches[i]<0)
            continue;

        //Create MapPoint.
        cv::Mat worldPos(mvIniP3D[i]);

        MapPoint* pMP = new MapPoint(worldPos,pKFcur,mpMap);

        pKFini->AddMapPoint(pMP,i);
        pKFcur->AddMapPoint(pMP,mvIniMatches[i]);

        pMP->AddObservation(pKFini,i);
        pMP->AddObservation(pKFcur,mvIniMatches[i]);

        pMP->ComputeDistinctiveDescriptors();
        pMP->UpdateNormalAndDepth();

        //Fill Current Frame structure
        mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
        mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

        //Add to Map
        mpMap->AddMapPoint(pMP);
    }

    // Update Connections
    pKFini->UpdateConnections();
    pKFcur->UpdateConnections();

    // Bundle Adjustment
    cout << "New Map created with " << mpMap->MapPointsInMap() << " points" << endl;

    Optimizer::GlobalBundleAdjustemnt(mpMap,20);

    // Set median depth to 1
    float medianDepth = pKFini->ComputeSceneMedianDepth(2);
    float invMedianDepth = 1.0f/medianDepth;

    if(medianDepth<0 || pKFcur->TrackedMapPoints(1)<100)
    {
        cout << "Wrong initialization, reseting..." << endl;
        Reset();
        return;
    }

    // Scale initial baseline
    cv::Mat Tc2w = pKFcur->GetPose();
    Tc2w.col(3).rowRange(0,3) = Tc2w.col(3).rowRange(0,3)*invMedianDepth;
    pKFcur->SetPose(Tc2w);

    // Scale points
    vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
    for(size_t iMP=0; iMP<vpAllMapPoints.size(); iMP++)
    {
        if(vpAllMapPoints[iMP])
        {
            MapPoint* pMP = vpAllMapPoints[iMP];
            pMP->SetWorldPos(pMP->GetWorldPos()*invMedianDepth);
        }
    }

    mpLocalMapper->InsertKeyFrame(pKFini);
    mpLocalMapper->InsertKeyFrame(pKFcur);

    mCurrentFrame.SetPose(pKFcur->GetPose());
    mnLastKeyFrameId=mCurrentFrame.mnId;
    mpLastKeyFrame = pKFcur;

    mvpLocalKeyFrames.push_back(pKFcur);
    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints=mpMap->GetAllMapPoints();
    mpReferenceKF = pKFcur;
    mCurrentFrame.mpReferenceKF = pKFcur;

    mLastFrame = Frame(mCurrentFrame);

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose());

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    mState=OK;
}

void Tracking::CheckReplacedInLastFrame() // Q. 지난 frame에서의 map point - 현재 frame의 keypoint와의 correspondence
{
    for(int i =0; i<mLastFrame.N; i++) // 지난 frame에서의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // mLastFrame.mvpMapPoints -> 지난 frame 상의 keypoint에 해당하는 map point

        if(pMP) // pMP가 할당되어 있다면,
        {
            MapPoint* pRep = pMP->GetReplaced(); // MapPoint::mpReplaced -> Q.
            if(pRep) // pRep가 할당되어 있다면, (pRep가 Null 값이면 False를 return 한다.)
            {
                mLastFrame.mvpMapPoints[i] = pRep; // replaced map point가 존재한다면, 지난 frame의 map point를 replaced map point로 치환한다.
            }
        }
    }
}


bool Tracking::TrackReferenceKeyFrame()
{
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW(); // 현재 frame의 BoW를 계산한다. -> 현재 frame의 map point와 reference keyframe의 map point와의 correspondence를 빠르게 search하기 위함

    // We perform first an ORB matching with the reference keyframe
    // If enough matches are found we setup a PnP solver
    ORBmatcher matcher(0.7,true); // ORBmatcher(float nnratio, bool checkOri);
    vector<MapPoint*> vpMapPointMatches; // 현재 frame의 map point와 reference keyframe의 map point와의 correspondences

    // 모든 frame에 대한 BoW는 존재하지 않으나, 모든 keyframe에 대한 BoW는 존재한다.
    // 현재 frame의 keypoint에 해당하는 map point는 알 수 없기에, reference keyframe과의 keypoint association을 통해 reference keyframe의 map point를 가지고 온다.
    int nmatches = matcher.SearchByBoW(mpReferenceKF,mCurrentFrame,vpMapPointMatches); // BoW의 두 번째 기능 -> data association
    // vpMapPointMatches : reference keyframe과의 keypoint association을 통해 구한 현재 frame의 map points, nmatches : 현재 frame의 map points 개수
    if(nmatches<15) // 현재 frame의 map point가 15개보다 적다면,
        return false; // bOK = false -> reference keyframe으로 tracking x

    mCurrentFrame.mvpMapPoints = vpMapPointMatches; // keyframe과 공유하고 있는 map points -> 현재 frame의 mvpMapPoints vector
    // 지난 frame에서의 tracking이 성공적이었기 때문에,
    mCurrentFrame.SetPose(mLastFrame.mTcw); // 지난 frame의 world to camera coordinate -> 지난 frame의 camera to world coordinate, 초기화
    Optimizer::PoseOptimization(&mCurrentFrame); // Q. motion-only BA : initial pose estimation(지난 frame의 camera to world coordinate) + map points -> 현재 pose 최적화 

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 frame의 keypoint에 해당하는 map point가 존재한다면,
        {
            if(mCurrentFrame.mvbOutlier[i]) // 현재 frame의 keypoint에 해당하는 map point가 outlier라면,
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // outlier인 map point -> pMP
                // 현재 frame의 map point가 outlier라면, 해당 map point를 Null 값으로 변경
                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // 현재 frame의 map point -> NULL
                mCurrentFrame.mvbOutlier[i]=false; // 현재 frame의 map point는 outlier가 아니다. -> Null 값으로 변경했기 때문
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--; // outlier 제거
            } // 현재 frame의 keypoint에 해당하는 map point가 outlier가 아니라면,
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // map point가 관측되는 keyframe의 개수 > 0
                nmatchesMap++; // outlier를 제거한, 현재 frame의 map point와 겹치는 reference keyframe의 map point의 개수
        }
    }

    // nmatches > nmatchesMap
    // 현재 frame의 map point와 겹치는 reference keyframe의 map point가 10개 이상이라면, reference keyframe을 이용하여 tracking
    return nmatchesMap>=10; // nmatchesMap >= 10 -> true, nmatchesMap < 10 -> false
}

// 지난 frame의 camera to world coordinate의 pose를 얻고, localization mode의 경우 far point는 포함시키지 않고, map point의 개수는 100개로 한정한다.
void Tracking::UpdateLastFrame()
{
    // Update pose according to reference keyframe
    KeyFrame* pRef = mLastFrame.mpReferenceKF; // 지난 frame의 reference keyframe
    cv::Mat Tlr = mlRelativeFramePoses.back(); // last frame과 last frame에 해당하는 reference keyframe 사이의 relative pose -> from reference KF to last F
    // back() : vector의 마지막 요소를 반환한다. -> 가장 최근의 frame과 reference keyframe 사이의 relative pose를 얻고 싶은 경우이기 때문
    mLastFrame.SetPose(Tlr*pRef->GetPose()); // reference keyframe의 world to camera coordinate과 last frame과 reference keyframe과의 relative pose를 통해
    // last frame의 world to camera coordinate을 구할 수 있다.
    // pRef->GetPose() : 지난 frame에 대한 reference keyframe의 world to camera coordinate
    // Tlr*Trw = Tlw -> world to last frame camera coordinate pose
    // SetPose() -> input : world to camera coordinate, output : camera to world coordinate
    // 최종적으로 last frame의 camera to world coordinate의 pose를 구할 수 있다.

    // mnLastKeyFrameId==mLastFrame.mnId -> 지난 frame이 keyframe으로 선정되었을 경우 -> 지난 frame으로 map point를 생성할 수 있기 때문
    // mSensor==System::MONOCULAR -> monocular mode인 경우
    // !mbOnlyTracking -> Localization mode가 아닌 경우
    if(mnLastKeyFrameId==mLastFrame.mnId || mSensor==System::MONOCULAR || !mbOnlyTracking)
        return; // 해당 함수를 빠져나와라.

    // Localization mode인 경우
    // Create "visual odometry" MapPoints
    // We sort points according to their measured depth by the stereo/RGB-D sensor
    vector<pair<float,int> > vDepthIdx;
    // vector<[Type],[Type]> -> 2개의 각각 지정한 타입의 값을 저장한다.
    // 저장한 값은 .first와 .second로 각각 접근할 수 있다.
    // 2개의 연관된 값을 같이 저장할 수 있어서 관리를 용이하게 할 수 있다.
    // 특히, 연관된 2개의 값에서 각각의 조건에 따라 정렬한 결과를 얻고자 할 때 사용하면 좋다.(즉, 2개의 정렬 조건으로 정렬하고 싶은 경우)
    vDepthIdx.reserve(mLastFrame.N); // last frame의 keypoint 개수만큼 vector의 capacity를 설정
    for(int i=0; i<mLastFrame.N;i++) // last frame의 keypoint 개수만큼 반복
    {
        float z = mLastFrame.mvDepth[i]; // 지난 frame의 각 keypoint의 depth
        if(z>0) // monocular가 아닌 경우
        {
            vDepthIdx.push_back(make_pair(z,i)); // 지난 frame의 각 keypoint의 depth + 각 keypoint의 index -> vDepthIdx vector
        }
    }

    if(vDepthIdx.empty()) // vDepthIdx.empty() = true
        return; // 해당 함수를 빠져나가라.

    sort(vDepthIdx.begin(),vDepthIdx.end()); // first를 기준으로 오름차순으로 정렬 -> 지난 frame의 각 keypoint의 depth를 기준으로 오름차순으로 정렬

    // We insert all close points (depth<mThDepth)
    // If less than 100 close points, we insert the 100 closest ones.
    int nPoints = 0;
    for(size_t j=0; j<vDepthIdx.size();j++) // 지난 frame의 keypoint 개수만큼 반복 -> depth가 큰 순서대로
    {
        int i = vDepthIdx[j].second; // depth 값에 대하여 오름차순 정렬된 후의 각 keypoint의 index(depth가 큰 순서대로)

        bool bCreateNew = false;
        
        MapPoint* pMP = mLastFrame.mvpMapPoints[i]; // 지난 frame 상의 keypoint에 해당하는 map point
        if(!pMP) // pMP가 할당되어 있지 않다면,
            bCreateNew = true; // 새로운 map point의 생성과 관련된 flag를 변경
        else if(pMP->Observations()<1) // MapPoint::nObs < 1 -> Q. 지난 frame 상의 keypoint에 해당하는 map point가 없다면 or map point가 관측되는 keyframe이 없다면
        {
            bCreateNew = true; // 새로운 map point의 생성과 관련된 flag를 변경
        }

        if(bCreateNew) // bCreateNew = true -> 새로운 map point를 생성하지만, 만들어진 map point는 Visual Odometry를 수행할 때만 이용되고, 실제 map을 구성하는데 이용되지는 않는다.
        {
            // last frame은 keyframe이 아니어야 한다.
            // keypoint -> map point
            cv::Mat x3D = mLastFrame.UnprojectStereo(i); // 지난 frame 상의 keypoint의 pixel 좌표계 상의 값 -> camera to world coordinate transformation matrix를 곱하여, world 좌표계 상의 3D point로 변환한다.
            MapPoint* pNewMP = new MapPoint(x3D,mpMap,&mLastFrame,i); // MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF)

            mLastFrame.mvpMapPoints[i]=pNewMP;

            // mlpTemporalPoints -> nmatches를 계산하는데 쓰이지만, 
            // local mapping에서는 keyframe이 아닌 frame의 map point는 이용되지 않기 때문에, local mapping thread에 들어가기 전에 삭제된다.
            mlpTemporalPoints.push_back(pNewMP); // pNewMP -> Tracking::mlpTemporalPoints
            nPoints++;
        }
        else // bCreateNew = false
        {
            nPoints++; // nPoints -> 총 map point 개수
        }

        // far point이고, point가 100개를 넘는 경우 -> far point인 경우 depth가 정확하지 않기 때문에
        if(vDepthIdx[j].first>mThDepth && nPoints>100) // 해당 frame에 대한 temporal points는 100개를 넘어야 한다.
        // vDepthIdx[j].first -> depth
        // mThDepth : close와 far points를 결정짓는 depth 값
        // vDepthIdx[j].first > mThDepth -> far points
            break; // for문을 나가라.
    }
}

// 지난 frame까지 tracking이 성공했을 경우, 등속도 운동을 가정하여 현재 frame의 pose estimation
bool Tracking::TrackWithMotionModel()
{
    ORBmatcher matcher(0.9,true); // ORBmatcher(float nnratio, bool checkOri)

    // Update last frame pose according to its reference keyframe
    // Create "visual odometry" points if in Localization Mode
    UpdateLastFrame(); // 지난 frame의 camera to world coordinate의 pose를 얻고, Localization mode의 경우 Visual Odometry를 위한 temporal map points(실제로 map을 구성 x)를 생성한다.

    // 등속도 운동을 가정하여, 지난 frame의 pose에서 현재 frame의 pose estimation
    mCurrentFrame.SetPose(mVelocity*mLastFrame.mTcw); // 등속도(relative pose : camera(t-1) to camera(t)) x 지난 frame의 world to camera coordinate의 pose
    // SetPose() -> input : world to camera coordinate, output : camera to world coordinate
    fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL));
    // fill 함수는 어떤 연속성을 띈 자료구조(벡터 혹은 배열)의 시작점부터 연속된 범위를 어떤 값이나 객체로 모두 지정하고 싶을 때 사용하는 함수
    // fill(ForwardIterator first, ForwardIterator last, const T& val)
    // first -> 채우고자 하는 자료구조의 시작위치 iterator, last -> 채우고자 하는 자료구조의 끝위치 iterator(last는 포함x)
    // val -> first부터 last 전까지 채우고자 하는 값으로 어떤 객체나 자료형을 넘겨줘도 템플릿 T에 의해 가능하다.
    // 현재 frame의 map point의 처음과 끝을 모두 NULL 값으로 초기화한다.

    // Project points seen in previous frame
    int th;
    if(mSensor!=System::STEREO)
        th=15;
    else // mSensor==System::STEREO -> stereo의 경우
        th=7;
    // 현재 frame의 keypoint에 해당하는 map point는 알지 못한다. 따라서, 지난 frame의 map point를 현재 frame에 reprojection 하여, 현재 frame의 map point를 구한다.
    int nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR);

    // If few matches, uses a wider window search
    if(nmatches<20) // 지난 frame을 현재 frame에 reprojection 하여, grid search를 통해 구한 현재 frame의 map point 개수
    {
        // fill 함수는 어떤 연속성을 띈 자료구조(벡터나 배열 같은)의 시작점부터 연속된 범위를 어떤 값이나 객체로 모두 지정하고 싶을 때 사용하는 함수이다.
        // fill(ForwardIterator first, ForwardIterator last, const T& val)
        // first : 채우고자 하는 자료구조의 시작위치 iterator, last : 채우고자 하는 자료구조의 끝위치 iterator이며 last는 포함 x,
        // val : first부터 last 전까지 채우고자 하는 값으로 어떤 객체나 자료형을 넘겨줘도 템플릿 T에 의해서 가능
        // 다시 현재 frame의 모든 map points 초기화
        fill(mCurrentFrame.mvpMapPoints.begin(),mCurrentFrame.mvpMapPoints.end(),static_cast<MapPoint*>(NULL)); // 현재 frame의 mvpMapPoints를 모두 Null 값으로 초기화
        nmatches = matcher.SearchByProjection(mCurrentFrame,mLastFrame,2*th,mSensor==System::MONOCULAR);
        // float radius = th*CurrentFrame.mvScaleFactors[nLastOctave] -> threshold를 2배로 하여 window의 size를 2배로 늘린다.
    }

    if(nmatches<20) // wider window search를 수행했음에도 적은 matches를 가진다면,
        return false; // false를 return

    // Optimize frame pose with all matches
    Optimizer::PoseOptimization(&mCurrentFrame); // Q. motion-only BA : initial camera pose + map points

    // Discard outliers
    int nmatchesMap = 0;
    for(int i =0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 frame의 keypoint에 해당하는 map point가 존재한다면,
        {
            if(mCurrentFrame.mvbOutlier[i]) // 현재 frame의 keypoint에 해당하는 map point가 outlier라면,
            {
                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // outlier라고 판단되는, 현재 frame의 keypoint에 해당하는 map point -> pMP

                mCurrentFrame.mvpMapPoints[i]=static_cast<MapPoint*>(NULL); // outlier이기 때문에 Null 값으로 초기화
                mCurrentFrame.mvbOutlier[i]=false;
                pMP->mbTrackInView = false;
                pMP->mnLastFrameSeen = mCurrentFrame.mnId;
                nmatches--; // outlier map point이기 때문
            }
            else if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // outlier가 아니라고 판단되고, 해당 map point와 일치하는 keyframe의 keypoint 개수가 0보다 크다면,
                nmatchesMap++;
        }
    }    

    // nmatches = nmatchesMap + temporal points
    if(mbOnlyTracking) // mbOnlyTracking = true -> Localization mode
    {
        mbVO = nmatchesMap<10; // map point와 일치하는 keypoint의 개수 < 10 -> mbVO = true
        return nmatches>20; // nmatches > 20 -> return true, nmatches <= 20 -> return false
    }

    // nmatchesMap >= 10 -> return true, nmatchesMap < 10 -> return false
    return nmatchesMap>=10;
}

// Track Local Map : initial camera pose와 correspondence를 알고 있기 때문에, local map을 현재 frame에 투영시키면 더 많은 map point correspondence를 찾을 수 있다.
bool Tracking::TrackLocalMap()
{
    // We have an estimation of the camera pose and some map points tracked in the frame.
    // We retrieve the local map and try to find matches to points in the local map.

    UpdateLocalMap(); // Local keyframes + Local map points

    SearchLocalPoints(); // local map points를 현재 frame에 투영시켜, unmatch된 map points에 대한 correspondence를 찾는다.

    // Optimize Pose
    // local map을 현재 frame에 투영시켜 더 많은 correspondence를 찾고, motion-only BA를 통해 보다 정확한 현재 frame의 pose를 얻는다.
    Optimizer::PoseOptimization(&mCurrentFrame); // Q. motion-only BA -> 현재 frame의 initial pose + 현재 frame의 keypoint - map point correspondences
    mnMatchesInliers = 0;

    // Update MapPoints Statistics
    for(int i=0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 frame 상의 i번째 keypoint에 해당하는 map point가 존재한다면,
        {
            if(!mCurrentFrame.mvbOutlier[i]) // 현재 frame 상의 i번째 keypoint에 해당하는 map point가 outlier가 아니라면,
            {
                mCurrentFrame.mvpMapPoints[i]->IncreaseFound(); // default -> n = 1, mnFound++, Tracking counters(tracking에 몇 번 이용되었는가)
                if(!mbOnlyTracking) // mbOnlyTracking = false
                {
                    if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) // 현재 frame의 i번째 keypoint에 해당하는 map point가 하나 이상의 keyframe에서 관측되는 경우
                        mnMatchesInliers++; // 무조건 map points를 tracking 해야하기 때문
                }
                else // mbOnlyTracking = true -> Localization mode
                    mnMatchesInliers++; // 무조건 map points를 tracking 하지 않아도 되기 때문
            }
            else if(mSensor==System::STEREO) // 현재 frame 상의 i번째 keypoint에 해당하는 map point가 outlier이고, stereo sensor를 이용한다면,
                mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL); // Null 값으로 초기화

        }
    }

    // Decide if the tracking was succesful
    // More restrictive if there was a relocalization recently
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && mnMatchesInliers<50) // Q. 
        return false;

    // local map을 현재 frame에 투영시켰을 때, 얻은 point를 포함한 총 map point의 개수가 30 미만이라면,
    if(mnMatchesInliers<30)
        return false;
    else // mnMatchesInliers >= 30
        return true;
}

bool Tracking::NeedNewKeyFrame()
{
    if(mbOnlyTracking) // mbOnlyTracking = true -> Localization mode
        return false; // Localization mode는 Local mapping 과정을 거치지 않는다.

    // If Local Mapping is freezed by a Loop Closure do not insert keyframes
    // mpLocalMapper->istStopped() = true or mpLocalMapper->stopRequested() = true
    if(mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
        return false; // Local mapping thread가 중지되거나 중지 요청을 받으면, keyframe insertion을 수행할 수 없다.

    const int nKFs = mpMap->KeyFramesInMap(); // 전체 map에 포함되는 keyframe의 개수

    // Do not insert keyframes if not enough frames have passed from last relocalisation
    // mMinFrames = 0, mMaxFrames = fps(frame per second)
    // 현재 frame이 최근의 relocalization에서 얼마 지나지 않았고, 전체 map의 keyframe 수가 fps보다 크다면,
    if(mCurrentFrame.mnId<mnLastRelocFrameId+mMaxFrames && nKFs>mMaxFrames)
        return false; // keyframe이 아예 없는 정도가 아니고, relocalization에서 얼마 지나지 않았다면,

    // Tracked MapPoints in the reference keyframe
    int nMinObs = 3; // reference keyframe(현재 frame 기준)이 tracking 하는 map points가 발견되어야 하는 keyframe의 수
    if(nKFs<=2) // 전체 map에 포함되는 keyframe의 개수가 2 이하라면,
        nMinObs=2; // reference keyframe이 tracking 하는 map points가 발견되어야 하는 keyframe의 수
    int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs); // nMinObs 이상의 keyframe에서 발견되는 reference keyframe의 map points

    // Local Mapping accept keyframes?
    bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames(); // return mbAcceptKeyFrames

    // Check how many "close" points are being tracked and how many could be potentially created.
    int nNonTrackedClose = 0;
    int nTrackedClose= 0;
    if(mSensor!=System::MONOCULAR) // monocular sensor를 이용하는 것이 아니라면,
    {
        for(int i =0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
        {
            if(mCurrentFrame.mvDepth[i]>0 && mCurrentFrame.mvDepth[i]<mThDepth) // close points
            // 0 < 현재 frame의 i번째 keypoint의 depth < mThDepth
            {
                if(mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
                // 현재 frame의 i번째 keypoint에 해당하는 map point가 존재하고, 해당 map point가 outlier가 아니라면,
                    nTrackedClose++; // close points tracking
                else
                    nNonTrackedClose++; // close points no tracking
            }
        }
    }

    // close points를 더 삽입해야 하는가에 대한 flag
    bool bNeedToInsertClose = (nTrackedClose<100) && (nNonTrackedClose>70);
    // close points tracking < 100 and close points no tracking > 70 -> bNeedToInsertClose = true
    // close points tracking >= 100 or close points no tracking <= 70 -> bNeedToInsertClose = false

    // Thresholds
    float thRefRatio = 0.75f;
    if(nKFs<2) // 전체 map에 포함되어 있는 keyframe의 개수가 2 미만이라면,
        thRefRatio = 0.4f; // 전체 map을 이루고 있는 keyframe의 개수가 작기 때문에, keyframe insertion이 필수적

    if(mSensor==System::MONOCULAR)
        thRefRatio = 0.9f;

    // Condition 1a: More than "MaxFrames" have passed from last keyframe insertion
    const bool c1a = mCurrentFrame.mnId>=mnLastKeyFrameId+mMaxFrames;
    // mMaxFrames = fps(frame per second) -> 현재 frame은 최근의 keyframe insertion에서 몇 frame 정도 지나있어야 한다.
    // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
    const bool c1b = (mCurrentFrame.mnId>=mnLastKeyFrameId+mMinFrames && bLocalMappingIdle);
    // mMinFrames = 0, 현재 frame은 최근의 keyframe insertion보다 같거나 이상이며, Local mapping thread가 keyframe을 받아들일 수 있어야 한다.
    //Condition 1c: tracking is weak
    const bool c1c =  mSensor!=System::MONOCULAR && (mnMatchesInliers<nRefMatches*0.25 || bNeedToInsertClose) ;
    // monocular sensor를 이용하지 않는 경우 -> 현재 frame의 inlier의 개수(track local map 이후)가 nRefMatches(nMinObs 이상의 keyframe에서 발견되는 reference keyframe의 map points = reference keyframe의 의미 있는 map points)
    // 의 25 퍼센트 아래이거나, tracking 할 수 있는 close map points가 적은 경우
    // Condition 2: Few tracked points compared to reference keyframe. Lots of visual odometry compared to map matches.
    const bool c2 = ((mnMatchesInliers<nRefMatches*thRefRatio|| bNeedToInsertClose) && mnMatchesInliers>15);
    // 현재 frame의 inlier의 개수(track local map 이후)가 15를 넘는 경우 -> 현재 frame의 inlier의 개수가 nRefMatches(nMinsObs 이상의 keyframe에서 발견되는 reference keyframe의 map points = reference keyframe의 의미 있는 map points)
    // 의 75 퍼센트 아래이거나, tracking 할 수 있는 close map points가 적은 경우

    if((c1a||c1b||c1c)&&c2) // (c1a = true or c1b = true or c1c = true) and c2 = true
    {
        // If the mapping accepts keyframes, insert keyframe.
        // Otherwise send a signal to interrupt BA
        if(bLocalMappingIdle) // bLocalMappingIdle = true -> Local mapping thread가 keyframe을 받아들일 수 있는 상태라면,
        {
            return true;
        }
        else // bLocalMappingIdle = false -> Local mapping thread가 Local BA로 인해 keyframe을 받아들일 수 없는 상태라면,
        {
            mpLocalMapper->InterruptBA(); // mbAbortBA = true -> Local mapping thread의 Local BA를 중지하는 flag 변경
            if(mSensor!=System::MONOCULAR) // monocular sensor를 이용하지 않는다면,
            {
                if(mpLocalMapper->KeyframesInQueue()<3) // local mapping thread의 처리 대상인 새로운 keyframe의 개수가 3개 미만이라면,
                    return true;
                else // mpLocalMapper->KeyframesInQueue() >= 3 -> local mapping thread의 처리 대상인 새로운 keyframe의 개수가 3개 이상이라면,
                    return false;
            }
            else
                return false;
        }
    }
    else
        return false;
}

void Tracking::CreateNewKeyFrame()
{
    // Q. Local mapping thread가 멈추면(Loop closing 등의 이유로 인해), 새로운 keyframe을 받아들일 수 없다.
    if(!mpLocalMapper->SetNotStop(true)) // mpLocalMapper->SetNotStop(true) = false -> mbStopped = true
        return; // 해당 함수를 빠져나가라.

    // 현재 frame -> 새로운 keyframe
    KeyFrame* pKF = new KeyFrame(mCurrentFrame,mpMap,mpKeyFrameDB); // KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB)

    mpReferenceKF = pKF;
    mCurrentFrame.mpReferenceKF = pKF;

    // 현재 frame의 map points를 보강 -> 현재 frame의 unmatched keypoints에 해당하는 map points를 depth 정보로 만들어 낸다.
    if(mSensor!=System::MONOCULAR) // monocular sensor를 이용하지 않는다면,
    {
        mCurrentFrame.UpdatePoseMatrices(); // world to camera coordinate의 pose -> camera to world coordinate의 pose

        // We sort points by the measured depth by the stereo/RGBD sensor.
        // We create all those MapPoints whose depth < mThDepth.
        // If there are less than 100 close points we create the 100 closest.
        vector<pair<float,int> > vDepthIdx; // float -> depth 값, int -> keypoint index
        // pair<[Type],[Type]> : 2개의 각각 지정한 타입의 값을 저장한다. 저장한 값은 first와 second로 각각 접근할 수 있다.
        // 2개의 연관된 값을 같이 저장할 수 있어 관리를 용이하게 할 수 있다. 특히, 연관된 2개의 값에서 각각의 조건에 따라 정렬한 결과를 얻고자 할 때
        // 사용하면 좋다.(2개의 정렬 조건으로 정렬하고 싶은 경우)
        vDepthIdx.reserve(mCurrentFrame.N); // 현재 frame의 keypoint 개수만큼 reserve(초기화x, capacity만 확보)
        for(int i=0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
        {
            float z = mCurrentFrame.mvDepth[i]; // 현재 frame의 i번째 keypoint에 해당하는 map point의 depth
            if(z>0) // monocular가 아닌 경우
            {
                vDepthIdx.push_back(make_pair(z,i)); // depth와 keypoint index -> vDepthIdx vector
            }
        }

        if(!vDepthIdx.empty()) // vDepthIdx.empty() = false
        {
            sort(vDepthIdx.begin(),vDepthIdx.end()); // depth를 기준으로 오름차순으로 정렬

            int nPoints = 0;
            for(size_t j=0; j<vDepthIdx.size();j++) // keypoint의 개수만큼 반복
            {
                int i = vDepthIdx[j].second; // 오름차순으로 정렬된 depth에 해당하는 keypoint의 index

                bool bCreateNew = false;

                MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // i번째 keypoint(오름차순으로 정렬된 depth)에 해당하는 map point
                if(!pMP) // 해당 map point가 존재하지 않으면,
                    bCreateNew = true;
                else if(pMP->Observations()<1) // 해당 map point가 관측되는 keyframe이 없다면,
                {
                    bCreateNew = true;
                    mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL); // 해당 map point는 Null 값으로 초기화
                }

                // 새로운 map point를 만들어라.
                if(bCreateNew) // bCreateNew = true
                {
                    cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
                    // keypoint의 pixel 좌표계 상의 값 -> camera to world coordinate transformation matrix를 곱하여, world 좌표계 상의 3D point로 변환한다.
                    MapPoint* pNewMP = new MapPoint(x3D,pKF,mpMap);
                    pNewMP->AddObservation(pKF,i); // Data Association -> 새로운 map point가 어느 keyframe에서 발견된 몇 번째 keypoint인지 저장한다.
                    pKF->AddMapPoint(pNewMP,i); // 해당 map point를 해당 keyframe의 mvpMapPoints vector에 귀속시킨다.
                    pNewMP->ComputeDistinctiveDescriptors(); // 해당 map point가 representative descriptor(hamming distance가 가장 작은 descriptor)를 저장한다.
                    pNewMP->UpdateNormalAndDepth(); // 해당 map point의 normal vector를 저장한다.
                    mpMap->AddMapPoint(pNewMP); // pNewMP -> Map::mspMapPoints

                    mCurrentFrame.mvpMapPoints[i]=pNewMP; // pNewMP -> 현재 frame의 mvpMapPoints
                    nPoints++;
                }
                else // bCreateNew = false
                {
                    nPoints++;
                }

                if(vDepthIdx[j].first>mThDepth && nPoints>100) // depth > mThDepth : far points -> far points의 depth 정보는 부정확할 수 있기 때문
                    break; // 해당 for문을 빠져나가라.
            }
        }
    }

    mpLocalMapper->InsertKeyFrame(pKF);
    // 현재 keyframe을 local mapping thread의 처리 대상인 new keyframe list에 추가하고, 
    // local mapping thread가 새로운 keyframe을 받아드릴 때, local BA를 중지하도록 flag를 변경한다.

    mpLocalMapper->SetNotStop(false); // mpLocalMapper->SetNotStop(false) = true, Q. mbNotStop = false -> Local mapping thread를 멈추지 않게 한다.

    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKF;
}

void Tracking::SearchLocalPoints()
{
    // Do not search map points already matched
    // 현재 frame 상의 keypoint와 association 관계를 가지는 map point들에 대해 반복
    for(vector<MapPoint*>::iterator vit=mCurrentFrame.mvpMapPoints.begin(), vend=mCurrentFrame.mvpMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; // de-reference -> map point
        if(pMP) // 해당 map point가 존재한다면,
        {
            if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
            {
                *vit = static_cast<MapPoint*>(NULL); // outlier 제거
            }
            else // pMP->isBad() = false -> 해당 map point가 나쁘지 않다고 판단되면,
            {
                pMP->IncreaseVisible(); // default -> n = 1, mnVisible++
                pMP->mnLastFrameSeen = mCurrentFrame.mnId; // 현재 frame 상의 keypoint와 association 관계가 있는 map point라면, 더 이상 search하지 않기 위함
                pMP->mbTrackInView = false;
            }
        }
    }

    int nToMatch=0;

    // Project points in frame and check its visibility
    for(vector<MapPoint*>::iterator vit=mvpLocalMapPoints.begin(), vend=mvpLocalMapPoints.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; // de-reference -> local map point
        if(pMP->mnLastFrameSeen == mCurrentFrame.mnId) // 현재 frame 상의 keypoint와 association 관계가 있는 map point라면, search에서 제외하라.
            continue; // 해당 루프의 끝으로 이동한다.
        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
            continue; // 해당 루프의 끝으로 이동한다.
        // Project (this fills MapPoint variables for matching)
        if(mCurrentFrame.isInFrustum(pMP,0.5)) // 해당 local map point가 현재 frame의 frustum 안에 존재한다면,
        {
            pMP->IncreaseVisible(); // default -> n = 1, mnVisible++, pMP->IncreaseVisible() >= 2
            nToMatch++; // 현재 frame의 frustum 안에 존재하는 map point 개수(단, 현재 frame 상의 keypoint와 association 관계가 있으면 안된다. -> 이미 correspondence를 찾았기 때문)
        }
    }

    // 현재 frame과 match되지 않은 map point들을 현재 frame에 투영시켜, 더 많은 correspondence를 찾는다.
    if(nToMatch>0)
    {
        ORBmatcher matcher(0.8);
        int th = 1;
        if(mSensor==System::RGBD)
            th=3;
        // If the camera has been relocalised recently, perform a coarser search
        // 현재 frame이 최근의 relocalization으로부터 3 frame 이상 지났지 못했을 경우
        if(mCurrentFrame.mnId<mnLastRelocFrameId+2)
            th=5; // search window size를 키운다. -> Q.
        // local map points를 현재 frame에 투영시켜 unmatch된 map point에 대한 더 많은 correspondence를 찾는다.
        matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th);
    }
}

// Local keyframes + Local map points
void Tracking::UpdateLocalMap()
{
    // This is for visualization -> Q.
    mpMap->SetReferenceMapPoints(mvpLocalMapPoints); // mpMap : 전체 map, SetReferenceMapPoints(mvpLocalMapPoints) : local map points -> reference map points

    // Update
    // Local Map의 기준
    // 1) K1 : 현재 frame과 map points를 공유하고 있는 keyframes
    // 2) K2 : covisibility graph 상에서 neighbors인 keyframes
    // 3) K1에 속하는 keyframe 중 현재 frame과 map points를 가장 많이 공유하는 keyframe -> reference keyframe
    UpdateLocalKeyFrames(); // Local Keyframe : 현재 frame과 map points를 공유하고 있는 keyframes + covisibility graph 상에서의 neighbor keyframes
    UpdateLocalPoints(); // Local Keyframe에 있는 모든 map points
}

void Tracking::UpdateLocalPoints()
{
    mvpLocalMapPoints.clear(); // 기존에 있던 mvpLocalMapPoints 삭제

    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        KeyFrame* pKF = *itKF; // de-reference -> Local keyframe
        const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // local keyframe의 map points

        for(vector<MapPoint*>::const_iterator itMP=vpMPs.begin(), itEndMP=vpMPs.end(); itMP!=itEndMP; itMP++)
        {
            MapPoint* pMP = *itMP; // de-reference -> local keyframe의 map point
            if(!pMP) // 해당 map point가 존재하지 않는다면,
                continue; // 해당 루프의 끝으로 이동한다.
            if(pMP->mnTrackReferenceForFrame==mCurrentFrame.mnId) // 같은 map point가 중복되어 mvpLocalPoints에 들어가는 것을 방지
                continue; // 해당 루프의 끝으로 이동한다.
            if(!pMP->isBad()) // 해당 map point가 나쁘지 않다고 판단되면,
            {
                mvpLocalMapPoints.push_back(pMP);
                pMP->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 중복 방지
            }
        }
    }
}

// Local Keyframe : 현재 frame과 map points를 공유하고 있는 keyframes + covisibility graph 상에서의 neighbor keyframes
void Tracking::UpdateLocalKeyFrames()
{
    // Each map point vote for the keyframes in which it has been observed
    map<KeyFrame*,int> keyframeCounter; // 현재 frame 상의 map point들은 어떠한 keyframe에서 관측되고, 각 keyframe에서 몇 번 관측 되는가.
    // map : map은 각 노드가 key와 value 쌍으로 이루어진 트리 구조로 중복을 허용하지 않는다.
    // first-key, second-value
    for(int i=0; i<mCurrentFrame.N; i++) // 현재 frame의 keypoint 개수만큼 반복
    {
        if(mCurrentFrame.mvpMapPoints[i]) // 현재 frame 상의 i번째 map point가 존재한다면,
        {
            MapPoint* pMP = mCurrentFrame.mvpMapPoints[i]; // 현재 frame 상의 i번째 map point -> pMP
            if(!pMP->isBad()) // 해당 map point가 나쁘지 않다고 판단되면,
            {
                const map<KeyFrame*,size_t> observations = pMP->GetObservations(); // key : keyframe, value : keypoint index
                for(map<KeyFrame*,size_t>::const_iterator it=observations.begin(), itend=observations.end(); it!=itend; it++)
                    keyframeCounter[it->first]++; // it->first : keyframe, keyframeCounter[keyframe]++
                    // 현재 frame의 map point가 다른 keyframe에서 몇 번 관측되는가.
            }
            else // pMP->isBad() = true -> 해당 map point가 나쁘다고 판단되면,
            {
                mCurrentFrame.mvpMapPoints[i]=NULL; // outlier 제거
            }
        }
    }

    // keyframeCounter -> 현재 frame의 map point들은 어떠한 keyframe에서 몇 번 관측되는가.

    if(keyframeCounter.empty()) // keyframeCounter.empty() = true
        return; // 해당 함수를 빠져나가라.

    int max=0;
    KeyFrame* pKFmax= static_cast<KeyFrame*>(NULL); // 현재 frame과 가장 많이 map point를 공유하는 keyframe -> Null 값으로 초기화

    mvpLocalKeyFrames.clear(); // 기존에 있던 local keyframe 제거
    // clear() : vector에 저장된 값들은 제거되지만, vector에 할당된 메모리는 삭제되지 않는다.
    mvpLocalKeyFrames.reserve(3*keyframeCounter.size()); // 현재 frame과 map point를 공유하는 모든 keyframe의 개수 x 3 
    // reserve -> capacity, resize -> size(초기화 포함)

    // All keyframes that observe a map point are included in the local map. Also check which keyframe shares most points
    for(map<KeyFrame*,int>::const_iterator it=keyframeCounter.begin(), itEnd=keyframeCounter.end(); it!=itEnd; it++)
    {
        KeyFrame* pKF = it->first; // it->first : keyframe

        if(pKF->isBad()) // 해당 keyframe이 나쁘다고 판단되면,
            continue; // 해당 루프의 끝으로 이동한다.

        if(it->second>max) // it->second : 현재 frame의 map point가 각 keyframe에서 관측되는 횟수
        {
            max=it->second;
            pKFmax=pKF; // 현재 frame과 가장 많이 map point를 공유하는 keyframe
        }

        mvpLocalKeyFrames.push_back(it->first); // it->first : keyframe, 현재 frame과 map point를 공유하는 keyframe(나쁘다고 판단되는 keyframe 제외) -> mvpLocalKeyFrames
        pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId; // 중복 방지
    }


    // Include also some not-already-included keyframes that are neighbors to already-included keyframes
    for(vector<KeyFrame*>::const_iterator itKF=mvpLocalKeyFrames.begin(), itEndKF=mvpLocalKeyFrames.end(); itKF!=itEndKF; itKF++)
    {
        // Limit the number of keyframes
        if(mvpLocalKeyFrames.size()>80) // local keyframe의 개수가 80 이상이 되지 않도록 한다.
            break; // 해당 루프를 종료한다.

        KeyFrame* pKF = *itKF; // de-reference -> keyframe

        const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10); // 해당 local keyframe의 10개 이하의 neighbor keyframes를 추출

        for(vector<KeyFrame*>::const_iterator itNeighKF=vNeighs.begin(), itEndNeighKF=vNeighs.end(); itNeighKF!=itEndNeighKF; itNeighKF++)
        {
            KeyFrame* pNeighKF = *itNeighKF; // de-reference -> neighbor keyframe
            if(!pNeighKF->isBad()) // 해당 neighbor keyframe이 나쁘지 않다고 판단되면,
            {
                // 해당 local keyframe의 neighbor keyframe -> mvpLocalKeyFrames vector
                if(pNeighKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 중복 방지
                {
                    mvpLocalKeyFrames.push_back(pNeighKF); // 같은 neighbor keyframe이 중복되어 mvpLocalKeyFrames에 들어가는 것을 방지
                    pNeighKF->mnTrackReferenceForFrame=mCurrentFrame.mnId;
                    break; // 해당 루프를 종료한다. -> Q. 해당 local keyframe의 neighbor keyframes 중, 가장 큰 edge weight(해당 local keyframe 사이의)를 가진 keyframe을 넣는 것인가?
                }
            }
        }

        const set<KeyFrame*> spChilds = pKF->GetChilds(); // 해당 local keyframe의 mspChildrens set
        for(set<KeyFrame*>::const_iterator sit=spChilds.begin(), send=spChilds.end(); sit!=send; sit++)
        {
            KeyFrame* pChildKF = *sit; // de-reference -> 해당 local keyframe의 child keyframe
            if(!pChildKF->isBad()) // 해당 child keyframe이 나쁘지 않다고 판단되면,
            {
                // 해당 local keyframe의 child keyframe -> mvpLocalKeyFrames vector
                if(pChildKF->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 중복 방지
                {
                    mvpLocalKeyFrames.push_back(pChildKF);
                    pChildKF->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 같은 child keyframe이 중복되어 mvpLocalKeyFrames에 들어가는 것을 방지
                    break; // 해당 루프를 종료한다. -> Q. 해당 local keyframe의 child keyframes 중, 가장 큰 edge weight(해당 local keyframe 사이의)를 가진 keyframe을 넣는 것인가?
                }
            }
        }

        KeyFrame* pParent = pKF->GetParent(); // 해당 local keyframe의 mpParent keyframe
        if(pParent) // pParent keyframe이 존재한다면,
        {
            // 해당 local keyframe의 parent keyframe -> mvpLocalKeyFrames vector
            if(pParent->mnTrackReferenceForFrame!=mCurrentFrame.mnId) // 중복 방지
            {
                mvpLocalKeyFrames.push_back(pParent);
                pParent->mnTrackReferenceForFrame=mCurrentFrame.mnId; // 같은 parent keyframe이 중복되어 mvpLocalKeyFrames에 들어가는 것을 방지
                break; // 해당 루프를 종료한다.
            }
        }

    }

    if(pKFmax) // pKFmax keyframe이 존재한다면,
    {
        mpReferenceKF = pKFmax; // 현재 frame과 가장 많이 map point를 공유하는 keyframe -> reference keyframe
        mCurrentFrame.mpReferenceKF = mpReferenceKF;
    }
}

// 이전 frame에서 tracking이 실패하였을 경우
bool Tracking::Relocalization()
{
    // Compute Bag of Words Vector
    mCurrentFrame.ComputeBoW(); // 현재 frame의 BoW를 계산한다. -> 현재 frame의 BoW vector, feature vector 추출

    // Relocalization is performed when tracking is lost
    // Track Lost: Query KeyFrame Database for keyframe candidates for relocalisation
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
    // mpKeyFrameDB -> keyframe의 BoW로 만든 recognition database
    

    if(vpCandidateKFs.empty())
        return false;

    const int nKFs = vpCandidateKFs.size(); // relocalization keyframe candidates의 개수

    // We perform first an ORB matching with each candidate
    // If enough matches are found we setup a PnP solver
    // PnP solver -> 2D-3D matching
    ORBmatcher matcher(0.75,true); // ORBmatcher(float nnratio, bool checkOri)

    // 각 relocalization keyframe마다 PnP solver가 존재한다.
    vector<PnPsolver*> vpPnPsolvers;
    vpPnPsolvers.resize(nKFs); // relocalization keyframe candidates의 개수로 resize

    vector<vector<MapPoint*> > vvpMapPointMatches; // 이중 벡터
    vvpMapPointMatches.resize(nKFs); // relocalization keyframe candidates의 개수로 resize

    vector<bool> vbDiscarded; // 해당 keyframe을 버릴 것인가에 대한 bool type의 vector
    vbDiscarded.resize(nKFs); // relocalization keyframe candidates의 개수로 resize

    int nCandidates=0;

    for(int i=0; i<nKFs; i++) // relocalization keyframe candidates의 개수만큼 반복
    {
        KeyFrame* pKF = vpCandidateKFs[i]; // i번째의 relocalization candidate keyframe
        if(pKF->isBad()) // pKF->isBad() = true
            vbDiscarded[i] = true; // i번째의 relocalization candidate keyframe discard 여부 = true
        else // pKF->isBad() = false
        {
            // 현재 frame의 map point는 알 수 없다. i번째 relocalization candidate keyframe의 keypoint와 현재 frame의 keypoint와의 association 관계(BoW의 두 번째 기능)를 통해, 현재 frame의 keypoint에 해당하는 map point를 얻을 수 있다.
            int nmatches = matcher.SearchByBoW(pKF,mCurrentFrame,vvpMapPointMatches[i]); // nmatches -> SearchByBoW를 통해 구한 현재 frame의 map points
        
            if(nmatches<15) // i번째 relocalization keyframe candidate keyframe의 map point와 유사한 현재 frame의 map point가 15개 미만이라면,
            {
                vbDiscarded[i] = true; // i번째의 relocalization candidate keyframe discard 여부 = true
                continue; // 루프의 끝으로 이동한다. -> i번째의 relocalization candidate keyframe은 고려 x
            }
            // nmatches >= 15
            else // i번째 relocalization keyframe candidate keyframe의 map point와 유사한 현재 frame의 map point가 15개 이상이라면,
            {
                // 현재 frame의 2D points - i번째 relocalization keyframe candidate keyframe의 map point와 유사한 현재 frame의 map points(3D)
                PnPsolver* pSolver = new PnPsolver(mCurrentFrame,vvpMapPointMatches[i]); // PnPsolver(const Frame &F, const vector<MapPoint*> &vpMapPointMatches)
                pSolver->SetRansacParameters(0.99,10,300,4,0.5,5.991); // RANSAC parameter setting
                vpPnPsolvers[i] = pSolver;
                nCandidates++; // 충분한 nmatches를 가져 버려지지 않은 relocalization candidate keyframe의 개수
            }
        }
    }

    // Alternatively perform some iterations of P4P RANSAC
    // Until we found a camera pose supported by enough inliers
    bool bMatch = false;
    ORBmatcher matcher2(0.9,true); // ORBmatcher(float nnratio, bool checkOri)

    while(nCandidates>0 && !bMatch) // nCandidates > 0 and true
    {
        for(int i=0; i<nKFs; i++) // relocalization keyframe candidates의 개수만큼 반복
        {
            if(vbDiscarded[i]) // vbDiscarded[i] = true
                continue; // 루프의 끝으로 이동한다.

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            PnPsolver* pSolver = vpPnPsolvers[i]; // 해당 keyframe의 PnP solver -> keyframe마다 PnP solver는 달라진다.
            cv::Mat Tcw = pSolver->iterate(5,bNoMore,vbInliers,nInliers); // 보류
            // input : 현재 frame의 2D points - i번째 relocalization keyframe candidate keyframe의 map point와 유사한 현재 frame의 map points(3D)
            // output : world to camera coordinate의 pose
            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore) // bNoMore = true
            {
                vbDiscarded[i]=true; // i번째 relocalization keyframe candidate keyframe을 discard
                nCandidates--;
            }

            // If a Camera Pose is computed, optimize
            if(!Tcw.empty()) // Tcw가 할당되어 있다면,
            {
                Tcw.copyTo(mCurrentFrame.mTcw); // src.copyTo(dst) : src를 dst에 복사 -> 깊은 복사
                // PnP solver로 구한 world to camera coordinate의 pose -> 현재 frame의 world to camera coordinate의 pose

                set<MapPoint*> sFound;
                // set : 노드 기반 컨테이너이며, 균형 이진트리로 구현되어 있다. key라 불리는 원소들의 집합으로 이루어진 컨테이너이다.
                // key 값은 중복되지 않는다. 원소가 insert 멤버 함수에 의해 삽입이 되면, 원소는 자동으로 정렬 된다.

                const int np = vbInliers.size(); // keypoint의 개수

                // RANSAC 이후의 inlier에 대하여 map point 추가
                for(int j=0; j<np; j++) // 하나의 keyframe 내의 keypoint의 개수
                {
                    if(vbInliers[j]) // j번째 keypoint가 inlier라면,
                    {
                        // vvpMapPointMatches[i] -> i번째 relocalization candidate keyframe의 map point와 유사한 현재 frame의 map points(3D)
                        // vvpMapPointMatches[i][j] -> i번째 keyframe의 map points와 유사한 현재 frame의 map points 중 j번째 point
                        mCurrentFrame.mvpMapPoints[j]=vvpMapPointMatches[i][j];
                        sFound.insert(vvpMapPointMatches[i][j]); // i번째 keyframe의 map points와 유사한 현재 frame의 map points 중 j번째 point -> sFound set
                    }
                    else // j번째 keypoint가 inlier가 아니라면,
                        mCurrentFrame.mvpMapPoints[j]=NULL; // outlier 제거
                }
                
                // nGood = nInitialCorrespondences - nBad
                int nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 보류

                if(nGood<10) // optimization 후의 inliers
                    continue; // 루프의 끝으로 이동한다.
                
                // outlier 제거
                for(int io =0; io<mCurrentFrame.N; io++) // 현재 frame의 keypoint 개수만큼 반복
                    if(mCurrentFrame.mvbOutlier[io]) // 현재 frame의 io번째 map point가 outlier라면,
                        mCurrentFrame.mvpMapPoints[io]=static_cast<MapPoint*>(NULL); // 현재 frame의 io번째 map point를 Null 값으로 초기화하라.

                // Guided Search
                // If few inliers, search by projection in a coarse window and optimize again
                if(nGood<50)
                {
                    // i번째 relocalization candidate keyframe의 map points를 현재 frame에 projection
                    // sFound -> i번째 relocalization candidate keyframe의 map points와 겹치는 현재 frame의 map points -> already found
                    // (이미 correspondence를 가지는) sFound를 제외하고, unmatched keypoints에 해당하는 map points를 찾는다.
                    int nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100);

                    if(nadditional+nGood>=50) // 이전의 matches + 추가적으로 찾은 matches
                    {
                        // nGood = nInitialCorrespondences - nBad;
                        nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 보류

                        // If many inliers but still not enough, search by projection again in a narrower window
                        // the camera has been already optimized with many points
                        if(nGood>30 && nGood<50) // optimization 이후의 inlier의 개수가 30 초과 50 미만이라면,
                        {
                            sFound.clear(); // i번째 keyframe의 map points와 유사한 현재 frame의 map points 중 j번째 point, already found
                            // 이미 존재하던 database 삭제
                            // sFound 갱신
                            for(int ip =0; ip<mCurrentFrame.N; ip++) // 현재 frame의 keypoint 개수만큼 반복
                                if(mCurrentFrame.mvpMapPoints[ip]) // 현재 frame 상의 ip번째 map point가 존재한다면,
                                    sFound.insert(mCurrentFrame.mvpMapPoints[ip]); // 현재 frame 상의 ip번째 map point -> sFound
                            nadditional =matcher2.SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,3,64);

                            // Final optimization
                            if(nGood+nadditional>=50)
                            {
                                nGood = Optimizer::PoseOptimization(&mCurrentFrame); // 보류

                                for(int io =0; io<mCurrentFrame.N; io++) // 현재 frame의 keypoint 개수만큼 반복
                                    if(mCurrentFrame.mvbOutlier[io]) // 현재 frame 상의 io번째 map point가 outlier라면,
                                        mCurrentFrame.mvpMapPoints[io]=NULL; // outlier 제거
                            }
                        }
                    }
                }


                // If the pose is supported by enough inliers stop ransacs and continue
                if(nGood>=50)
                {
                    bMatch = true; // bMatch = true -> while문 종료
                    break; // 루프를 종료하고, 루프 밖의 첫 번째 문에서 실행을 계속한다. -> for문 종료
                }
            }
        }
    }

    if(!bMatch) // bMatch = false -> relocalization 실패한 경우
    {
        return false;
    }
    else // bMatch = true -> relocalization 성공한 경우
    {
        mnLastRelocFrameId = mCurrentFrame.mnId;
        return true;
    }

}

void Tracking::Reset()
{
    cout << "System Reseting" << endl;
    // mpViewer가 동작하지 않게끔 한다.
    if(mpViewer) // mpViewer가 false가 아니라면, pointer -> mpViewer의 memory가 할당되어 있다면, -> Null 값(nullptr, 가리키는 변수가 없다면)이면 False로 동작
    {
        mpViewer->RequestStop(); // mbStopRequested = true;
        while(!mpViewer->isStopped()) // mbStopped = false
            usleep(3000); // mpViewer가 아직 중지되지 않았다면, 중지되기까지 tracking thread를 반복적으로 일시중지한다.
    }

    // Reset Local Mapping
    cout << "Reseting Local Mapper...";
    mpLocalMapper->RequestReset(); // mbResetRequested = false가 되기 전까지 local mapping thread는 계속 대기한다.
    cout << " done" << endl;

    // Reset Loop Closing
    cout << "Reseting Loop Closing...";
    mpLoopClosing->RequestReset(); // mbResetrequested = false가 되기 전까지 loop closing thread는 계속 대기한다.
    cout << " done" << endl;

    // Clear BoW Database
    cout << "Reseting Database...";
    mpKeyFrameDB->clear(); // keyframe 관련 bag of words를 삭제한다.
    cout << " done" << endl;

    // Clear Map (this erase MapPoints and KeyFrames)
    mpMap->clear(); // mpMap(전체 map)에 저장되어 있는 MapPoints와 KeyFrames를 모두 지운다.

    // keyframe, frame의 index 초기화
    KeyFrame::nNextId = 0;
    Frame::nNextId = 0;
    mState = NO_IMAGES_YET; // eTrackingState : NO_IMAGES_YET -> image가 안들어오는 상태로 초기화

    if(mpInitializer) // monocular인 경우에만 해당 -> mpInitializer에 대한 메모리가 할당되어 있다면, monocular의 경우일 것이다.
    {
        delete mpInitializer; // Q. de-reference를 해서 delete하는 것과 포인터를 그대로 delete하는 것은 어떠한 차이가 있는가?
        mpInitializer = static_cast<Initializer*>(NULL);
    }

    mlRelativeFramePoses.clear(); // 상대적인 pose
    mlpReferences.clear(); // reference keyframe
    mlFrameTimes.clear(); // timestamps
    mlbLost.clear(); // tracking에 실패했는지에 대한 여부

    if(mpViewer) // Q. mpViewer가 메모리를 할당 받는다면, -> 예외 처리
        mpViewer->Release(); // reset이 완료되면, mpViewer의 동작을 재게하도록 flag를 변경한다.
}

void Tracking::ChangeCalibration(const string &strSettingPath)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
    float fx = fSettings["Camera.fx"];
    float fy = fSettings["Camera.fy"];
    float cx = fSettings["Camera.cx"];
    float cy = fSettings["Camera.cy"];

    cv::Mat K = cv::Mat::eye(3,3,CV_32F);
    K.at<float>(0,0) = fx;
    K.at<float>(1,1) = fy;
    K.at<float>(0,2) = cx;
    K.at<float>(1,2) = cy;
    K.copyTo(mK);

    cv::Mat DistCoef(4,1,CV_32F);
    DistCoef.at<float>(0) = fSettings["Camera.k1"];
    DistCoef.at<float>(1) = fSettings["Camera.k2"];
    DistCoef.at<float>(2) = fSettings["Camera.p1"];
    DistCoef.at<float>(3) = fSettings["Camera.p2"];
    const float k3 = fSettings["Camera.k3"];
    if(k3!=0)
    {
        DistCoef.resize(5);
        DistCoef.at<float>(4) = k3;
    }
    DistCoef.copyTo(mDistCoef);

    mbf = fSettings["Camera.bf"];

    Frame::mbInitialComputations = true;
}

// Local Mapping thread가 비활성화되어, 오로지 tracking thread만 실행하고자 할 때, flag를 받아 변경하는 함수
void Tracking::InformOnlyTracking(const bool &flag)
{
    mbOnlyTracking = flag;
}



} //namespace ORB_SLAM
