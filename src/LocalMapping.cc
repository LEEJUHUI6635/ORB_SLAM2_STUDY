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

#include "LocalMapping.h"
#include "LoopClosing.h"
#include "ORBmatcher.h"
#include "Optimizer.h"

#include<mutex>

namespace ORB_SLAM2
{

LocalMapping::LocalMapping(Map *pMap, const float bMonocular):
    mbMonocular(bMonocular), mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mbAbortBA(false), mbStopped(false), mbStopRequested(false), mbNotStop(false), mbAcceptKeyFrames(true)
{
}

void LocalMapping::SetLoopCloser(LoopClosing* pLoopCloser)
{
    mpLoopCloser = pLoopCloser;
}

void LocalMapping::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LocalMapping::Run()
{

    mbFinished = false;

    while(1)
    {
        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(false); // mbAcceptKeyFrames = false -> 새로운 keyframe insertion과 관련된 flag 변경
        // queue에 있는 Tracking thread에 존재하는 keyframe을 처리하기 위해, 더 이상의 keyframe을 받지 않는다.

        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames()) // Tracking thread가 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame(); // 현재 keyframe(tracking thread -> local mapping thread)의 모든 map point들에 대한 update, covisibility graph 상의 edge update, 현재 keyframe -> 전체 map

            // Check recent MapPoints
            MapPointCulling();
            // 1) 해당 map point는 그것이 발견될 수 있는 모든 keyframe 중 25% 이상을 tracking 해야 한다.
            // 2) 해당 map point가 생성된 후, 세 개 이상의 keyframe에서 발견되어야 한다. <-> map point가 세 개 이상의 keyframe에서 발견되지 못한다면, 언제든지 삭제될 수 있다.

            // Triangulate new MapPoints
            CreateNewMapPoints();

            if(!CheckNewKeyFrames())
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors();
            }

            mbAbortBA = false;

            if(!CheckNewKeyFrames() && !stopRequested())
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2)
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap); // 보류

                // Check redundant local Keyframes
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop())
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish())
            {
                usleep(3000);
            }
            if(CheckFinish())
                break;
        }

        ResetIfRequested();

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true);

        if(CheckFinish())
            break;

        usleep(3000);
    }

    SetFinish();
}

void LocalMapping::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexNewKFs); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexNewKFs를 소유한다.
    mlNewKeyFrames.push_back(pKF); // 현재 keyframe을 local mapping thread의 처리 대상인 new keyframe list에 추가한다.
    mbAbortBA=true; // local mapping thread가 새로운 keyframe을 받아드릴 때, local BA를 중지하도록 flag를 변경한다.
}

bool LocalMapping::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexNewKFs); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexNewKFs를 소유한다.
    return(!mlNewKeyFrames.empty()); // mlNewKeyFrames.empty() = true -> return false, mlNewKeyFrames.empty() = false -> return true
    // Tracking thread에서 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
}

// 현재 keyframe(tracking thread -> local mapping thread)의 모든 map point들에 대한 update, covisibility graph 상의 edge update, 현재 keyframe -> 전체 map
void LocalMapping::ProcessNewKeyFrame()
{
    // Critical Section
    {
        unique_lock<mutex> lock(mMutexNewKFs); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexNewKFs를 소유한다.
        mpCurrentKeyFrame = mlNewKeyFrames.front(); // front() : 벡터의 첫 번째 요소를 반환한다.
        // Tracking thread가 생성한 새로운 keyframe list의 첫 번째 요소 -> mpCurrentKeyFrame
        mlNewKeyFrames.pop_front(); // pop_front() : 리스트 제일 앞의 원소 삭제
        // Local mapping thread의 처리 대상인 새로운 keyframe list에서 현재 처리 중인 keyframe을 삭제한다.
    }

    // Compute Bags of Words structures
    mpCurrentKeyFrame->ComputeBoW(); // 현재 keyframe의 BoW vector 혹은 Feature vector가 존재하지 않는다면, 다시 계산한다.

    // Associate MapPoints to the new keyframe and update normal and descriptor
    const vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 keyframe의 keypoint와 association 관계를 가지는 모든 map point

    for(size_t i=0; i<vpMapPointMatches.size(); i++) // Q. Null 값도 포함할 것이기 때문에, 현재 keyframe의 keypoint 개수와 같을 것이다.
    {
        MapPoint* pMP = vpMapPointMatches[i]; // 현재 keyframe의 i번째 keypoint와 association 관계를 가지는 map point -> pMP
        if(pMP) // pMP가 존재한다면,
        {
            if(!pMP->isBad()) // 해당 map point가 나쁘지 않다고 판단되면,
            {
                // Q. tracking thread에서 local mapping thread로 현재 frame을 keyframe으로 만들어 넘겨줄 때, depth 정보를 이용하여 correspondence가 없는 keypoint에 대하여 생성한 map point?
                if(!pMP->IsInKeyFrame(mpCurrentKeyFrame)) // Q. 해당 map point가 현재 keyframe에서 발견되었다면 true를 return, 발견되지 않았다면 false를 return
                // pMP->IsInKeyFrame(mpCurrentKeyFrame) = false -> 해당 map point가 현재 keyframe에서 발견되지 않았다면,
                {
                    pMP->AddObservation(mpCurrentKeyFrame, i); // Data Association -> 해당 map point가 현재 keyframe에서 발견된 i번째 keypoint인지 저장한다.
                    pMP->UpdateNormalAndDepth(); // 해당 map point의 min distance, max distance와 normal vector를 계산한다.
                    pMP->ComputeDistinctiveDescriptors(); // 해당 map point의 representative descriptor(다른 descriptor와의 hamming distance가 가장 작은 descriptor)를 저장한다.
                }
                else // this can only happen for new stereo points inserted by the Tracking -> Q.
                // pMP->IsInKeyFrame(mpCurrentKeyFrame) = true -> 해당 map point가 현재 keyframe에서 발견되었다면,
                {
                    // Q. tracking thread에서 local mapping thread로 현재 frame을 keyframe으로 만들어 넘겨줄 때, 이미 keypoint와 association 관계가 있던 map point?
                    mlpRecentAddedMapPoints.push_back(pMP);
                }
            }
        }
    }    

    // Update links in the Covisibility Graph
    mpCurrentKeyFrame->UpdateConnections(); 
    // 현재 keyframe의 15개 이상의 map point들이 관측된 keyframe, 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수(>= 15) = weight -> parent keyframe, child keyframe update

    // Insert Keyframe in Map
    mpMap->AddKeyFrame(mpCurrentKeyFrame); // mspKeyFrames.insert(mpCurrentKeyFrame)
}

void LocalMapping::MapPointCulling()
{
    // Check Recent Added MapPoints
    list<MapPoint*>::iterator lit = mlpRecentAddedMapPoints.begin(); // 현재 keyframe(tracking thread -> local mapping thread) 상의 keypoint와 association 관계를 갖는 map point
    const unsigned long int nCurrentKFid = mpCurrentKeyFrame->mnId; // 현재 keyframe id

    int nThObs;
    if(mbMonocular) // mbMonocular = true
        nThObs = 2;
    else // mbMonocular = false
        nThObs = 3;
    const int cnThObs = nThObs;

    while(lit!=mlpRecentAddedMapPoints.end()) // 현재 keyframe(tracking thread -> local mapping thread) 상의 keypoint와 association 관계를 갖는 모든 map point에 대하여 반복
    {
        MapPoint* pMP = *lit; // de-reference -> map point
        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
        {
            lit = mlpRecentAddedMapPoints.erase(lit); // mlpRecentAddedMapPoints에서 해당 map point를 삭제
            // erase() : 벡터 v에서 i번째 원소를 삭제, erase 함수의 인자는 지우고 싶은 원소의 주소이다.
        }
        // tracking을 할 때, 해당 map point는 그것이 발견될 수 있는 모든 keyframe 중 25% 이상에서 발견되어야 한다.
        // mnFound -> tracking에서 해당 map point가 이용되는 횟수
        // mnVisible -> 해당 map point가 관측되는 횟수
        else if(pMP->GetFoundRatio()<0.25f ) // mnFound/mnVisible -> 해당 map point가 관측될 수 있는 모든 keyframe 중 실제로 tracking에 이용되는 map point 비율 < 0.25f
        {
            pMP->SetBadFlag(); // 해당 map point가 관측되는 keyframe과 전체 map에서 해당 map point를 삭제한다.
            lit = mlpRecentAddedMapPoints.erase(lit); // mlpRecentAddedMapPoints에서 해당 map point를 삭제
        }
        // 해당 map point는 만들어진 후, 적어도 세 개 이상의 keyframe에서 발견되어야 한다.
        // ex) nCurrentKFid = 2, pMP->mnFirstKFid = 0 -> 해당 map point는 0, 1, 2에서 발견될 수 있다.
        // 현재 keyframe id - 해당 map point가 처음 만들어질 때의 keyframe id >= 2, 해당 map point가 관측되는 keyframe <= 3(현재 keyframe 제외)
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=2 && pMP->Observations()<=cnThObs)
        {
            pMP->SetBadFlag(); // 해당 map point가 관측되는 keyframe과 전체 map에서 해당 map point를 삭제한다.
            lit = mlpRecentAddedMapPoints.erase(lit); // mlpRecentAddedMapPoints에서 해당 map point를 삭제
        }
        // 현재 keyframe id - 해당 map point가 처음 만들어질 때의 keyframe id >= 3 -> Q. 
        else if(((int)nCurrentKFid-(int)pMP->mnFirstKFid)>=3)
            lit = mlpRecentAddedMapPoints.erase(lit); // mlpRecentAddedMapPoints에서 해당 map point를 삭제
        else
            lit++; // lit++ -> 다음의 map point
    }
}

void LocalMapping::CreateNewMapPoints()
{
    // Retrieve neighbor keyframes in covisibility graph
    int nn = 10;
    if(mbMonocular) // mbMonocular = true
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); // 현재 keyframe과 covisibility graph 상의 10개 이하의 neighbor keyframes

    ORBmatcher matcher(0.6,false);

    cv::Mat Rcw1 = mpCurrentKeyFrame->GetRotation(); // 현재 keyframe의 world to camera coordinate의 rotation
    cv::Mat Rwc1 = Rcw1.t(); // 현재 keyframe의 camera to world coordinate의 rotation
    cv::Mat tcw1 = mpCurrentKeyFrame->GetTranslation(); // 현재 keyframe의 world to camera coordinate의 translation
    cv::Mat Tcw1(3,4,CV_32F); // Mat(int rows, int cols, int type) -> [3, 4] matrix
    Rcw1.copyTo(Tcw1.colRange(0,3)); // 현재 keyframe의 world to camera coordinate의 rotation -> Tcw1
    tcw1.copyTo(Tcw1.col(3)); // 현재 keyframe의 world to camera coordinate의 translation -> Tcw1
    cv::Mat Ow1 = mpCurrentKeyFrame->GetCameraCenter(); // world 좌표계 상에서의 현재 keyframe의 위치, camera to world coordinate의 translation

    // intrinsic parameter
    const float &fx1 = mpCurrentKeyFrame->fx;
    const float &fy1 = mpCurrentKeyFrame->fy;
    const float &cx1 = mpCurrentKeyFrame->cx;
    const float &cy1 = mpCurrentKeyFrame->cy;
    const float &invfx1 = mpCurrentKeyFrame->invfx;
    const float &invfy1 = mpCurrentKeyFrame->invfy;

    const float ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor;

    int nnew=0;

    // Search matches with epipolar restriction and triangulate
    // 현재 keyframe의 covisibility graph 상의 10개 이하의 neighbor keyframes 개수만큼 반복
    for(size_t i=0; i<vpNeighKFs.size(); i++)
    {
        // 하나 이상의 neighbor keyframe을 처리한 후, tracking thread에서 생성한 새로운 keyframe이 존재하면,
        if(i>0 && CheckNewKeyFrames()) // CheckNewKeyFrames() -> Tracking thread에서 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
            return; // 해당 함수를 빠져나가라.

        KeyFrame* pKF2 = vpNeighKFs[i]; // 현재 keyframe의 covisibility graph 상의 i번째 neighbor keyframe

        // Check first that baseline is not too short
        // parallax가 너무 작으면, 정확도가 높은 3D point를 생성할 수 없기 때문
        cv::Mat Ow2 = pKF2->GetCameraCenter(); // world 좌표계 상에서의 neighbor keyframe의 위치, camera to world coordinate의 translation
        cv::Mat vBaseline = Ow2-Ow1; // neighbor keyframe - 현재 keyframe baseline
        const float baseline = cv::norm(vBaseline);

        if(!mbMonocular) // mbMonocular = false
        {
            if(baseline<pKF2->mb) // neighbor keyframe과 현재 keyframe 사이의 거리 < neighbor keyframe의 baseline
            continue; // 해당 루프의 끝으로 이동한다.
        }
        else // mbMonocular = true
        {
            const float medianDepthKF2 = pKF2->ComputeSceneMedianDepth(2);
            const float ratioBaselineDepth = baseline/medianDepthKF2;

            if(ratioBaselineDepth<0.01)
                continue;
        }

        // Compute Fundamental Matrix
        cv::Mat F12 = ComputeF12(mpCurrentKeyFrame,pKF2); // F = K1^(-1) x t12 x R12 x K2^(-1)
        // R12, t12 : neighbor keyframe to 현재 keyframe coordinate의 rotation, translation

        // Search matches that fullfil epipolar constraint
        vector<pair<size_t,size_t> > vMatchedIndices; // 현재 keyframe의 i번째 keypoint index, 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false);

        cv::Mat Rcw2 = pKF2->GetRotation(); // keyframe 2(neighbor keyframe)의 world to camera coordinate의 rotation
        cv::Mat Rwc2 = Rcw2.t(); // keyframe 2(neighbor keyframe)의 camera to world coordinate의 rotation
        cv::Mat tcw2 = pKF2->GetTranslation(); // keyframe 2(neighbor keyframe)의 world to camera coordinate의 translation
        cv::Mat Tcw2(3,4,CV_32F); // keyframe 2(neighbor keyframe)의 world to camera coordinate의 transformation
        Rcw2.copyTo(Tcw2.colRange(0,3));
        tcw2.copyTo(Tcw2.col(3));

        // intrinsic parameter
        const float &fx2 = pKF2->fx;
        const float &fy2 = pKF2->fy;
        const float &cx2 = pKF2->cx;
        const float &cy2 = pKF2->cy;
        const float &invfx2 = pKF2->invfx;
        const float &invfy2 = pKF2->invfy;

        // Triangulate each match
        const int nmatches = vMatchedIndices.size(); // 현재 keyframe의 i번째 keypoint index, 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first; // 현재 keyframe의 i번째 keypoint index
            const int &idx2 = vMatchedIndices[ikp].second; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index

            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1]; // 현재 keyframe의 i번째 keypoint -> left image
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1]; // 현재 keyframe의 i번째 keypoint -> right image
            bool bStereo1 = kp1_ur>=0; // kp1_ur >= 0 -> bStereo1 = true

            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2]; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint -> left image
            const float kp2_ur = pKF2->mvuRight[idx2]; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint -> right image
            bool bStereo2 = kp2_ur>=0; // kp2_ur >= 0 -> bStereo2 = true

            // Check parallax between rays
            // 현재 keyframe
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0); // pixel 좌표계 -> normalized plane의 metric 좌표계
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0); // pixel 좌표계 -> normalized plane의 metric 좌표계

            cv::Mat ray1 = Rwc1*xn1; // 
            cv::Mat ray2 = Rwc2*xn2; // 
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2));

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1)
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1]));
            else if(bStereo2)
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2]));

            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2);

            cv::Mat x3D;
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                cv::Mat A(4,4,CV_32F);
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);

                x3D = vt.row(3).t();

                if(x3D.at<float>(3)==0)
                    continue;

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3);

            }
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1);                
            }
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2);
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t();

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2);
            if(z1<=0)
                continue;

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2);
            if(z2<=0)
                continue;

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave];
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0);
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1);
            const float invz1 = 1.0/z1;

            if(!bStereo1)
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else
            {
                float u1 = fx1*x1*invz1+cx1;
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                float errX1_r = u1_r - kp1_ur;
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1)
                    continue;
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave];
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0);
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1);
            const float invz2 = 1.0/z2;
            if(!bStereo2)
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else
            {
                float u2 = fx2*x2*invz2+cx2;
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2)
                    continue;
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1;
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2;
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue;

            const float ratioDist = dist2/dist1;
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor)
                continue;

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);

            pMP->AddObservation(mpCurrentKeyFrame,idx1);            
            pMP->AddObservation(pKF2,idx2);

            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors();

            pMP->UpdateNormalAndDepth();

            mpMap->AddMapPoint(pMP);
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++;
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular)
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn);
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId)
            continue;
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId;

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5);
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2;
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
                continue;
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit;

        matcher.Fuse(pKFi,vpMapPointMatches);
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size());

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++)
    {
        KeyFrame* pKFi = *vitKF;

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches();

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++)
        {
            MapPoint* pMP = *vitMP;
            if(!pMP)
                continue;
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId)
                continue;
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId;
            vpFuseCandidates.push_back(pMP);
        }
    }

    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches();
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i];
        if(pMP)
        {
            if(!pMP->isBad())
            {
                pMP->ComputeDistinctiveDescriptors();
                pMP->UpdateNormalAndDepth();
            }
        }
    }

    // Update connections in covisibility graph
    mpCurrentKeyFrame->UpdateConnections();
}

cv::Mat LocalMapping::ComputeF12(KeyFrame *&pKF1, KeyFrame *&pKF2)
{
    cv::Mat R1w = pKF1->GetRotation(); // keyframe 1의 world to camera coordinate의 rotation
    cv::Mat t1w = pKF1->GetTranslation(); // keyframe 1의 world to camera coordinate의 translation
    cv::Mat R2w = pKF2->GetRotation(); // keyframe 2의 world to camera coordinate의 rotation
    cv::Mat t2w = pKF2->GetTranslation(); // keyframe 2의 world to camera coordinate의 translation

    cv::Mat R12 = R1w*R2w.t(); // keyframe 1의 world to camera coordinate의 rotation x keyframe 2의 camera to world coordinate
    // keyframe 2의 camera to keyframe 1의 camera coordinate의 rotation
    cv::Mat t12 = -R1w*R2w.t()*t2w+t1w; // -R2w.t()*t2w -> keyframe 2의 camera to world coordinate의 translation
    // keyframe 1의 world to camera coordinate의 transformation x keyframe 2의 camera to world coordinate의 translation
    // keyframe 2의 camera to keyframe 1의 camera coordinate의 translation

    cv::Mat t12x = SkewSymmetricMatrix(t12); // keyframe 2의 camera to keyframe 1의 camera coordinate의 translation vector -> translation matrix

    const cv::Mat &K1 = pKF1->mK; // keyframe 1의 intrinsic parameter
    const cv::Mat &K2 = pKF2->mK; // keyframe 2의 intrinsic parameter


    return K1.t().inv()*t12x*R12*K2.inv(); // F = K1^(-1) x t12 x R12 x K2^(-1)
}

// 새로운 keyframe이 들어오면, Local Mapping thread에게 Local BA를 중단하라는 메시지를 보내는 함수
// 특정 thread가 특정 영역에 들어가서 mutex를 lock 하면, 다른 thread는 특정 영역에 들어올 수 없고 대기한다.
void LocalMapping::RequestStop()
{
    // 기본적으로 생성과 동시에 lock이 걸리고 소멸시에 unlock되지만 그 밖에도 옵션을 통해 생성시 lock을 안 시키고 따로 특정 시점에 lock을 걸 수 있다.
    // unique_lock은 호출시 uniqueLock.lock()을 호출하기 전까지 lock이 걸리는 것을 미룬다.
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexStop을 소유한다.
    // unique_lock class 객체인 lock은 mutex 객체인 mMutexStop을 소유하고 있다.
    // Q. lock 객체는 mMutexStop에 대하여 lock을 걸어준 것인가?
    mbStopRequested = true; // LocalMapping class에 있는 RequestStop() 함수가 선언되면, mbStopRequested = true
    
    // 아래 조건은 새로운 keyframe이 들어왔을 때, lock2 객체는 mMutexNewKFs라는 mutex 객체를 소유하게 되고, 아래 코드가 critical section으로 동작하게 된다.
    unique_lock<mutex> lock2(mMutexNewKFs); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexNewKFs를 소유한다.
    mbAbortBA = true;
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop);
    if(mbStopRequested && !mbNotStop)
    {
        mbStopped = true;
        cout << "Local Mapping STOP" << endl;
        return true;
    }

    return false;
}

bool LocalMapping::isStopped()
{
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock이 mutex 객체인 mMutexStop을 소유한다.
    return mbStopped; // Local mapping이 중지 되었는가에 대한 flag
}

bool LocalMapping::stopRequested()
{
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexStop을 소유한다.
    return mbStopRequested; // Local mapping이 중지 요청을 받았는가에 대한 flag
}

// Q. Tracking thread에서 새로운 keyframe이 들어오면, Local Mapping을 중지한다면, 새로운 keyframe을 받아들이고 난 후에는 Local Mapping을 재게해야 한다.
// Localization mode가 비활성화되면, 
void LocalMapping::Release()
{
    // mMutexStop과 mMutexFinish 모두에 해당하는 상황에서만 Release() 함수에 접근할 수 있다.
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock이 mutex 객체인 mMutexStop을 소유하고 있다.
    unique_lock<mutex> lock2(mMutexFinish); // unique_lock class의 객체인 lock2가 mutex 객체인 mMutexFinish를 소유하고 있다.
    if(mbFinished) // mbFinished = true -> Local Mapping이 끝났는지에 대한 flag
        return; // Local Mapping이 끝났다면, 해당 함수를 나가라.
    mbStopped = false; // Local Mapping의 중지와 관련된 flag를 false로 변경
    mbStopRequested = false; // Local Mapping을 중지하라는 요청과 관련된 flag를 false로 변경
    
    // list container는 sequence container의 일종이므로 순서를 유지하는 구조이다. vector와는 다르게 멤버 함수에서 정렬(sort, merge), 이어붙이기(splice)가 있다.
    // 원소를 탐색할 때, 임의접근 반복자(at(), [])는 불가능하고, 양방향 반복자 (++, --)를 이용하여 탐색한다.
    // push_front(), push_back(), pop_front(), pop_back() 멤버 함수를 이용해서 list 양 끝에서 삽입, 삭제가 가능하다.
    // list의 사용 : list<Data Type> [변수 이름];
    // list<KeyFrame*>::iterator -> list의 원소들을 전부 출력하려면 iterator(반복자)를 이용해야 한다. 
    // mlNewKeyFrames list를 삭제하는 과정
    // Q. 만일 이전에 localization mode만 실행이 되었다면, 새로운 keyframe이 만들어졌을 것이기 때문에 해당 list를 지우는 과정
    // mlNewKeyFrames는 전체 keyframe에 대한 list가 아니라, tracking module에서 새롭게 결정한 keyframe일 것이다.
    
    for(list<KeyFrame*>::iterator lit = mlNewKeyFrames.begin(), lend=mlNewKeyFrames.end(); lit!=lend; lit++)
        delete *lit; // lit -> pointer, delete *lit -> de-reference 하여 list의 값을 삭제
    mlNewKeyFrames.clear(); // mlNewKeyFrames는 pointer -> memory 삭제
    // delete : 해당 메모리의 할당이 취소된다.
    // clear() : 리스트의 모든 요소를 제거한다.

    // Local mapping thread가 다시 시작할 때, 이전의 새로운 keyframe list를 삭제
    cout << "Local Mapping RELEASE" << endl;
}

bool LocalMapping::AcceptKeyFrames()
{
    unique_lock<mutex> lock(mMutexAccept); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexAccept를 소유한다.
    return mbAcceptKeyFrames;
}

void LocalMapping::SetAcceptKeyFrames(bool flag)
{
    unique_lock<mutex> lock(mMutexAccept); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexAccept를 소유한다.
    mbAcceptKeyFrames=flag;
}

// flag = true
bool LocalMapping::SetNotStop(bool flag)
{
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexStop을 소유한다.

    if(flag && mbStopped) // flag = true and mbStopped = true
        return false; // stop

    mbNotStop = flag;

    return true;
}

void LocalMapping::InterruptBA()
{
    mbAbortBA = true;
}

void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames();

    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit;
        if(pKF->mnId==0)
            continue;
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
                if(!pMP->isBad())
                {
                    if(!mbMonocular)
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0)
                            continue;
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs)
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave;
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
                        int nObs=0;
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first;
                            if(pKFi==pKF)
                                continue;
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave;

                            if(scaleLeveli<=scaleLevel+1)
                            {
                                nObs++;
                                if(nObs>=thObs)
                                    break;
                            }
                        }
                        if(nObs>=thObs)
                        {
                            nRedundantObservations++;
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs)
            pKF->SetBadFlag();
    }
}

// [a1, a2, a3]' -> [0, -a3, a2; a3, 0, -a1; -a2, a1, 0]
cv::Mat LocalMapping::SkewSymmetricMatrix(const cv::Mat &v)
{
    // cv::Mat_ : Mat_ 객체를 선언할 때에 << 연산자를 이용하여 개별 원소를 한 번에 초기화할 수 있기 때문에 원소의 개수가 작은 행렬의 값을 쉽게 지정할 수 있다.
    return (cv::Mat_<float>(3,3) <<             0, -v.at<float>(2), v.at<float>(1),
            v.at<float>(2),               0,-v.at<float>(0),
            -v.at<float>(1),  v.at<float>(0),              0);
}

// reset이 완료되기 전까지 대기하는 명령을 보내는 함수
void LocalMapping::RequestReset()
{
    // critical section
    {
        unique_lock<mutex> lock(mMutexReset); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexReset을 소유한다.
        mbResetRequested = true; // mbResetRequested flag를 true로 변경
    }
    
    // mbResetRequested flag가 false
    // mbResetRequested = false가 되기 전까지, 계속 대기하라.
    while(1)
    {
        // critical section
        {
            unique_lock<mutex> lock2(mMutexReset); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexReset을 소유한다.
            if(!mbResetRequested) // mbResetRequested = false -> while문을 빠져 나가라.
                break;
        }
        usleep(3000); // 3000마이크로 초 동안 대기하라. -> Local mapping thread
    }
}

void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlNewKeyFrames.clear();
        mlpRecentAddedMapPoints.clear();
        mbResetRequested=false;
    }
}

void LocalMapping::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LocalMapping::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;    
    unique_lock<mutex> lock2(mMutexStop);
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
