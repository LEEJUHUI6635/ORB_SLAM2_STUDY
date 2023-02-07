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
        // CheckNewKeyFrames() = true -> Tracking thread가 생성한 새로운 keyframe이 존재하면,
        {
            // BoW conversion and insertion in Map
            ProcessNewKeyFrame(); // 현재 keyframe(tracking thread -> local mapping thread)의 모든 map point들에 대한 update, covisibility graph 상의 edge update, 현재 keyframe -> 전체 map

            // Check recent MapPoints
            MapPointCulling();
            // 1) 해당 map point는 그것이 발견될 수 있는 모든 keyframe 중 25% 이상을 tracking 해야 한다.
            // 2) 해당 map point가 생성된 후, 세 개 이상의 keyframe에서 발견되어야 한다. <-> map point가 세 개 이상의 keyframe에서 발견되지 못한다면, 언제든지 삭제될 수 있다.

            // Triangulate new MapPoints
            CreateNewMapPoints(); // 현재 keyframe과 covisibility graph 상의 10개 이하의 neighbor keyframes와의 triangulation을 통해 새로운 map point를 생성

            if(!CheckNewKeyFrames()) // Tracking thread에서 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
            // CheckNewKeyFrames() = false -> Tracking thread에서 생성한 새로운 keyframe이 존재하지 않는다면,
            {
                // Find more matches in neighbor keyframes and fuse point duplications
                SearchInNeighbors(); // Tracking thread에서 생성한 새로운 keyframe이 존재하지 않는다면, 보다 많은 map point를 생성하기 위해, 기존 keyframe의 neighbor keyframes와의 correspondence를 더 찾는다.
                // 현재 keyframe의 map point들을 neighbor keyframe에 projection하여, 더 많은 correspondence를 찾고, 현재 keyframe의 map point와 중복되는 neighbor keyframe의 map point를 fusion 한다.
            }

            mbAbortBA = false;
            // CheckNewKeyFrames() -> Tracking thread에서 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
            if(!CheckNewKeyFrames() && !stopRequested()) // CheckNewKeyFrames() = false and stopRequested() = false -> Tracking thread에서 생성한 새로운 keyframe이 존재하지 않고, Local mapping thread가 중지 요청을 받지 않았을 경우
            {
                // Local BA
                if(mpMap->KeyFramesInMap()>2) // 전체 map에 포함되는 keyframe의 개수 > 2
                    Optimizer::LocalBundleAdjustment(mpCurrentKeyFrame,&mbAbortBA, mpMap); // 보류

                // Check redundant local Keyframes
                // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe에서 다른 keyframe에서 관측되는 횟수가 3번 이상인 map point의 개수 >= covisibility graph 상에서 연결되어 있는 keyframe의 모든 map point의 개수 x 0.9
                // 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
                KeyFrameCulling();
            }

            mpLoopCloser->InsertKeyFrame(mpCurrentKeyFrame);
        }
        else if(Stop()) // CheckNewKeyFrames() = false -> Tracking thread가 생성한 새로운 keyframe이 존재하지 않으면,
        // Stop() = true <- mbStopRequested = true and mbNotStop = false
        {
            // Safe area to stop
            while(isStopped() && !CheckFinish()) // isStopped() = true and CheckFinish() = false
            // isStoppped() -> Local mapping이 중지 되었는가에 대한 flag인 mbStopped = true인 경우
            // CheckFinish() -> mbFinishRequested = false인 경우
            {
                usleep(3000); // 3000us 동안 호출 프로세스의 실행을 일시 중지한다.
            }
            if(CheckFinish()) // CheckFinish() = true
                break; // 해당 루프를 종료한다.
        }

        ResetIfRequested(); // mbResetRequested = true -> local mapping thread의 처리 대상인 새로운 keyframe list + mlpRecentAddedMapPoints 삭제 

        // Tracking will see that Local Mapping is busy
        SetAcceptKeyFrames(true); // mbAcceptKeyFrames = true

        if(CheckFinish()) // CheckFinish() = true
            break; // 해당 루프를 종료한다.

        usleep(3000); // 3000us 동안 호출 프로세스의 실행을 일시 중지한다.
    }

    SetFinish(); // mbFinished = true, mbStopped = true
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

// 현재 keyframe과 covisibility graph 상의 10개 이하의 neighbor keyframes와의 triangulation을 통해 새로운 map point를 생성
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
        // Q. 적어도 하나 이상의 neighbor keyframe과 현재 keyframe 간의 map point를 생성할 수 있는 조건이 되었고, 새로운 keyframe이 들어오면, -> 이것을 처리하기 위해
        // 하나 이상의 neighbor keyframe을 처리한 후, tracking thread에서 생성한 새로운 keyframe이 존재하면,
        if(i>0 && CheckNewKeyFrames()) // CheckNewKeyFrames() -> Tracking thread에서 생성한 새로운 keyframe이 존재하면 true를 return하고, 존재하지 않으면 false를 return한다.
            return; // 해당 함수를 빠져나가라.

        KeyFrame* pKF2 = vpNeighKFs[i]; // 현재 keyframe의 covisibility graph 상의 i번째 neighbor keyframe

        // Check first that baseline is not too short
        // parallax가 너무 작으면, 정확도가 높은 3D point를 생성할 수 없기 때문
        cv::Mat Ow2 = pKF2->GetCameraCenter(); // world 좌표계 상에서의 neighbor keyframe의 위치, camera to world coordinate의 translation
        cv::Mat vBaseline = Ow2-Ow1; // neighbor keyframe - 현재 keyframe baseline -> parallax
        const float baseline = cv::norm(vBaseline); // baseline의 크기 -> L2-norm

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
        matcher.SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false); // vMatchedIndices[i] = 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index

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
        const int nmatches = vMatchedIndices.size();
        // vMatchedIndices[ikp] = 현재 keyframe의 i번째 keypoint index, 현재 keyframe의 ikp번째 keypoint와 대응되는 neighbor keyframe의 keypoint index
        for(int ikp=0; ikp<nmatches; ikp++)
        {
            const int &idx1 = vMatchedIndices[ikp].first; // 현재 keyframe의 i번째 keypoint index
            const int &idx2 = vMatchedIndices[ikp].second; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index

            // 현재 keyframe
            const cv::KeyPoint &kp1 = mpCurrentKeyFrame->mvKeysUn[idx1]; // 현재 keyframe의 i번째 keypoint -> left image
            const float kp1_ur=mpCurrentKeyFrame->mvuRight[idx1]; // 현재 keyframe의 i번째 keypoint -> right image
            bool bStereo1 = kp1_ur>=0; // kp1_ur >= 0 -> bStereo1 = true

            // neighbor keyframe
            const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2]; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint -> left image
            const float kp2_ur = pKF2->mvuRight[idx2]; // 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint -> right image
            bool bStereo2 = kp2_ur>=0; // kp2_ur >= 0 -> bStereo2 = true

            // Check parallax between rays
            // 현재 keyframe
            cv::Mat xn1 = (cv::Mat_<float>(3,1) << (kp1.pt.x-cx1)*invfx1, (kp1.pt.y-cy1)*invfy1, 1.0); // pixel 좌표계 -> normalized plane의 metric 좌표계
            cv::Mat xn2 = (cv::Mat_<float>(3,1) << (kp2.pt.x-cx2)*invfx2, (kp2.pt.y-cy2)*invfy2, 1.0); // pixel 좌표계 -> normalized plane의 metric 좌표계
            
            // translation x 
            cv::Mat ray1 = Rwc1*xn1; // keyframe 1(현재 keyframe)의 camera to world coordinate의 rotation x normalized plane의 metric 좌표계 상의, 현재 keyframe의 keypoint
            cv::Mat ray2 = Rwc2*xn2; // keyframe 2(neighbor keyframe)의 camera to world coordinate의 rotation x normalized plane의 metric 좌표계 상의, neighbor keyframe의 keypoint
            const float cosParallaxRays = ray1.dot(ray2)/(cv::norm(ray1)*cv::norm(ray2)); // cos(theta) = cos(ray1과 ray2 사이의 각도)

            float cosParallaxStereo = cosParallaxRays+1;
            float cosParallaxStereo1 = cosParallaxStereo;
            float cosParallaxStereo2 = cosParallaxStereo;

            if(bStereo1) // bStereo1 = true -> 현재 keyframe의 right image 상의 keypoint가 존재하는가.
                cosParallaxStereo1 = cos(2*atan2(mpCurrentKeyFrame->mb/2,mpCurrentKeyFrame->mvDepth[idx1])); // 현재 keyframe의 cos(parallax / 2)
                // atan2(y, x) = tan^(-1)(y/x)
            else if(bStereo2) // bStereo2 = true -> neighbor keyframe의 right image 상의 keypoint가 존재하는가.
                cosParallaxStereo2 = cos(2*atan2(pKF2->mb/2,pKF2->mvDepth[idx2])); // neighbor keyframe의 cos(parallax / 2)

            // parallax가 큰 경우를 선택해야, depth를 보다 정확히 추정할 수 있기 때문
            cosParallaxStereo = min(cosParallaxStereo1,cosParallaxStereo2); // 최소값의 cos(parallax / 2) -> 최대값의 parallax

            cv::Mat x3D;
            // (ray1과 ray2 사이의 각도 > 최대값의 parallax) and (ray1과 ray2 사이의 각도 -> 0~90) and (bStereo1 > 0 or bStereo2 > 0 or ray1과 ray2 사이의 각도 > 0에 근사한 값)
            // 현재 keyframe(left image)와 neighbor keyframe(left image)와의 parallax가 충분히 크다면, 현재 keyframe의 keypoint와 neighbor keyframe의 keypoint와의 triangulation을 통해 map point를 생성한다.
            if(cosParallaxRays<cosParallaxStereo && cosParallaxRays>0 && (bStereo1 || bStereo2 || cosParallaxRays<0.9998))
            {
                // Linear Triangulation Method
                // AX = 0 (X = 3D point, A = [u x p3 - p1; v x p3 - p2; u' x p3' - p1'; v' x p3' - p2']
                cv::Mat A(4,4,CV_32F);
                // world to camera coordinate의 transformation x normalized plane 상의 metric 좌표 = projection matrix x pixel 좌표
                A.row(0) = xn1.at<float>(0)*Tcw1.row(2)-Tcw1.row(0);
                A.row(1) = xn1.at<float>(1)*Tcw1.row(2)-Tcw1.row(1);
                A.row(2) = xn2.at<float>(0)*Tcw2.row(2)-Tcw2.row(0);
                A.row(3) = xn2.at<float>(1)*Tcw2.row(2)-Tcw2.row(1);

                cv::Mat w,u,vt;
                cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV); // svd(A) = w x u x vt
                // cv::SVD::compute(InputArray src, OutputArray w, OutputArray u, OutputArray vt, int flags=0)
                // MODIFY_A : allow the algorithm to modify the decomposed matrix; it can save space and speed up processing.
                // FULL_UV : when the matrix is not squre, by default the algorithm produces u and vt matrices of sufficiently large size for the further A reconstruction;
                // if, however, FULL_UV flag is specified, u and vt will be full-size square orthogonal matrices.

                x3D = vt.row(3).t(); // SVD를 통해 least-square 방식으로 구한 map point

                if(x3D.at<float>(3)==0) // z값이 0이라면,
                    continue; // 해당 루프의 끝으로 이동한다.

                // Euclidean coordinates
                x3D = x3D.rowRange(0,3)/x3D.at<float>(3); // Q. normalized plane 상의 map point

            }
            // bStereo1 = true and 현재 keyframe의 parallax / 2 > neighbor keyframe의 parallax / 2
            else if(bStereo1 && cosParallaxStereo1<cosParallaxStereo2)
            {
                x3D = mpCurrentKeyFrame->UnprojectStereo(idx1); // 현재 keyframe 상의 keypoint와 해당하는 depth를 이용해 map point를 생성한다.    
            }
            // bStereo2 = true and neighbor keyframe의 parallax / 2 > 현재 keyframe의 parallax / 2
            else if(bStereo2 && cosParallaxStereo2<cosParallaxStereo1)
            {
                x3D = pKF2->UnprojectStereo(idx2); // neighbor keyframe 상의 keypoint와 해당하는 depth를 이용해 map point를 생성한다.
            }
            else
                continue; //No stereo and very low parallax

            cv::Mat x3Dt = x3D.t(); // [x, y, z]' -> [x, y, z]

            //Check triangulation in front of cameras
            float z1 = Rcw1.row(2).dot(x3Dt)+tcw1.at<float>(2); // world to camera coordinate x world 좌표계 상의 map point
            if(z1<=0) // 생성된 map point를 현재 keyframe의 camera 좌표계 상으로 위치시켰을 때의 z값
                continue; // 해당 루프의 끝으로 이동한다.

            float z2 = Rcw2.row(2).dot(x3Dt)+tcw2.at<float>(2); // world to camera coordinate x world 좌표계 상의 map point
            if(z2<=0) // 생성된 map point를 neighbor keyframe의 camera 좌표계 상으로 위치시켰을 때의 z값
                continue; // 해당 루프의 끝으로 이동한다.

            //Check reprojection error in first keyframe
            const float &sigmaSquare1 = mpCurrentKeyFrame->mvLevelSigma2[kp1.octave]; // Q.
            const float x1 = Rcw1.row(0).dot(x3Dt)+tcw1.at<float>(0); // world to camera coordinate x world 좌표계 상의 map point
            const float y1 = Rcw1.row(1).dot(x3Dt)+tcw1.at<float>(1); // world to camera coordinate x world 좌표계 상의 map point
            const float invz1 = 1.0/z1;

            if(!bStereo1) // bStereo1 = false -> Q. monocular
            {
                float u1 = fx1*x1*invz1+cx1;
                float v1 = fy1*y1*invz1+cy1;
                float errX1 = u1 - kp1.pt.x;
                float errY1 = v1 - kp1.pt.y;
                if((errX1*errX1+errY1*errY1)>5.991*sigmaSquare1)
                    continue;
            }
            else // bStereo1 = true -> stereo
            {
                // 생성된 map point를 현재 keyframe의 pixel 좌표계로 projection
                float u1 = fx1*x1*invz1+cx1; // left image의 pixel coordinate
                float u1_r = u1 - mpCurrentKeyFrame->mbf*invz1; // right image의 pixel coordinate
                float v1 = fy1*y1*invz1+cy1; // left image의 pixel coordinate = right image의 pixel coordinate
                float errX1 = u1 - kp1.pt.x; // 생성된 map point를 projection한 pixel 좌표 - map point를 생성하는데 이용되었던 keypoint의 pixel 좌표
                float errY1 = v1 - kp1.pt.y; // 생성된 map point를 projection한 pixel 좌표 - map point를 생성하는데 이용되었던 keypoint의 pixel 좌표
                float errX1_r = u1_r - kp1_ur; 
                if((errX1*errX1+errY1*errY1+errX1_r*errX1_r)>7.8*sigmaSquare1) // 총 reprojection error가 특정 threshold보다 크다면,
                    continue; // 해당 루프의 끝으로 이동한다.
            }

            //Check reprojection error in second keyframe
            const float sigmaSquare2 = pKF2->mvLevelSigma2[kp2.octave]; // Q.
            const float x2 = Rcw2.row(0).dot(x3Dt)+tcw2.at<float>(0); // world to camera coordinate x world 좌표계 상의 map point
            const float y2 = Rcw2.row(1).dot(x3Dt)+tcw2.at<float>(1); // world to camera coordinate x world 좌표계 상의 map point
            const float invz2 = 1.0/z2;
            if(!bStereo2) // bStereo2 = false -> Q. monocular
            {
                float u2 = fx2*x2*invz2+cx2;
                float v2 = fy2*y2*invz2+cy2;
                float errX2 = u2 - kp2.pt.x;
                float errY2 = v2 - kp2.pt.y;
                if((errX2*errX2+errY2*errY2)>5.991*sigmaSquare2)
                    continue;
            }
            else // bStereo2 = true -> stereo
            {
                // 생성된 map point를 neighbor keyframe의 pixel 좌표계로 projection
                float u2 = fx2*x2*invz2+cx2; // left image의 pixel coordinate
                float u2_r = u2 - mpCurrentKeyFrame->mbf*invz2; // right image의 pixel coordinate
                float v2 = fy2*y2*invz2+cy2; // left image의 pixel coordinate = right image의 pixel coordinate
                float errX2 = u2 - kp2.pt.x; // 생성된 map point를 projection한 pixel 좌표 - map point를 생성하는데 이용되었던 keypoint의 pixel 좌표
                float errY2 = v2 - kp2.pt.y; // 생성된 map point를 projection한 pixel 좌표 - map point를 생성하는데 이용되었던 keypoint의 pixel 좌표
                float errX2_r = u2_r - kp2_ur;
                if((errX2*errX2+errY2*errY2+errX2_r*errX2_r)>7.8*sigmaSquare2) // 총 reprojection error가 특정 threshold보다 크다면,
                    continue; // 해당 루프의 끝으로 이동한다.
            }

            //Check scale consistency
            cv::Mat normal1 = x3D-Ow1; // map point - world 좌표계 상에서의 현재 keyframe의 위치(camera to world coordinate의 translation)
            float dist1 = cv::norm(normal1);

            cv::Mat normal2 = x3D-Ow2; // map point - world 좌표계 상에서의 neighbor keyframe의 위치(camera to world coordinate의 translation)
            float dist2 = cv::norm(normal2);

            if(dist1==0 || dist2==0)
                continue; // 해당 루프의 끝으로 이동한다.

            const float ratioDist = dist2/dist1; // map point - neighbor keyframe / map point - 현재 keyframe
            const float ratioOctave = mpCurrentKeyFrame->mvScaleFactors[kp1.octave]/pKF2->mvScaleFactors[kp2.octave];

            /*if(fabs(ratioDist-ratioOctave)>ratioFactor)
                continue;*/
            if(ratioDist*ratioFactor<ratioOctave || ratioDist>ratioOctave*ratioFactor) // Q. 
            // ratioFactor = 1.5f*mpCurrentKeyFrame->mfScaleFactor
                continue; // 해당 루프의 끝으로 이동한다.

            // Triangulation is succesfull
            MapPoint* pMP = new MapPoint(x3D,mpCurrentKeyFrame,mpMap);
            
            // Data Association -> 새로운 map point가 어느 keyframe에서 발견된 몇 번째 keypoint인지 저장한다.
            pMP->AddObservation(mpCurrentKeyFrame,idx1); // 생성된 map point는 현재 keyframe의 idx1번째 keypoint와 association 관계를 가진다.
            pMP->AddObservation(pKF2,idx2); // 생성된 map point는 neighbor keyframe의 idx2번째 keypoint와 association 관계를 가진다.

            // 생성된 map point를 각 keyframe의 mvpMapPoints vector에 귀속시킨다.
            mpCurrentKeyFrame->AddMapPoint(pMP,idx1);
            pKF2->AddMapPoint(pMP,idx2);

            pMP->ComputeDistinctiveDescriptors(); // 생성된 map point의 representative descriptor(다른 descriptor와의 hamming distance가 가장 작은 descriptor)를 저장한다.

            pMP->UpdateNormalAndDepth(); // 생성된 map point는 max distance와 min distance, mean normal vector를 가지고 있다.

            mpMap->AddMapPoint(pMP); // pMP -> Map::mspMapPoints
            mlpRecentAddedMapPoints.push_back(pMP);

            nnew++; // triangulation 혹은 stereo camera의 known depth를 이용하여 생성한 map point 개수
        }
    }
}

void LocalMapping::SearchInNeighbors()
{
    // Retrieve neighbor keyframes
    int nn = 10;
    if(mbMonocular) // mbMonocular = true
        nn=20;
    const vector<KeyFrame*> vpNeighKFs = mpCurrentKeyFrame->GetBestCovisibilityKeyFrames(nn); // 현재 keyframe의 covisibility graph 상에서의 10개 이하의 neighbor keyframes
    vector<KeyFrame*> vpTargetKFs;
    for(vector<KeyFrame*>::const_iterator vit=vpNeighKFs.begin(), vend=vpNeighKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit; // de-reference -> neighbor keyframe
        if(pKFi->isBad() || pKFi->mnFuseTargetForKF == mpCurrentKeyFrame->mnId) // 해당 neighbor keyframe이 나쁘다고 판단되거나, 해당 neighbor keyframe이 fuse될 target keyframe이 현재 keyframe이라면(중복 방지),
            continue; // 해당 루프의 끝으로 이동한다.
        vpTargetKFs.push_back(pKFi);
        pKFi->mnFuseTargetForKF = mpCurrentKeyFrame->mnId; // 중복 방지

        // Extend to some second neighbors
        const vector<KeyFrame*> vpSecondNeighKFs = pKFi->GetBestCovisibilityKeyFrames(5); // 해당 neighbor keyframe의 covisibility graph 상에서의 5개 이하의 neighbor keyframes
        for(vector<KeyFrame*>::const_iterator vit2=vpSecondNeighKFs.begin(), vend2=vpSecondNeighKFs.end(); vit2!=vend2; vit2++)
        {
            KeyFrame* pKFi2 = *vit2; // de-reference -> neighbor keyframe의 neighbor keyframe
            if(pKFi2->isBad() || pKFi2->mnFuseTargetForKF==mpCurrentKeyFrame->mnId || pKFi2->mnId==mpCurrentKeyFrame->mnId)
            // 해당 neighbor keyframe이 나쁘다고 판단되거나, 해당 neighbor keyframe이 fuse될 target keyframe이 현재 keyframe이라면(중복 방지),
            // + 현재 keyframe의 neighbor keyframe의 neighbor keyframe이 다시 현재 keyframe이 될 수 있기 때문에 이 또한 방지한다.
                continue; // 해당 루프의 끝으로 이동한다.
            vpTargetKFs.push_back(pKFi2);
        }
    }


    // Search matches by projection from current KF in target KFs
    // 현재 keyframe의 map point들을 neighbor keyframe에 projection하여, 더 많은 correspondence를 찾는다. 현재 keyframe의 map point와 일치하는 neighbor keyframe의 map point가 존재한다면 fusion(map point replacement)하고, 그렇지 않다면 neighbor keyframe의 observation을 갱신한다.
    ORBmatcher matcher;
    vector<MapPoint*> vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 keyframe 상의 keypoint와 association의 관계를 가지는 map point
    for(vector<KeyFrame*>::iterator vit=vpTargetKFs.begin(), vend=vpTargetKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit; // de-reference -> neighbor keyframe

        matcher.Fuse(pKFi,vpMapPointMatches);
        
    }

    // Search matches by projection from target KFs in current KF
    vector<MapPoint*> vpFuseCandidates;
    vpFuseCandidates.reserve(vpTargetKFs.size()*vpMapPointMatches.size()); // 현재 keyframe의 neighbor keyframe의 개수 x 현재 keyframe 상의 map point의 개수

    for(vector<KeyFrame*>::iterator vitKF=vpTargetKFs.begin(), vendKF=vpTargetKFs.end(); vitKF!=vendKF; vitKF++) // 현재 keyframe의 neighbor keyframe의 개수만큼 반복
    {
        KeyFrame* pKFi = *vitKF; // de-reference -> 현재 keyframe의 neighbor keyframe

        vector<MapPoint*> vpMapPointsKFi = pKFi->GetMapPointMatches(); // neighbor keyframe 상의 keypoint와 association의 관계를 가지는 map point

        for(vector<MapPoint*>::iterator vitMP=vpMapPointsKFi.begin(), vendMP=vpMapPointsKFi.end(); vitMP!=vendMP; vitMP++) // neighbor keyframe의 map point의 개수만큼 반복
        {
            MapPoint* pMP = *vitMP; // de-reference -> neighbor keyframe의 map point
            if(!pMP) // pMP = NULL
                continue; // 해당 루프의 끝으로 이동한다.
            if(pMP->isBad() || pMP->mnFuseCandidateForKF == mpCurrentKeyFrame->mnId) // neighbor keyframe의 map point가 나쁘다고 판단하거나, 해당 map point가 이미 fusion의 대상으로 들어간 경우,
                continue; // 해당 루프의 끝으로 이동한다.
            pMP->mnFuseCandidateForKF = mpCurrentKeyFrame->mnId; // 중복 방지
            vpFuseCandidates.push_back(pMP);
        }
    }

    // neighbor keyframe의 map point들을 현재 keyframe에 projection하여, 더 많은 correspondence를 찾는다. neighbor keyframe의 map point와 일치하는 현재 keyframe의 map point가 존재한다면 fusion(map point replacement)하고, 그렇지 않다면 현재 keyframe의 observation을 갱신한다.
    matcher.Fuse(mpCurrentKeyFrame,vpFuseCandidates);


    // Update points
    vpMapPointMatches = mpCurrentKeyFrame->GetMapPointMatches(); // 현재 keyframe 상의 keypoint와 association의 관계를 가지는 map point
    for(size_t i=0, iend=vpMapPointMatches.size(); i<iend; i++)
    {
        MapPoint* pMP=vpMapPointMatches[i]; // 현재 keyframe의 i번째 keypoint와 association의 관계를 가지는 map point
        if(pMP) // 해당 map point가 존재한다면,
        {
            if(!pMP->isBad()) // 해당 map point가 나쁘다고 판단하지 않는다면,
            {
                pMP->ComputeDistinctiveDescriptors(); // 해당 map point의 representative descriptor(다른 descriptor와의 hamming distance가 가장 작은 descriptor)를 저장한다.
                pMP->UpdateNormalAndDepth(); // 해당 map point는 max distance와 min distance, mean normal vector를 계산한다.
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
    mbAbortBA = true; // 새로운 keyframe이 들어오면, Local Mapping thread에게 Local BA를 중단하라는 메시지를 보낸다.
}

bool LocalMapping::Stop()
{
    unique_lock<mutex> lock(mMutexStop); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexStop을 소유한다.
    if(mbStopRequested && !mbNotStop) // mbStopRequested = true and mbNotStop = false
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

// 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe에서 다른 keyframe에서 관측되는 횟수가 3번 이상인 map point의 개수 >= covisibility graph 상에서 연결되어 있는 keyframe의 모든 map point의 개수 x 0.9
void LocalMapping::KeyFrameCulling()
{
    // Check redundant keyframes (only local keyframes)
    // A keyframe is considered redundant if the 90% of the MapPoints it sees, are seen
    // in at least other 3 keyframes (in the same or finer scale)
    // We only consider close stereo points
    vector<KeyFrame*> vpLocalKeyFrames = mpCurrentKeyFrame->GetVectorCovisibleKeyFrames(); // covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 정렬

    // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들에 대하여 반복
    for(vector<KeyFrame*>::iterator vit=vpLocalKeyFrames.begin(), vend=vpLocalKeyFrames.end(); vit!=vend; vit++)
    {
        KeyFrame* pKF = *vit; // de-reference -> covisibility graph 상에서 연결되어 있는 keyframe
        if(pKF->mnId==0) // 해당 keyframe이 맨 처음의 keyframe이라면,
            continue; // 해당 루프의 끝으로 이동한다.
        const vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches(); // covisibility graph 상에서 연결되어 있는 keyframe 상의 keypoint와 association의 관계를 가지는 map point

        int nObs = 3;
        const int thObs=nObs;
        int nRedundantObservations=0;
        int nMPs=0;
        // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들의 모든 map point에 대하여 반복
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i]; // covisibility graph 상에서 연결되어 있는 keyframe의 i번째 keypoint에 해당하는 map point
            if(pMP) // 해당 map point가 존재하면,
            {
                if(!pMP->isBad()) // 해당 map point가 나쁘다고 판단하지 않으면,
                {
                    if(!mbMonocular) // mbMonocular = false
                    {
                        if(pKF->mvDepth[i]>pKF->mThDepth || pKF->mvDepth[i]<0) // 해당 keyframe의 i번째 map point의 depth가 threshold보다 크거나(far points의 경우) 0보다 작은 경우
                            continue; // 해당 루프의 끝으로 이동한다.
                    }

                    nMPs++;
                    if(pMP->Observations()>thObs) // 해당 map point와 일치하는 keyframe의 keypoint 개수 > 3 = 해당 map point가 3번 이상 관측된다면,
                    {
                        const int &scaleLevel = pKF->mvKeysUn[i].octave; // covisibility graph 상에서 연결되어 있는 keyframe의 i번째 keypoint의 scale level
                        const map<KeyFrame*, size_t> observations = pMP->GetObservations(); // key : keyframe, value : keypoint index
                        int nObs=0;
                        // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들의 모든 map point가 관측되는 모든 keyframe(covisibility graph 상에서 연결되어 있는 자기 자신의 keyframe에 대하여는 제외), keypoint에 대하여 반복
                        for(map<KeyFrame*, size_t>::const_iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
                        {
                            KeyFrame* pKFi = mit->first; // mit->first : 해당 map point가 관측되는 keyframe
                            if(pKFi==pKF) // 자기 자신의 keyframe(covisibility graph 상에서 연결되어 있는 keyframe)의 경우에 대하여 제외한다.
                                continue; // 해당 루프의 끝으로 이동한다.
                            const int &scaleLeveli = pKFi->mvKeysUn[mit->second].octave; // 해당 map point가 관측되는 keyframe의 mit->second번째 keypoint의 scale level

                            if(scaleLeveli<=scaleLevel+1) // Q.
                            {
                                nObs++; // 해당 map point가 관측되는 keyframe의 개수
                                if(nObs>=thObs) // 해당 map point가 관측되는 keyframe의 개수 >= 3
                                    break; // 해당 루프를 종료시킨다.
                            }
                        }
                        if(nObs>=thObs) // 해당 map point가 관측되는 keyframe의 개수 >= 3
                        {
                            nRedundantObservations++; 
                            // covisibility graph 상에서 연결되어 있는 keyframe에서 다른 keyframe에서 관측되는 횟수가 3번 이상인 map point의 개수
                        }
                    }
                }
            }
        }  

        if(nRedundantObservations>0.9*nMPs) // covisibility graph 상에서 연결되어 있는 keyframe에서 다른 keyframe에서 관측되는 횟수가 3번 이상인 map point의 개수 >= covisibility graph 상에서 연결되어 있는 keyframe의 모든 map point의 개수 x 0.9
            pKF->SetBadFlag(); // 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
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

// Local mapping thread를 취소
void LocalMapping::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexReset을 소유한다.
    if(mbResetRequested) // mbResetRequested = true
    {
        mlNewKeyFrames.clear(); // local mapping thread의 처리 대상인 새로운 keyframe list를 삭제
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
    unique_lock<mutex> lock(mMutexFinish); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFinish를 소유한다.
    return mbFinishRequested;
}

void LocalMapping::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFinish를 소유한다.
    mbFinished = true;
    unique_lock<mutex> lock2(mMutexStop); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexStop을 소유한다.
    mbStopped = true;
}

bool LocalMapping::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}

} //namespace ORB_SLAM
