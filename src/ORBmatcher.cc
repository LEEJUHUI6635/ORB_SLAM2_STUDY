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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

// matcher.SearchByProjection(mCurrentFrame,mvpLocalMapPoints,th)
// input : F, vpMapPoints, th, output : nmatches
// local map points를 현재 frame에 투영시켜, 현재 frame의 keypoint에 대응되는 더 많은 map point를 찾는다.
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0; // th != 1.0 -> bFactor = true, th = 1.0 -> bFactor = false

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++) // local map point 개수만큼 반복
    {
        MapPoint* pMP = vpMapPoints[iMP]; // iMP번째 local map point -> pMP
        if(!pMP->mbTrackInView) // pMP->mbTrackInView = false -> 현재 frame의 frustum에 존재 x
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
            continue; // 해당 루프의 끝으로 이동한다.

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // image optical center와 map point를 잇는 벡터와 map point의 mean viewing vector가 이루는 각도에 따라 search window의 size가 결정된다.
        float r = RadiusByViewingCos(pMP->mTrackViewCos);
        // image optical center와 map point를 잇는 벡터와 map point의 mean viewing vector와의 각도가 크다면, search window의 size를 키운다.

        if(bFactor) // bFactor = true -> th != 1.0
            r*=th;

        // Grid Search -> local map point를 현재 frame에 projection하여, grid search를 통해 찾은 특정 grid 내의 keypoint들의 index
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        const cv::Mat MPdescriptor = pMP->GetDescriptor(); // 해당 map point가 관측되는 keyframe의 keypoint descriptor, 해당 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist=256; // descriptor distance의 최대값
        int bestLevel= -1;
        int bestDist2=256; // descriptor distance의 최대값
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> keypoint의 index

            // 이미 찾은 correspondence이기 때문에, correspondence를 새로 계산할 필요 x
            if(F.mvpMapPoints[idx]) // grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 map point가 존재한다면,
                if(F.mvpMapPoints[idx]->Observations()>0) // 해당 map point를 관측하는 keyframe이 하나 이상 발견된다면,
                    continue; // 해당 루프의 끝으로 이동한다.

            if(F.mvuRight[idx]>0) // monocular가 아닌 경우
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue; // 해당 루프의 끝으로 이동한다.
            }

            const cv::Mat &d = F.mDescriptors.row(idx); // grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 descriptor

            const int dist = DescriptorDistance(MPdescriptor,d);
            // local map point의 representative descriptor와 grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 descriptor 간의 거리

            if(dist<bestDist) // bestDist -> 두 descriptor 간의 가장 작은 distance
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2) // bestDist2 -> 두 descriptor 간의 두 번째로 작은 distance
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH) // TH_HIGH = 100
        {
            // Q. bestLevel == bestLevel2 -> 두 후보가 거의 비슷하다.
            // bestDist > mfNNratio * bestDist2 -> 두 후보가 꽤 큰 차이를 보이지 않으면 search를 중단한다.
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue; // 해당 루프의 끝으로 이동한다.

            F.mvpMapPoints[bestIdx]=pMP; // local map points -> 현재 frame의 mvpMapPoints vector
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998) // image optical center와 map point를 잇는 벡터와 map point의 mean viewing vector와의 각도가 크지 않으면,
        return 2.5;
    else // viewCos <= 0.998 -> image optical center와 map point를 잇는 벡터와 map point의 mean viewing vector와의 각도가 크다면,
        return 4.0; // search window의 size를 키운다.
}

// kp1은 현재 keyframe 상의 idx1번째 keypoint, kp2는 neighbor keyframe 상의 idx2번째 keypoint
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    // F = [f1, f2, f3; f4, f5, f6; f7, f8, f9]
    // projected point x가 다른 image plane 위에 그려지는 epipolar line이 되려면 fundamental matrix F를 곱해주면 된다.
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0); // x*f1 + y*f2 + f3
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1); // x*f4 + y*f5 + f6
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2); // x*f7 + y*f8 + f9
    // F x 현재 keyframe 상의 keypoint = neighbor keyframe 상의 epipolar line

    // neighbor keyframe 상의 epipolar line과 neighbor keyframe 상의 idx2번째 keypoint인 kp2와의 거리
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0) // a = 0 and b = 0
        return false;

    // 점과 선과의 거리^2
    const float dsqr = num*num/den; // dsqr = ((ax+by+c)/root(a^2+b^2))^2
    
    // neighbor keyframe 상의 epipolar line과 neighbor keyframe 상의 idx2번째 keypoint인 kp2와의 거리가 특정 threshold보다 낮다면,
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave]; // dsqr < 3.84*pKF2->mvLevelSigma2[kp2.octave] -> return true, dsqr >= 3.84*pKF2->mvLevelSigma2[kp2.octave]
}

// 모든 frame에 대한 BoW는 존재하지 않지만, 모든 keyframe에 대한 BoW는 존재한다.
// input : keyframe, frame, output : vpMapPointMatches(keyframe의 map points - frame의 map points matching), nmatches
// BoW의 두 번째 기능인 data association을 통해 현재 frame과 reference keyframe 간의 matching되는 map point를 찾는 함수
// 현재 frame의 map point는 알 수 없다. reference keyframe의 keypoint와 현재 frame의 keypoint와의 association 관계(BoW의 두 번째 기능)를 통해, 현재 frame의 keypoint에 해당하는 map point를 얻을 수 있다.
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches) // keyframe의 map points - frame의 map points
{
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches(); // reference keyframe의 map points

    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL)); // frame의 keypoint 개수만큼의 Null 값의 map point를 가지는 vector

    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec; // reference keyframe의 feature vector

    int nmatches=0;

    // rotHist[30][500]
    vector<int> rotHist[HISTO_LENGTH]; // HISTO_LENGTH = 30
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500); // 30x500
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
    // feature vector -> image 1개를 표현할 수 있는 vector
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin(); // keyframe의 feature vector의 시작 pointer
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin(); // frame의 feature vector의 시작 pointer
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end(); // keyframe의 feature vector의 끝 pointer
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end(); // frame의 feature vector의 끝 pointer

    while(KFit != KFend && Fit != Fend) // KFit과 Fit 두 반복자로 반복
    {
        // 같은 node 단에서 data association
        if(KFit->first == Fit->first) // keyframe feature vector의 node id == frame feature vector의 node id
        {
            // KFit(Fit)->second : cvflann::lsh::Bucket
            const vector<unsigned int> vIndicesKF = KFit->second; // 특정 node에 존재하는 keypoint의 idx로 이루어진 vector
            const vector<unsigned int> vIndicesF = Fit->second;
            
            // 현재 frame의 map point와 reference keyframe의 map point 중에서 겹치는 것이 있는지 확인
            // Q. vIndicesKF -> 동일한 node id에 있는 keypoint index의 집합
            // vIndicesKF.size() : 동일한 node에 존재하는 keypoint의 개수
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++) // 동일한 node id에 있는 keyframe 상의 keypoint의 개수만큼 반복
            {
                const unsigned int realIdxKF = vIndicesKF[iKF]; // realIdxKF -> 동일 node에 있는 keypoint index
                // realIdxKF -> 특정 node에 해당하는 keyframe id

                // map point
                MapPoint* pMP = vpMapPointsKF[realIdxKF]; // realIdxKF의 keypoint와 match되는 reference keyframe의 map point

                if(!pMP) // realIdxKf index의 keypoint에 해당하는 map point가 없으면,
                    continue; // continue -> loop 몸체 끝으로 점프한다. = 해당 keypoint를 고려하지 않는다.

                if(pMP->isBad()) // 해당하는 map point를 찾았지만 나쁘다고 판단되면,
                    continue; // continue -> loop 몸체 끝으로 점프한다. = 해당 keypoint를 고려하지 않는다.

                // descriptor
                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF); // keyframe 상의 realIdxKF index를 가지는 descriptor

                int bestDist1=256; // DescriptorDistance의 최대값
                int bestIdxF =-1 ;
                int bestDist2=256; // DescriptorDistance의 최대값

                for(size_t iF=0; iF<vIndicesF.size(); iF++) // 동일한 node id에 있는 frame 상의 keypoint의 개수만큼 반복
                {
                    const unsigned int realIdxF = vIndicesF[iF]; // realIdxF -> 동일 node에 있는 keypoint index

                    // nmatches 중복을 막기 위해
                    if(vpMapPointMatches[realIdxF]) // 동일 node에 있는 keypoint index에 해당하는 frame의 map point가 존재하면,
                        continue; // continue -> loop 몸체 끝으로 점프한다.

                    // descriptor
                    const cv::Mat &dF = F.mDescriptors.row(realIdxF); // frame 상의 realIdxF index를 가지는 descriptor

                    const int dist =  DescriptorDistance(dKF,dF); // 동일한 node id에 있는 frame의 keypoint의 descriptor - keyframe의 map point의 descriptor

                    // bestDist1 -> keyframe의 특정 map point descriptor에 대해 frame의 map point descriptor의 가장 작은 distance 값, bestDist2 -> 두 번째로 작은 distance 값
                    if(dist<bestDist1) // Q. 최종적으로는 bestDist1보다 작은 distance 값을 가지는 keypoint의 index를 추출하기 위함
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                // keyframe의 특정 map point descriptor에 대해 frame의 map point descriptor의 가장 작은 distance 값 < TH_LOW
                if(bestDist1<=TH_LOW) // TH_LOW = 50 -> 해당 조건을 만족해야 꽤 적당한 quality의 data association(keyframe의 keypoint와 겹치는 frame의 keypoint)를 얻었다고 할 수 있다.
                {
                    // bestDist1 < mfNNratio x bestDist2 -> 첫 번째 descriptor 후보의 distance는 두 번째 descriptor 후보의 distance와 꽤 큰 차이를 보여야 한다.
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)) // bestDist1 < mfNNratio x bestDist2
                    {
                        vpMapPointMatches[bestIdxF]=pMP; // realIdxKF의 keypoint와 match되는 reference keyframe의 map point -> 현재 frame의 map point를 구하는 과정
                        // bestIdxF=realIdxF;
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF]; // realIdxKF의 index 값을 가지는 keyframe 상의 keypoint
                        
                        // rotation histogram을 만드는 과정
                        if(mbCheckOrientation) // mbCheckOrientation = true
                        {
                            // rot -> keypoint가 회전한 정도
                            // 하나의 frame에서 다른 frame으로의 모든 keypoint의 rotation 정도는 비슷할 것이다.
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle; // keyframe의 keypoint의 angle - frame의 keypoint의 angle = rot
                            if(rot<0.0) // rot의 값이 음수라면,
                                rot+=360.0f; // rot = rot + 360
                            int bin = round(rot*factor); // rot / HISTO_LENGTH = bin -> bin 1개 = 12도(360/30)
                            if(bin==HISTO_LENGTH) // round(rot) = 30 x 30
                                bin=0; // 반올림한 rot의 값이 900이면 bin = 0, rotHist[30] : bin = 30 -> bin = 0
                            assert(bin>=0 && bin<HISTO_LENGTH); // assert(조건) : 해당 조건식이 false라면 runtime error가 나도록 한다.
                            rotHist[bin].push_back(bestIdxF); // bestIdxF -> rotHist[bin]
                            // rotHist[bin] = bestIdxF1, bestIdxF2, ...
                        }
                        // associated keypoint의 개수
                        nmatches++; // keyframe 상의 map point의 descriptor와 frame 상의 descriptor 사이의 distance가 일정 값보다 작다면,
                    }
                }
            }

            KFit++;
            Fit++;
        }
        // Q. node id의 크기에 따라 다르게 처리 -> vocabulary tree 형식이기 때문
        else if(KFit->first < Fit->first) // keyframe feature vector의 node id < frame feature vector의 node id
        {
            KFit = vFeatVecKF.lower_bound(Fit->first); // Finds the beginning of a subsequence matching given key
            // 주어진 key(frame의 node id)와 match 되는 subsequence의 시작을 찾는다.
        }
        else // keyframe feature vector의 node id > frame feature vector의 node id
        {
            Fit = F.mFeatVec.lower_bound(KFit->first); // Finds the beginning of a subsequence matching given key
            // 주어진 key(keyframe의 node id)와 match 되는 subsequence의 시작을 찾는다.
        }
    }

    // 하나의 frame에서 다른 frame으로의 모든 keypoint의 rotation 정도는 비슷할 것이다.
    // rotation histogram의 빈도수가 적은 keypoint는 잘못된 association이 수행되었다고 판단한다.
    if(mbCheckOrientation) // mbCheckOrientation = true, rotHist[30]
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        // input : rotHist(orientation 관련 histogram), HISTO_LENGTH(histogram의 길이 = word의 개수), 
        // output : ind1(가장 많이 나오는 word), ind2(두 번째로 많이 나오는 word), ind3(세 번째로 많이 나오는 word)
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3); // ind = bin

        for(int i=0; i<HISTO_LENGTH; i++) // word의 개수만큼 반복
        {
            if(i==ind1 || i==ind2 || i==ind3) // i가 첫 번째, 두 번째, 세 번째로 많이 나오는 word의 index라면,
                continue; // for문을 빠져나가라.
            // 많이 발견되는 bin에 속하지 않는 keypoint라면,
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL); // Null 값으로 초기화
                nmatches--; // nmatches를 줄인다.
            }
        }
    }

    return nmatches;
}

// matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10)
int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    // 현재 keyframe의 intrinsic parameter
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    // Scw -> scale을 고려한 현재 keyframe의 world to camera coordinate의 transformation
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3); // scale을 고려한 world to camera coordinate의 rotation
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0))); // Q. Rcw.row(0) x Rcw.row(0) = 1, scw -> scale
    cv::Mat Rcw = sRcw/scw; // scale을 고려하지 않은 determinant 값이 1인 world to camera coordinate의 rotation
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw; // scale을 고려하지 않은 world to camera coordinate의 translation
    cv::Mat Ow = -Rcw.t()*tcw; // camera to world coordinate의 translation

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end()); // mvpCurrentMatchedPoints -> spAlreadyFound
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL)); // mvpCurrentMatchedPoints에서 NULL 값 삭제

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    // vpPoints = mvpLoopMapPoints -> loop detection keyframe + neighbor keyframes의 map points
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP)) // 해당 map point가 나쁘다고 판단하거나, 이미 찾은 correspondence인 경우
            continue; // 해당 루프의 끝으로 이동한다.

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw; // 현재 keyframe의 world to camera coordinate의 transformation x world 좌표계 상의 map point의 position = 현재 keyframe 상의 map point의 position

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0) // 현재 keyframe 상의 map point position의 z 값 < 0.0
            continue; // 해당 루프의 끝으로 이동한다.

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v)) // pixel 좌표인 x와 y가 image boundary에 있는 경우 return true, image boundary 밖에 있는 경우 return false
        // pKF->IsInImage(u,v) = false -> map point를 현재 keyframe에 projection 하였을 때, image boundary를 벗어나는 경우
            continue; // 해당 루프의 끝으로 이동한다.

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        cv::Mat PO = p3Dw-Ow; // world 좌표계에서 map point - world 좌표계에서 camera의 위치(camera to world coordinate의 translation)
        const float dist = cv::norm(PO); // camera 좌표계 원점에서 map point까지의 거리

        if(dist<minDistance || dist>maxDistance)
            continue; // 해당 루프의 끝으로 이동한다.

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // 해당 map point의 mean viewing direction

        if(PO.dot(Pn)<0.5*dist) // 해당 map point의 mean viewing direction과 map point - camera 좌표계의 원점 벡터가 이루는 각도가 60을 초과하면,
            continue; // 해당 루프의 끝으로 이동한다.

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // search window의 크기

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius); // map point를 현재 keyframe에 projection하고, Grid Search를 통해 map point와 가장 유사한 현재 keyframe의 keypoint를 search 한다.
        // vIndices -> map point와 가장 유사한 현재 keyframe의 keypoint index

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> map point와 가장 유사한 현재 keyframe의 keypoint index
            if(vpMatched[idx]) // vpMatched[idx] != NULL
                continue; // 해당 루프의 끝으로 이동한다.

            const int &kpLevel= pKF->mvKeysUn[idx].octave; // 현재 keyframe의 idx번째 keypoint의 scale level

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue; // 해당 루프의 끝으로 이동한다.

            const cv::Mat &dKF = pKF->mDescriptors.row(idx); // 현재 keyframe의 idx번째 keypoint의 descriptor

            const int dist = DescriptorDistance(dMP,dKF); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor - 현재 keyframe의 idx번째 keypoint의 descriptor

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW) // TH_LOW = 50
        {
            vpMatched[bestIdx]=pMP; // mvpCurrentMatchedPoints[bestIdx] = pMP
            nmatches++; 
        }

    }

    return nmatches; // loop detection keyframe + neighbor keyframes의 map points와 현재 keyframe의 map points와의 새롭게 찾은 correspondence의 개수
}

int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
        if(level1>0)
            continue;

        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

        if(vIndices2.empty())
            continue;

        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;
        int bestDist2 = INT_MAX;
        int bestIdx2 = -1;

        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;

            cv::Mat d2 = F2.mDescriptors.row(i2);

            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

        if(bestDist<=TH_LOW)
        {
            if(bestDist<(float)bestDist2*mfNNratio)
            {
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

                if(mbCheckOrientation)
                {
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
                    int bin = round(rot*factor);
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}

// matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i])
// mpCurrentKF -> 현재 keyframe
// pKF -> i번째 loop detection candidate keyframe
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    // 현재 keyframe
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn; // 현재 keyframe의 keypoint
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec; // 현재 keyframe의 feature vector
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches(); // 현재 keyframe 상의 keypoint와 association의 관계를 가지는 map point
    const cv::Mat &Descriptors1 = pKF1->mDescriptors; // 현재 keyframe 상의 keypoint의 descriptor

    // i번째 loop detection candidate keyframe
    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn; // i번째 loop detection candidate keyframe의 keypoint
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec; // i번째 loop detection candidate keyframe의 feature vector
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches(); // i번째 loop detection candidate keyframe 상의 keypoint와 association의 관계를 가지는 map point
    const cv::Mat &Descriptors2 = pKF2->mDescriptors; // i번째 loop detection candidate keyframe 상의 keypoint의 descriptor

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL)); // 현재 keyframe의 map point 개수만큼 NULL 값으로 초기화
    vector<bool> vbMatched2(vpMapPoints2.size(),false); // i번째 loop detection candidate keyframe의 map point 개수만큼 false로 초기화

    vector<int> rotHist[HISTO_LENGTH]; // rotHist[30]
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500); // rotHist[30][500]

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    // BoW의 두 번째 기능 -> Data Association
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin(); // 현재 keyframe의 feature vector에 대한 반복자
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin(); // i번째 loop detection candidate keyframe의 feature vector에 대한 반복자
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first) // f1it->first : node id, f2it->first : node id
        // 현재 keyframe의 vocabulary tree의 node id = i번째 loop detection candidate keyframe의 vocabulary tree의 node id
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++) // f1it->second : 현재 keyframe의 특정 node id에 속한 keypoint index의 개수만큼 반복
            {
                const size_t idx1 = f1it->second[i1]; // 현재 keyframe의 특정 node id에 속한 i1번째 keypoint index

                MapPoint* pMP1 = vpMapPoints1[idx1]; // 현재 keyframe의 idx1번째 map point
                if(!pMP1) // 해당 map point가 존재하지 않는다면,
                    continue; // 해당 루프의 끝으로 이동한다.
                if(pMP1->isBad()) // 해당 map point가 나쁘다고 판단하면,
                    continue; // 해당 루프의 끝으로 이동한다.

                const cv::Mat &d1 = Descriptors1.row(idx1); // 현재 keyframe 상의 idx1번째 keypoint의 descriptor

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++) // f2it->second : i번째 loop detection candidate keyframe의 특정 node id에 속한 keypoint index의 개수만큼 반복
                {
                    const size_t idx2 = f2it->second[i2]; // i번째 loop detection candidate keyframe의 특정 node id에 속한 i2번째 keypoint index

                    MapPoint* pMP2 = vpMapPoints2[idx2]; // i번째 loop detection candidate keyframe의 idx2번째 map point

                    if(vbMatched2[idx2] || !pMP2) // 이미 idx2번째 map point와의 correspondence가 존재하거나(vbMatched2[idx2] = true), 해당 map point가 존재하지 않는다면,
                        continue; // 해당 루프의 끝으로 이동한다.

                    if(pMP2->isBad()) // 해당 map point가 나쁘다고 판단하면,
                        continue; // 해당 루프의 끝으로 이동한다.

                    const cv::Mat &d2 = Descriptors2.row(idx2); // i번째 loop detection candidate keyframe 상의 idx2번째 keypoint의 descriptor

                    int dist = DescriptorDistance(d1,d2); // 현재 keyframe 상의 idx1번째 keypoint의 descriptor - i번째 loop detection candidate keyframe 상의 idx2번째 keypoint의 descriptor

                    if(dist<bestDist1) // 현재 keyframe의 descriptor와 i번째 loop detection candidate keyframe의 descriptor 간의 가장 작은 거리
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2) // 현재 keyframe의 descriptor와 i번째 loop detection candidate keyframe의 descriptor 간의 두 번째로 작은 거리
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW) // TH_LOW = 50
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)) // 가장 작은 거리는 두 번째로 작은 거리와 꽤 큰 차이를 보여야 한다.
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2]; // 현재 keyframe의 idx1번째 map point - i번째 loop detection candidate keyframe의 bestIdx2번째 map point correspondences
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation) // mbCheckOrientation
                        {
                            // rotation histogram 만들기
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first) // 현재 keyframe의 node id < i번째 loop candidate keyframe의 node id
        {
            f1it = vFeatVec1.lower_bound(f2it->first); // i번째 loop candidate keyframe의 node id를 하한선으로 잡고, 그 아래의 node를 search한다.
        }
        else // i번째 loop candidate keyframe의 node id < 현재 keyframe의 node id
        {
            f2it = vFeatVec2.lower_bound(f1it->first); // 현재 keyframe의 node id를 하한선으로 잡고, 그 아래의 node를 search한다.
        }
    }

    if(mbCheckOrientation) // mbCheckOrientation = true
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--; // rotation consistency를 만족하지 않는 correspondence 삭제
            }
        }
    }

    return nmatches; // 현재 keyframe의 idx1번째 map point - i번째 loop detection candidate keyframe의 bestIdx2번째 map point correspondences
}

// Data association -> 현재 keyframe의 i번째 keypoint index, 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index
// SearchForTriangulation(mpCurrentKeyFrame,pKF2,F12,vMatchedIndices,false)
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec; // keyframe 1의 feature vector
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec; // keyframe 2의 feature vector

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter(); // world 좌표계 상에서의 keyframe 1(현재 keyframe)의 위치, keyframe 1의 camera to world coordinate의 translation
    cv::Mat R2w = pKF2->GetRotation(); // keyframe 2(neighbor keyframe)의 world to camera coordinate의 rotation
    cv::Mat t2w = pKF2->GetTranslation(); // keyframe 2(neighbor keyframe)의 world to camera coordinate의 translation
    cv::Mat C2 = R2w*Cw+t2w; // keyframe 1의 camera to keyframe 2의 camera coordinate의 translation
    // keyframe 2의 world to camera cooridnate x keyframe 1의 camera to world coordinate의 translation = keyframe 2(neighbor keyframe) 상의 keyframe 1(현재 keyframe)의 위치

    // pixel 좌표계 -> world 좌표계 상의 keyframe 1의 camera 좌표계 위치를 keyframe 2의 pixel 좌표계로 투영
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy; 

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false); // keyframe 2(neighbor keyframe)의 keypoint 개수만큼 false로 초기화
    vector<int> vMatches12(pKF1->N,-1); // keyframe 1(현재 keyframe)의 keypoint 개수만큼 -1로 초기화

    vector<int> rotHist[HISTO_LENGTH]; // rotHist[30]
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500); // rotHist[30][500]

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin(); // keyframe 1(현재 keyframe)의 feature vector의 반복자
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin(); // keyframe 2(neighbor keyframe)의 feature vector의 반복자
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        // f1it->first : node id, f2it->first : node id -> BoW의 두 번째 기능인 Data Association
        if(f1it->first == f2it->first) // keyframe 1의 feature vector 상의 특정 node = keyframe 2의 feature vector 상의 특정 node
        {
            // f1it->second : keyframe 1의 feature vector 상에서, 특정 node에 존재하는 여러 keypoint의 index
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1]; // 특정 node에 존재하는 keypoint의 index
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1); // 현재 keyframe 상의 idx1번째 keypoint와 association의 관계를 가지는 map point
                
                // If there is already a MapPoint skip
                // 해당 과정은 현재 keyframe과 neighbor keyframe의 keypoint association 관계를 통해 map point를 형성하는 것인데, 이미 map point가 존재한다면 이를 통해 더 이상 map point를 만들 수 없기 때문
                if(pMP1) // pMP1이 존재한다면,
                    continue; // 해당 루프의 끝으로 이동한다.

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0; // pKF1->mvuRight[idx1] >= 0 -> bStereo1 = true

                // bOnlyStereo = false
                if(bOnlyStereo) // bOnlyStereo = true
                    if(!bStereo1) // bStereo1 = false
                        continue; // 해당 루프의 끝으로 이동한다.
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1]; // 현재 keyframe 상의 idx1번째 keypoint
                
                // 현재 keyframe 상의 idx1번째 keypoint에 대한 descriptor
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1); // pKF1->mDescriptors : 현재 keyframe 상의 모든 keypoint에 대한 descriptor
                
                int bestDist = TH_LOW; // TH_LOW = 50 -> 하한선
                int bestIdx2 = -1;
                
                // f2it->second : keyframe 2의 feature vector 상에서, 특정 node에 존재하는 여러 keypoint의 index
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2]; // 특정 node에 존재하는 keypoint의 index
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2); // neighbor keyframe 상의 idx2번째 keypoint와 association 관계를 가지는 map point
                    
                    // If we have already matched or there is a MapPoint skip
                    // keyframe 2의 idx2번째 keypoint에 해당하는 keyframe 1의 keypoint가 존재하거나, 해당 keypoint로 형성된 map point가 이미 존재한다면,
                    if(vbMatched2[idx2] || pMP2) // vbMatched2[idx2] = true or pMP2 != NULL
                        continue; // 해당 루프의 끝으로 이동한다.

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0; // pKF2->mvuRight[idx2] >= 0 -> bStereo2 = true

                    // bOnlyStereo = false
                    if(bOnlyStereo) // bOnlyStereo = true
                        if(!bStereo2) // bStereo2 = false
                            continue; // 해당 루프의 끝으로 이동한다.
                    
                    // neighbor keyframe 상의 idx2번째 keypoint에 대한 descriptor
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2); // pKF2->mDescriptors : neighbor keyframe 상의 모든 keypoint에 대한 descriptor
                    
                    const int dist = DescriptorDistance(d1,d2); // 현재 keyframe 상의 idx1번째 keypoint에 대한 descriptor - neighbor keyframe 상의 idx2번째 keypoint에 대한 descriptor
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue; // 해당 루프의 끝으로 이동한다.

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2]; // neighbor keyframe 상의 idx2번째 keypoint

                    if(!bStereo1 && !bStereo2) // bStereo1 = false and bStereo2 = false -> Q. monocular?
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }
                    // kp1, kp2 -> kp1은 현재 keyframe 상의 idx1번째 keypoint, kp2는 neighbor keyframe 상의 idx2번째 keypoint
                    // 현재 keyframe 상의 idx1번째 keypoint kp1에 fundamental matrix를 곱하여 얻은 neighbor keyframe 상의 epipolar line과 neighbor keyframe 상의 idx2번째 keypoint와의 거리
                    // neighbor keyframe 상의 epipolar line과 neighbor keyframe 상의 idx2번째 keypoint인 kp2와의 거리가 특정 threshold보다 낮다면,
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2)) // CheckDistEpipolarLine(kp1,kp2,F12,pKF2) = true
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0) // bestIdx2가 존재한다면,
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2]; // 현재 keyframe 상의 idx1번째 keypoint에 가장 유사한 neighbor keyframe 상의 keypoint
                    vMatches12[idx1]=bestIdx2; // 현재 keyframe 상의 idx1번째 keypoint에 가장 유사한 neighbor keyframe 상의 keypoint index
                    nmatches++; // 현재 keyframe 상의 idx1번째 keypoint에 가장 유사한 neighbor keyframe 상의 keypoint의 개수

                    // rotation histogram을 만드는 과정
                    if(mbCheckOrientation) // mbCheckOrientation = true
                    {
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        // Q. 완벽하게 일치하는 node가 없는 경우?
        else if(f1it->first < f2it->first) // 현재 keyframe의 feature vector 상의 node id가 neighbor keyframe의 feature vector 상의 node id보다 작다면,
        {
            // neighbor keyframe의 feature vector 상의 node id가 더 큰 경우, 현재 keyframe의 node id로 neighbor keyframe의 node id를 하한선으로 잡고 search 한다.
            f1it = vFeatVec1.lower_bound(f2it->first); // Finds the beginning of a subsequence matching given key
        }
        else // f1it->first > f2it->first -> 현재 keyframe의 feature vector 상의 node id가 neighbor keyframe의 feature vector 상의 node id보다 크다면,
        {
            // 현재 keyframe의 feature vector 상의 node id가 더 큰 경우, neighbor keyframe의 node id로 현재 keyframe의 node id를 하한선으로 잡고 search 한다.
            f2it = vFeatVec2.lower_bound(f1it->first); // Finds the beginning of a subsequence matching given key
        }
    }

    if(mbCheckOrientation) // mbCheckOrientation = true
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3); // rotation histogram에서 첫 번째, 두 번째, 세 번째로 가장 많은 빈도수를 갖는 word의 index를 추출한다.

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    // vector<pair<size_t, size_t> > &vMatchedPairs
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches); // nmatches의 개수만큼 vector에 메모리 할당

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0) // rotation consistency 조건을 만족하지 않은 경우 -> vMatches12[rotHist[i][j]] = -1
            continue; // 해당 루프의 끝으로 이동한다.
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
        // i -> 현재 keyframe의 keypoint index, vMatches12[i] -> 현재 keyframe의 i번째 keypoint와 대응되는 neighbor keyframe의 keypoint index
    }

    return nmatches; // neighbor keyframe과 대응되는 현재 keyframe의 keypoint 개수
}

// matcher.Fuse(pKFi,vpMapPointMatches) -> pKFi : neighbor keyframe, vpMapPointMatches : 현재 keyframe 상의 keypoint와 association의 관계를 가지는 map point
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation(); // neighbor keyframe의 world to camera coordinate의 rotation
    cv::Mat tcw = pKF->GetTranslation(); // neighbor keyframe의 world to camera coordinate의 translation

    // neighbor keyframe의 intrinsic parameter
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter(); // world 좌표계 상에서의 camera의 위치, camera to world coordinate의 translation

    int nFused=0;

    const int nMPs = vpMapPoints.size(); // 현재 keyframe 상의 keypoint 개수

    // 현재 keyframe의 map point와 neighbor keyframe의 keypoint와의 correspondence를 찾는 과정
    for(int i=0; i<nMPs; i++) // 현재 keyframe 상의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = vpMapPoints[i]; // 현재 keyframe 상의 i번째 keypoint에 해당하는 map point

        if(!pMP) // 해당 map point가 존재하지 않는다면,
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF)) // 해당 map point가 나쁘다고 판단하거나, 해당 map point가 neighbor keyframe에서 발견되었다면(이미 생성된 map point이기 때문),
        // pMP->IsInKeyFrame(pKF) : 해당 map point가 특정 keyframe에서 발견되었다면 true, 발견되지 않았다면 false
            continue; // 해당 루프의 끝으로 이동한다.

        // 현재 keyframe 상의 keypoint와 association 관계를 가지는 map point를 neighbor keyframe으로 projection
        cv::Mat p3Dw = pMP->GetWorldPos(); // world 좌표계 상의 map point의 position
        cv::Mat p3Dc = Rcw*p3Dw + tcw; // world to camera coordinate x world 좌표계 상의 map point의 position = camera 좌표계 상의 map point의 position

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f) // camera 좌표계 상의 map point의 z값 < 0 
            continue; // 해당 루프의 끝으로 이동한다.

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz; // normalized plane 상의 metric 좌표
        const float y = p3Dc.at<float>(1)*invz; // normalized plane 상의 metric 좌표

        const float u = fx*x+cx; // metric 좌표계 -> pixel 좌표계
        const float v = fy*y+cy; // metric 좌표계 -> pixel 좌표계

        // Point must be inside the image
        if(!pKF->IsInImage(u,v)) // pixel 좌표인 x와 y가 image boundary에 있는 경우 return true, image boundary 밖에 있는 경우 return false
        // pKF->IsInImage(u,v) = false
            continue; // 해당 루프의 끝으로 이동한다.

        const float ur = u-bf*invz; // right image

        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        cv::Mat PO = p3Dw-Ow; // world 좌표계 상의 map point의 position - world 좌표계 상의 neighbor keyframe의 position
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance ) // (map point - neighbor keyframe < 해당 map point가 가질 수 있는 가장 작은 distance) or (map point - neighbor keyframe > 해당 map point가 가질 수 있는 가장 큰 distance)
            continue; // 해당 루프의 끝으로 이동한다.

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // 해당 map point의 mean viewing direction

        if(PO.dot(Pn)<0.5*dist3D) // (map point - neighbor keyframe) - map point의 mean viewing direction > 60
            continue; // 해당 루프의 끝으로 이동한다.

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF); // Q. 

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // search window의 size 결정

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius); // Grid Search를 통해 현재 keyframe의 keypoint와 유사한 neighbor keyframe 상의 keypoint를 찾는다.

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        // Match to the most similar keypoint in the radius

        // 현재 keyframe 상의 i번째 keypoint에 해당하는 map point의 descriptor
        const cv::Mat dMP = pMP->GetDescriptor(); // 하나의 map point가 관측되는 keyframe의 keypoint descriptor, 하나의 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> 현재 keyframe의 keypoint와 유사한 neighbor keyframe 상의 keypoint index

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx]; // 현재 keyframe의 keypoint와 유사한 neighbor keyframe 상의 keypoint

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue; // 해당 루프의 끝으로 이동한다.

            if(pKF->mvuRight[idx]>=0) // stereo
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x; // 현재 keyframe의 keypoint와 유사한 neighbor keyframe 상의 keypoint의 x 좌표 (pixel 좌표계)
                const float &kpy = kp.pt.y; // 현재 keyframe의 keypoint와 유사한 neighbor keyframe 상의 keypoint의 y 좌표 (pixel 좌표계)
                const float &kpr = pKF->mvuRight[idx]; 
                const float ex = u-kpx; // 현재 keyframe의 keypoint를 neighbor keyframe 상에 projection한 x 좌표 - Grid Search를 통해 찾은 neighbor keyframe 상의 keypoint의 x 좌표
                const float ey = v-kpy; // 현재 keyframe의 keypoint를 neighbor keyframe 상에 projection한 y 좌표 - Grid Search를 통해 찾은 neighbor keyframe 상의 keypoint의 y 좌표
                const float er = ur-kpr; // right image
                const float e2 = ex*ex+ey*ey+er*er; // reprojection error

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue; // 해당 루프의 끝으로 이동한다.
            }
            else // Q. monocular
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx); // neighbor keyframe의 idx번째 keypoint의 descriptor

            const int dist = DescriptorDistance(dMP,dKF); // 현재 keyframe이 생성한 map point의 descriptor와 neighbor keyframe의 idx번째 keypoint의 descirptor와의 distance

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW) // TH_LOW = 50
        {
            // 현재 keyframe의 bestIdx번째 map point와 가장 correspondence 관계가 높은 neighbor keyframe의 keypoint와 일치하는 neighbor keyframe의 map point
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx); // neighbor keyframe 상의 bestIdx번째 keypoint와 association의 관계를 가지는 map point
            if(pMPinKF) // pMPinKF != NULL, -> neighbor keyframe의 bestIdx번째 keypoint와 일치하는 map point가 이미 존재한다면, Fusion 수행
            {
                if(!pMPinKF->isBad()) // 해당 map point가 나쁘다고 판단하지 않는다면,
                {
                    if(pMPinKF->Observations()>pMP->Observations()) // bestIdx번째 map point와 일치하는 keyframe의 keypoint 개수 > 현재 keyframe의 map point와 일치하는 keyframe의 keypoint 개수
                        pMP->Replace(pMPinKF); // 새롭게 생성한 map point를 neighbor keyframe에서 이미 존재하는 map point로 대체
                    else // pMPinKF->Observations() <= pMP->Observations()
                        pMPinKF->Replace(pMP); // neighbor keyframe에서 이미 존재하는 map point를 새롭게 생성한 map point로 대체
                }
            }
            else // pMPinKF = NULL -> neighbor keyframe의 bestIdx번째 keypoint와 일치하는 map point가 존재하지 않는다면, 새롭게 생성한 map point에 대한 processing
            {
                pMP->AddObservation(pKF,bestIdx); // Data Association -> 해당 map point가 neighbor keyframe에서 발견된 bestIdx번째 keypoint임을 저장한다.
                pKF->AddMapPoint(pMP,bestIdx); // 해당 map point를 neighbor keyframe의 mvpMapPoints vector에 귀속시킨다.
            }
            nFused++; // 새롭게 생성한 map point와 neighbor keyframe에 이미 존재하는 map point와의 Fusion + neighbor keyframe에 존재하지 않으면서도, Grid Search를 통해 찾은 현재 keyframe의 map point와의 correspondence
        }
    }

    return nFused;
}

// 현재 keyframe의 map point와 겹치는 loop detection keyframe의 map point를 구하는 과정
// matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints)
// pKF -> 현재 keyframe 또는 현재 keyframe의 neighbor keyframe
// cvScw -> 현재 keyframe 또는 현재 keyframe의 neighbor keyframe의 보정된 world to camera coordinate의 pose
// mvpLoopMapPoints -> loop detection keyframe과 neighbor keyframe들의 map point들
int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    // 현재 keyframe 또는 현재 keyframe의 neighbor keyframe의 intrinsic parameter
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3); // 보정된 world to camera coordinate의 rotation -> determinant(sRcw) = scale
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0))); // scale
    cv::Mat Rcw = sRcw/scw; // 보정된 world to camera coordinate의 rotation -> determinant(Rcw) = 1
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw; // 보정된 world to camera coordinate의 translation
    cv::Mat Ow = -Rcw.t()*tcw; // 보정된 camera to world coordinate의 translation

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints(); // mvpMapPoints -> 해당하는 keyframe 상의 keypoint와 association 관계가 있는 모든 map point -> Q. 보정된 map points?

    int nFused=0;

    const int nPoints = vpPoints.size(); // loop detection keyframe과 neighbor keyframe들의 map point들의 개수

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++) // loop detection keyframe과 neighbor keyframe들의 map point들의 개수만큼 반복
    {
        MapPoint* pMP = vpPoints[iMP]; // loop detection keyframe과 neighbor keyframe들의 iMP번째 map point

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP)) // 해당하는 map point가 나쁘다고 판단하거나, 이미 찾은 correspondence라면,
            continue; // 해당 루프의 끝으로 이동한다.

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw; // 보정된 world to camera coordinate의 rotation x world 좌표계 상의 map point의 position = camera 좌표계 상의 map point의 position

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f) // camera 좌표계 상의 map point의 position의 z 값 < 0.0
            continue; // 해당 루프의 끝으로 이동한다.

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        // loop detection keyframe과 neighbor keyframe의 map points를 현재 keyframe과 neighbor keyframe에 projection
        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v)) // pixel 좌표인 x와 y가 image boundary에 있는 경우 return true, image boundary 밖에 있는 경우 return false
        // pKF->IsInImage(u,v) = false -> loop detection keyframe과 neighbor keyframe의 map points를 현재 keyframe과 neighbor keyframe에 projection한 좌표가 image boundary를 벗어나는 경우
            continue; // 해당 루프의 끝으로 이동한다.

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        cv::Mat PO = p3Dw-Ow; // world 좌표계 상의 map point의 position - world 좌표계 상의 keyframe의 position(보정된 camera to world coordinate의 translation)
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue; // 해당 루프의 끝으로 이동한다.

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal(); // 해당 map point의 mean viewing direction

        if(PO.dot(Pn)<0.5*dist3D) // 해당 map point의 mean viewing direction과 map point - keyframe 벡터가 이루는 각도가 60도를 초과하면,
            continue; // 해당 루프의 끝으로 이동한다.

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel]; // search window의 크기

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);
        // loop detection keyframe(neighbor keyframe)을 현재 keyframe(neighbor keyframe)에 투영하여 loop detection keyframe과 현재 keyframe의 map point correspondence를 찾는다.
        // vIndices -> loop detection keyframe의 map point와 대응되는 현재 keyframe의 keypoint index

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit; // de-reference -> loop detection keyframe의 map point와 대응되는 현재 keyframe의 keypoint index
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue; // 해당 루프의 끝으로 이동한다.

            const cv::Mat &dKF = pKF->mDescriptors.row(idx); // 현재 keyframe의 idx번째 keypoint의 descriptor

            int dist = DescriptorDistance(dMP,dKF); // 현재 keyframe의 idx번째 keypoint의 descriptor - loop detection keyframe의 map point의 representative descriptor

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW) // TH_LOW = 50
        {
            // bestIdx -> 현재 keyframe에 projection된 loop detection keyframe의 map point와 가장 유사한 현재 keyframe의 keypoint index
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx); // 현재 keyframe 상의 bestIdx번째 keypoint와 association의 관계를 가지는 map point
            if(pMPinKF) // 현재 keyframe 상의 bestIdx번째 keypoint에 해당하는 map point가 존재한다면,
            {
                if(!pMPinKF->isBad()) // 해당 map point가 나쁘다고 판단하지 않으면,
                    vpReplacePoint[iMP] = pMPinKF; // Grid Search를 통해 찾은 loop detection keyframe과의 correspondence로 대체한다.
                    // vpReplacePoint->first : loop detection의 map point index, 
                    // vpReplacePoint->second : loop detection keyframe과 correspondence의 관계를 갖는 현재 keyframe의 map point
            }
            else // 현재 keyframe 상의 bestIdx번째 keypoint에 해당하는 map point가 존재하지 않는다면,
            {
                // loop detection keyframe의 map point에 대한 처리
                pMP->AddObservation(pKF,bestIdx); // Data Association -> loop detection keyframe의 map point가 현재 keyframe에서 발견된 bestIdx번째 keypoint임을 저장한다.
                pKF->AddMapPoint(pMP,bestIdx); // loop detection keyframe의 map point를 현재 keyframe의 mvpMapPoints vector에 귀속시킨다.
            }
            nFused++; // loop detection keyframe의 map point들을 현재 keyframe에 projection 하여 찾은 loop detection keyframe과 현재 keyframe의 map point correspondence의 개수
        }
    }

    return nFused;
}

// matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5)
// s, R, t -> loop detection candidate keyframe to current keyframe coordinate의 similarity transformation
// vpMatches12 -> 현재 keyframe의 map point - i번째 loop detection candidate keyframe의 map point correspondences
// Sim3Solver를 통해 projection하여, 현재 keyframe의 map point와 loop detection candidate keyframe의 map point와의 더 많은 correspondence를 새롭게 찾는다.
int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    // 현재 keyframe의 intrinsic parameter
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation(); // 현재 keyframe의 world to camera coordinate의 rotation
    cv::Mat t1w = pKF1->GetTranslation(); // 현재 keyframe의 world to camera coordinate의 translation

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation(); // i번째 loop detection candidate keyframe의 world to camera coordinate의 rotation
    cv::Mat t2w = pKF2->GetTranslation(); // i번째 loop detection candidate keyframe의 world to camera coordinate의 translation

    //Transformation between cameras
    cv::Mat sR12 = s12*R12; // loop detection candidate keyframe to current keyframe coordinate의 similarity를 고려한 rotation
    cv::Mat sR21 = (1.0/s12)*R12.t(); // current keyframe to loop detection candidate keyframe의 similarity를 고려한 rotation
    cv::Mat t21 = -sR21*t12; // current keyframe to loop detection candidate keyframe의 similarity를 고려한 translation

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches(); // 현재 keyframe 상의 keypoint와 association의 관계를 가지는 map point
    const int N1 = vpMapPoints1.size(); // 현재 keyframe의 keypoint 개수

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches(); // i번째 loop detection candidate keyframe 상의 keypoint와 association의 관계를 가지는 map point
    const int N2 = vpMapPoints2.size(); // i번째 loop detection candidate keyframe의 keypoint 개수

    vector<bool> vbAlreadyMatched1(N1,false); // 현재 keyframe의 keypoint 개수만큼 false로 초기화
    vector<bool> vbAlreadyMatched2(N2,false); // i번째 loop detection candidate keyframe의 keypoint 개수만큼 false로 초기화

    for(int i=0; i<N1; i++) // 현재 keyframe의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = vpMatches12[i]; // vpMatches12 -> 현재 keyframe의 i번째 map point - i번째 loop detection candidate keyframe의 i번째 map point correspondences
        if(pMP) // 해당 map point가 존재하면,
        {
            vbAlreadyMatched1[i]=true; // 이미 찾아진 correspondence에 대하여,
            int idx2 = pMP->GetIndexInKeyFrame(pKF2); // 해당 map point와 association의 관계를 가지는 i번째 loop detection candidate keyframe의 keypoint index
            if(idx2>=0 && idx2<N2) // 0 <= idx2 < N2
                vbAlreadyMatched2[idx2]=true; // 이미 찾아진 correspondence에 대하여,
        }
    }

    vector<int> vnMatch1(N1,-1); // 현재 keyframe의 keypoint 개수만큼 -1로 초기화
    vector<int> vnMatch2(N2,-1); // i번째 loop detection candidate keyframe의 keypoint 개수만큼 -1로 초기화

    // Sim3Solver를 통해 구한 similarity transformation을 이용하여, Grid Search를 통해 더 많은 correspondence를 찾는다.
    // Transform from KF1 to KF2 and search
    
    // 현재 keyframe의 map point -> loop detection candidate keyframe으로 projection
    for(int i1=0; i1<N1; i1++) // 현재 keyframe의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = vpMapPoints1[i1]; // 현재 keyframe의 i1번째 map point

        if(!pMP || vbAlreadyMatched1[i1]) // 해당 map point가 존재하지 않거나, 이미 찾아진 correspondences에 해당한다면,
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단하면,
            continue; // 해당 루프의 끝으로 이동한다.

        cv::Mat p3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position
        cv::Mat p3Dc1 = R1w*p3Dw + t1w; // 현재 keyframe의 world to camera coordinate의 transformation x world 좌표계 상의 map point의 position = 현재 keyframe 상의 map point position
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21; // (Sim3Solver를 통해 계산한) current keyframe to loop detection candidate keyframe coordinate의 transformation x 현재 keyframe 상의 map point position
        // = loop detection candidate keyframe 상의 map point position

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0) // loop detection candidate keyframe 상의 map point position의 z 값 < 0.0
            continue; // 해당 루프의 끝으로 이동한다.

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v)) // pixel 좌표인 x와 y가 image boundary에 있는 경우 return true, image boundary 밖에 있는 경우 return false
        // pKF2->IsInImage(u,v) = false -> map point를 loop detection candidate keyframe으로 projection 하였을 때, image boundary를 벗어나는 경우
            continue; // 해당 루프의 끝으로 이동한다.

        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        const float dist3D = cv::norm(p3Dc2); // camera 좌표계 원점에서 해당 map point로의 거리

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue; // 해당 루프의 끝으로 이동한다.

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel]; // search window의 크기

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius); // map point를 loop detection candidate keyframe으로 projection하여 Grid Search를 통해, 해당 map point와 일치하는 keypoint를 search
        // vIndices -> map point와 correspondence의 관계가 있는 loop detection candidate keyframe의 keypoint index

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> map point와 correspondence의 관계가 있는 loop detection candidate keyframe의 keypoint index

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx]; // loop detection candidate keyframe의 idx번째 keypoint

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue; // 해당 루프의 끝으로 이동한다.

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx); // loop detection candidate keyframe의 idx번째 keypoint의 descriptor

            const int dist = DescriptorDistance(dMP,dKF); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor - loop detection candidate keyframe의 idx번째 keypoint의 descriptor

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH) // TH_HIGH = 100
        {
            vnMatch1[i1]=bestIdx; // 현재 keyframe의 i1번째 map point와 correspondence의 관계를 갖는 loop detection candidate keyframe의 keypoint index
        }
    }

    // Transform from KF2 to KF2 and search

    // loop detection candidate keyframe의 map point -> 현재 keyframe으로 projection
    for(int i2=0; i2<N2; i2++) // loop detection candidate keyframe의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = vpMapPoints2[i2]; // loop detection candidate keyframe의 i2번째 map point

        if(!pMP || vbAlreadyMatched2[i2]) // 해당 map point가 존재하지 않거나, 이미 찾아진 correspondences에 해당한다면,
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단하면,
            continue; // 해당 루프의 끝으로 이동한다.

        cv::Mat p3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position
        cv::Mat p3Dc2 = R2w*p3Dw + t2w; // loop detection candidate keyframe의 world to camera coordinate의 transformation x world 좌표계 상의 map point의 position = loop detection candidate keyframe 상의 map point position
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12; // (Sim3Solver를 통해 계산한) loop detection candidate keyframe to current keyframe coordinate의 transformation x loop detection candidate keyframe 상의 map point position
        // = current keyframe 상의 map point position

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0) // current keyframe 상의 map point position의 z 값 < 0.0
            continue; // 해당 루프의 끝으로 이동한다.

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v)) // pixel 좌표인 x와 y가 image boundary에 있는 경우 return true, image boundary 밖에 있는 경우 return false
        // pKF1->IsInImage(u,v) = false -> map point를 현재 keyframe으로 projection 하였을 때, image boundary를 벗어나는 경우
            continue; // 해당 루프의 끝으로 이동한다.

        const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
        const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance
        const float dist3D = cv::norm(p3Dc1); // camera 좌표계 원점에서 해당 map point로의 거리

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue; // 해당 루프의 끝으로 이동한다.

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel]; // search window의 크기

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius); // map point를 현재 keyframe으로 projection하여 Grid Search를 통해, 해당 map point와 일치하는 keypoint를 search
        // vIndices -> map point와 correspondence의 관계가 있는 현재 keyframe의 keypoint index

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> map point와 correspondence의 관계가 있는 현재 keyframe의 keypoint index

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx]; // 현재 keyframe의 idx번째 keypoint

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue; // 해당 루프의 끝으로 이동한다.

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx); // 현재 keyframe의 idx번째 keypoint의 descriptor

            const int dist = DescriptorDistance(dMP,dKF); // 해당 map point를 가장 잘 표현할 수 있는 representative descriptor - 현재 keyframe의 idx번째 keypoint의 descriptor

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH) // TH_HIGH = 100
        {
            vnMatch2[i2]=bestIdx; // loop detection candidate keyframe의 i2번째 map point와 correspondence의 관계를 갖는 현재 keyframe의 keypoint index
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++) // 현재 keyframe의 keypoint 개수만큼 반복
    {
        int idx2 = vnMatch1[i1]; // 현재 keyframe의 i1번째 map point와 correspondence의 관계를 갖는 loop detection candidate keyframe의 keypoint index

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2]; // loop detection candidate keyframe의 i2번째 map point와 correspondence의 관계를 갖는 현재 keyframe의 keypoint index
            if(idx1==i1) // 현재 keyframe의 map point = loop detection candidate keyframe의 map point
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound; // 현재 keyframe의 map point와 loop detection candidate keyframe의 map point와의 새롭게 찾아진 correspondence
}

// SearchByProjection(mCurrentFrame,mLastFrame,th,mSensor==System::MONOCULAR)
// 지난 frame의 map points를 현재 frame에 projection
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
    int nmatches = 0;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH]; // HISTO_LENGTH = 30 -> rotHist[30] -> 30개의 word로 구성된 rotation histogram
    for(int i=0;i<HISTO_LENGTH;i++) // 30번 반복
        rotHist[i].reserve(500); // rotHist[30][500]
    const float factor = 1.0f/HISTO_LENGTH;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3); // 현재 frame의 world to camera coordinate의 rotation
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3); // 현재 frame의 world to camera coordinate의 translation

    const cv::Mat twc = -Rcw.t()*tcw; // 현재 frame의 camera to world coordinate의 translation -> world 좌표계 상에서의 camera position

    const cv::Mat Rlw = LastFrame.mTcw.rowRange(0,3).colRange(0,3); // 지난 frame의 world to camera coordinate의 rotation 
    const cv::Mat tlw = LastFrame.mTcw.rowRange(0,3).col(3); // 지난 frame의 world to camera coordinate의 translation

    const cv::Mat tlc = Rlw*twc+tlw; // from current camera to last camera coordinate의 translation

    const bool bForward = tlc.at<float>(2)>CurrentFrame.mb && !bMono; // mb -> stereo baseline in meters
    // from current camera to last camera coordinate의 z 값 > mb and stereo인 경우 -> bForward = true -> 전진?
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    // from current camera to last camera coordinate의 -z 값 > mb and stereo인 경우 -> bBackward = true -> 후진?

    for(int i=0; i<LastFrame.N; i++) // 지난 frame의 keypoint의 개수만큼 반복
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i]; // 지난 frame의 map points = pMP

        if(pMP) // pMP가 할당되어 있다면, Null 값이라면 False를 return
        {
            if(!LastFrame.mvbOutlier[i]) // 지난 frame의 해당 map point가 outlier가 아니라면,
            {
                // Project
                // world 좌표계
                cv::Mat x3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position -> last frame의 map point
                // camera 좌표계
                cv::Mat x3Dc = Rcw*x3Dw+tcw; // world to camera transformation x world 좌표계 상의 map point -> last frame의 map point를 현재 frame의 camera coordinate으로 옮긴다.
                // world 좌표계 상의 map point -> camera 좌표계 상의 point

                const float xc = x3Dc.at<float>(0); // camera 좌표계 상의 point x 좌표
                const float yc = x3Dc.at<float>(1); // camera 좌표계 상의 point y 좌표
                const float invzc = 1.0/x3Dc.at<float>(2); // camera 좌표계 상의 point z 좌표의 inverse

                if(invzc<0) // z 좌표 < 0
                    continue; // 해당 for문의 끝으로 이동한다.

                // pixel 좌표계
                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx; // xc*invzc : normalized plane 상의 x 좌표 -> pixel 좌표계 상의 x 좌표
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy; // yc*invzc : normalized plane 상의 y 좌표 -> pixel 좌표계 상의 y 좌표

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX) // 현재 frame의 image boundary를 벗어나면,
                    continue; // 해당 for문의 끝으로 이동한다.
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY) // 현재 frame의 image boundary를 벗어나면,
                    continue; // 해당 for문의 끝으로 이동한다.

                int nLastOctave = LastFrame.mvKeys[i].octave; // 지난 frame 상의 keypoint가 어떠한 scale에서 추출 되었는가
                // nLastOctave가 커질수록, scale은 커진다. scale이 크다는 것은 더 global하게 볼 수 있다는 것을 의미한다.
                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; // scale이 커짐에 따라 window size도 커진다.

                vector<size_t> vIndices2;

                // GetFeaturesInArea -> Grid Search를 통해 projected map point와 유사한 keypoint의 index를 구한다.
                if(bForward) // 전진하는 경우
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave); // minLevel = nLastOctave -> 전진하는 경우, 특정 feature에 더 가까워지기 때문
                else if(bBackward) // 후진하는 경우
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave); // minLevel = 0, maxLevel = nLastOctave -> 후진하는 경우, 특정 feature에서 더 멀어지기 때문
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1); // minLevel = nLastOctave-1, maxLevel = nLastOctave+1 -> feature들은 비슷한 scale을 가질 것이기 때문

                if(vIndices2.empty())
                    continue; // 해당 for문의 끝으로 이동한다.

                const cv::Mat dMP = pMP->GetDescriptor(); // 특정 map point가 관측되는 keyframe의 keypoint descriptor

                int bestDist = 256; // 가장 큰 descriptor distance
                int bestIdx2 = -1;
                
                // Grid Search를 통해 구한, projected map point와 유사한 keypoint의 개수만큼 반복
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit; // 반복자 pointer de-reference, i2 -> keypoint의 index
                    if(CurrentFrame.mvpMapPoints[i2]) // 해당 keypoint를 갖는 현재 frame의 map point가 존재한다면,
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0) // 해당 map point와 일치하는 keyframe의 keypoint가 존재한다면,
                            continue; // 해당 for문의 끝으로 이동한다.
                    
                    // right image의 keypoint에 대해서도 error 확인
                    if(CurrentFrame.mvuRight[i2]>0) // right image의 i2번째 keypoint가 존재한다면,
                    {
                        const float ur = u - CurrentFrame.mbf*invzc; // CurrentFrame.mbf : fx x baseline, u - CurrentFrame.mbf*invzc : right image의 pixel 좌표
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue; // 해당 for문의 끝으로 이동한다.
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2); // reprojected map points와 유사한 keypoint(같은 grid 내의 keypoint)의 descriptor

                    const int dist = DescriptorDistance(dMP,d); // map point의 descriptor - reprojected map points와 유사한 keypoint의 descriptor
                    
                    // reprojected map points와 유사한 keypoint의 descriptor를 찾는다.
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=TH_HIGH) // TH_HIGH = 100
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP; // 지난 frame의 map point -> 현재 frame의 mvpMapPoints vector
                    nmatches++; // 지난 frame의 map point와 겹치는 현재 frame의 map point 개수

                    // rotation histogram이 만들어지는 과정
                    if(mbCheckOrientation) // mbCheckOrientation = true
                    {
                        // 하나의 frame의 모든 keypoint와 다른 frame의 해당하는 keypoint가 회전하는 각도는 모두 비슷할 것이다.
                        float rot = LastFrame.mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle; // 지난 frame 상의 keypoint의 angle - 현재 frame 상의 keypoint의 angle
                        if(rot<0.0) // 현재 frame 상의 keypoint의 angle > 지난 frame 상의 keypoint의 angle
                            rot+=360.0f;
                        int bin = round(rot*factor); // rot / 30
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH); // assert(조건식) -> 조건식이 거짓일 경우 프로그램 실행 도중 프로그램을 중단한다.
                        rotHist[bin].push_back(bestIdx2); // 해당 keypoint의 index -> rotHist[bin]
                    }
                }
            }
        }
    }

    //Apply rotation consistency
    if(mbCheckOrientation) // mbCheckOrientation = true
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;
        
        // 하나의 frame의 모든 keypoint와 다른 frame의 해당하는 keypoint가 회전하는 각도는 모두 비슷할 것이다.
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3); // rotHist histogram에서 첫 번째로 빈도수가 높은 word의 index, 두 번째로 빈도수가 높은 word의 index,
        // 세 번째로 빈도수가 높은 word의 index를 추출한다.

        for(int i=0; i<HISTO_LENGTH; i++) // 30개의 word에 대하여 반복
        {
            if(i!=ind1 && i!=ind2 && i!=ind3) // 첫 번째로 빈도수가 높은 word, 두 번째로 빈도수가 높은 word, 세 번째로 빈도수가 높은 word에 속하지 않는다면,
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++) // 해당하는 word에 속하는 keypoint의 개수만큼 반복
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=static_cast<MapPoint*>(NULL); // rotHist[i][j] -> 해당하는 word에 속하는 keypoint의 index
                    nmatches--; // keypoint의 rotation 속성이 다르면, match에서 배제하라.
                }
            }
        }
    }

    return nmatches; // rotation consistency를 갖는 현재 frame의 map point 개수
}

// SearchByProjection(mCurrentFrame,vpCandidateKFs[i],sFound,10,100) -> vpCandidateKFs[i] : i번째 relocalization candidate keyframe,
// sFound : 특정 keyframe의 map points와 유사한 현재 frame의 map points 중 inlier points, th = 10, ORBdist = 100
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3); // 현재 frame의 world to camera coordinate pose의 rotation
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3); // 현재 frame의 world to camera coordinate pose의 translation
    const cv::Mat Ow = -Rcw.t()*tcw; // 현재 frame의 camera to world coordinate pose의 translation -> world 좌표계 상의 camera position

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH]; // rotHist[30] -> rotation histogram
    for(int i=0;i<HISTO_LENGTH;i++) // 30만큼 반복
        rotHist[i].reserve(500); // rotHist[30][500]
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches(); // i번째 relocalization candidate keyframe 상의 keypoints에 해당하는 map points

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++) // i번째 relocalization candidate keyframe의 map points 개수만큼 반복
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP) // 해당 map point가 존재한다면,
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            // pMP->isBad() = false and sAlreadyFound.count(pMP) = false -> 해당 map point가 나쁘지 않다고 판단되고, 해당 map point가 이미 찾아지지 않았다면(이미 찾아진 correspondence이기 때문),
            {
                // keyframe과 현재 frame의 일치하는 map point를 현재 frame에 project
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position
                cv::Mat x3Dc = Rcw*x3Dw+tcw; // world to camera coordinate x world 좌표계 상의 map point = camera 좌표계 상의 map point

                const float xc = x3Dc.at<float>(0); // camera 좌표계 상의 x 좌표
                const float yc = x3Dc.at<float>(1); // camera 좌표계 상의 y 좌표
                const float invzc = 1.0/x3Dc.at<float>(2); // 1 / camera 좌표계 상의 z 좌표

                // pixel 좌표계
                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx; // xc*invzc : normalized plane 상의 x 좌표 -> pixel 좌표계 상의 x 좌표
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy; // yc*invzc : normalized plane 상의 y 좌표 -> pixel 좌표계 상의 y 좌표
                
                // image boundary를 벗어나는 map point 고려 x
                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue; // 해당 루프의 끝으로 이동한다.
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue; // 해당 루프의 끝으로 이동한다.

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow; // image optical center - map point
                float dist3D = cv::norm(PO); // normalized distance(image optical center - map point)
                
                // scale invariance distances
                const float maxDistance = pMP->GetMaxDistanceInvariance(); // 1.2f*mfMaxDistance
                const float minDistance = pMP->GetMinDistanceInvariance(); // 0.8f*mfMinDistance

                // Depth must be inside the scale pyramid of the image
                // normalized distance가 scale invariance distances를 벗어나는 map point 고려 x
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue; // 해당 루프의 끝으로 이동한다.

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame); // 해당 map point의 predicted scale

                // Search in a window
                // radius -> window size
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel]; // th * 해당 map point의 predicted scale의 scale factor = radius

                // (비슷한 scale 범위에서) reprojected map point에 해당하는 feature를 찾기 위해 grid를 통해 빠르게 search하는 함수
                // output : 해당 grid에 존재하는 keypoint의 index
                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty()) // grid search를 통해 찾은 keypoint가 없다면,
                    continue; // 해당 루프의 끝으로 이동한다.

                const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point가 관측되는 keyframe의 keypoint descriptor, 하나의 map point를 가장 잘 표현할 수 있는 representative descriptor

                int bestDist = 256; // descriptor의 최대 distance
                int bestIdx2 = -1;

                // Grid Search를 통해 찾은, i번째 relocalization candidate keyframe의 map points와 유사한 현재 frame의 keypoints index
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit; // de-reference -> keypoint의 index
                    // 이미 찾은 correspondence이기 때문,
                    if(CurrentFrame.mvpMapPoints[i2]) // 해당 (frame 상의 keypoint에 해당하는)map point가 존재하면,
                        continue; // 해당 루프의 끝으로 이동한다.

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2); // 현재 frame의 i2번째 descriptor

                    const int dist = DescriptorDistance(dMP,d); // map point의 representative descriptor - grid search로 찾은 keypoint descriptor
                    
                    // map point에 해당하는 keypoint descriptor에 가장 유사한, grid search로 찾은 keypoint descriptor
                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                // ORBdist = 100
                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP; // pMP -> 현재 frame의 mvpMapPoints vector
                    nmatches++;

                    // rotation histogram을 만드는 과정
                    if(mbCheckOrientation) // mbCheckOrientation = true
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle; // keyframe 상의 i번째 keypoint의 angle - 현재 frame 상의 i번째 keypoint의 angle
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor); // rot / 30
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH); // 0 <= bin < 30 -> index
                        rotHist[bin].push_back(bestIdx2); // rotHist[bin][500] -> bestIdx2, bestIdx2, ...
                    }
                }

            }
        }
    }

    if(mbCheckOrientation) // mbCheckOrientation = true
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3); // rotHist histogram에서 첫 번째로 빈도수가 높은 word, 두 번째로 빈도수가 높은 word, 세 번째로 빈도수가 높은 word의 index를 추출

        for(int i=0; i<HISTO_LENGTH; i++) // 30만큼 반복
        {
            if(i!=ind1 && i!=ind2 && i!=ind3) // 첫 번째 빈도수가 높은 word, 두 번째로 빈도수가 높은 word, 세 번째로 빈도수가 높은 word에 속하지 않는다면,
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++) // rotHist[i](i번째 word)에 속하는 모든 descriptor의 index
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

// histogram에서 3개의 가장 큰 값의 index를 추출한다.
// ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3)
// input : histogram, histogram의 length, output : max_index1, max_index2, max_index3
void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    // max1 > max2 > max3
    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size(); // i번째 word의 빈도수 -> s
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1) // max1이 max2보다 굉장히 큰 값으로 크다면,
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1) // max1이 max3보다 굉장히 큰 값으로 크다면,
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
// Q. 
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>(); // Mat::ptr -> 원소 값에 접근하기 위해, int32_t -> 4 byte의 정수형
    const int *pb = b.ptr<int32_t>(); // Mat::ptr -> 원소 값에 접근하기 위해, int32_t -> 4 byte의 정수형

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb; // ^ : XOR 연산
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM