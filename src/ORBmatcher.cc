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
int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0; // th != 1.0 -> bFactor = true, th = 1.0 -> bFactor = false

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++) // local map point의 개수만큼 반복
    {
        MapPoint* pMP = vpMapPoints[iMP]; // iMP번째 local map point -> pMP
        if(!pMP->mbTrackInView) // pMP->mbTrackInView = false -> Q.
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
            continue; // 해당 루프의 끝으로 이동한다.

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        // image optical center와 map point를 잇는 벡터와 map point의 mean viewing vector가 이루는 각도에 따라 search window의 size가 결정된다.
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor) // bFactor = true -> th != 1.0
            r*=th;

        // Grid Search -> local map point를 현재 frame에 projection하여, grid search를 통해 찾은 특정 grid 내의 keypoint들의 index
        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty()) // vIndices.empty() = true
            continue; // 해당 루프의 끝으로 이동한다.

        const cv::Mat MPdescriptor = pMP->GetDescriptor(); // 해당 map point가 관측되는 keyframe의 keypoint descriptor

        int bestDist=256; // descriptor distance의 최대값
        int bestLevel= -1;
        int bestDist2=256; // descriptor distance의 최대값
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit; // de-reference -> keypoint의 index

            if(F.mvpMapPoints[idx]) // grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 map point가 존재한다면,
                if(F.mvpMapPoints[idx]->Observations()>0) // 해당 map point를 관측하는 keyframe이 하나 이상 발견된다면,
                    continue; // 해당 루프의 끝으로 이동한다.

            if(F.mvuRight[idx]>0) // Q. RGB-D?
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx); // grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 descriptor

            const int dist = DescriptorDistance(MPdescriptor,d);
            // local map point의 descriptor와 grid search를 통해 찾은 keypoint의 index에 해당하는 현재 frame의 descriptor 간의 거리

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


bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

    if(den==0)
        return false;

    const float dsqr = num*num/den;

    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

// 모든 frame에 대한 BoW는 존재하지 않지만, 모든 keyframe에 대한 BoW는 존재한다.
// input : keyframe, frame, output : vpMapPointMatches(keyframe의 map points - frame의 map points matching), nmatches
// BoW의 두 번째 기능인 data association을 통해 현재 frame과 reference keyframe 간의 matching되는 map point를 찾는 함수
// 현재 frame의 map point에 matching되는 keyframe의 map point 개수를 찾는 함수
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

                    const int dist =  DescriptorDistance(dKF,dF); // 동일한 node id에 있는 frame의 map point의 descriptor - keyframe의 map point의 descriptor

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
                if(bestDist1<=TH_LOW) // TH_LOW = 50 -> 해당 조건을 만족해야 꽤 적당한 quality의 map point(keyframe의 map point와 겹치는 frame의 map point)를 얻었다고 할 수 있다.
                {
                    // bestDist1 < mfNNratio x bestDist2 -> 첫 번째 descriptor 후보의 distance는 두 번째 descriptor 후보의 distance와 꽤 큰 차이를 보여야 한다.
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2)) // bestDist1 < mfNNratio x bestDist2
                    {
                        vpMapPointMatches[bestIdxF]=pMP; // realIdxKF의 keypoint와 match되는 reference keyframe의 map point -> frame과 keyframe의 matching되는 map points
                        // bestIdxF=realIdxF;
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF]; // realIdxKF의 index 값을 가지는 keyframe 상의 keypoint
                        
                        if(mbCheckOrientation) // mbCheckOrientation = true
                        {
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

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
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

int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
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
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
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
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
    cv::Mat Cw = pKF1->GetCameraCenter();
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it!=f1end && f2it!=f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];
                
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
                if(pMP1)
                    continue;

                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;
                int bestIdx2 = -1;
                
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    size_t idx2 = f2it->second[i2];
                    
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
                    vMatches12[idx1]=bestIdx2;
                    nmatches++;

                    if(mbCheckOrientation)
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
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
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
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
    cv::Mat Rcw = pKF->GetRotation();
    cv::Mat tcw = pKF->GetTranslation();

    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;
    const float &bf = pKF->mbf;

    cv::Mat Ow = pKF->GetCameraCenter();

    int nFused=0;

    const int nMPs = vpMapPoints.size();

    for(int i=0; i<nMPs; i++)
    {
        MapPoint* pMP = vpMapPoints[i];

        if(!pMP)
            continue;

        if(pMP->isBad() || pMP->IsInKeyFrame(pKF))
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc = Rcw*p3Dw + tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        const float ur = u-bf*invz;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

            const int &kpLevel= kp.octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            if(pKF->mvuRight[idx]>=0)
            {
                // Check reprojection error in stereo
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float &kpr = pKF->mvuRight[idx];
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float er = ur-kpr;
                const float e2 = ex*ex+ey*ey+er*er;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>7.8)
                    continue;
            }
            else
            {
                const float &kpx = kp.pt.x;
                const float &kpy = kp.pt.y;
                const float ex = u-kpx;
                const float ey = v-kpy;
                const float e2 = ex*ex+ey*ey;

                if(e2*pKF->mvInvLevelSigma2[kpLevel]>5.99)
                    continue;
            }

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                {
                    if(pMPinKF->Observations()>pMP->Observations())
                        pMP->Replace(pMPinKF);
                    else
                        pMPinKF->Replace(pMP);
                }
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
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
    // from current camera to last camera coordinate의 z 값 > mb and stereo인 경우 -> bForward = true -> Q. 전진?
    const bool bBackward = -tlc.at<float>(2)>CurrentFrame.mb && !bMono;
    // from current camera to last camera coordinate의 -z 값 > mb and stereo인 경우 -> bBackward = true -> Q. 후진?

    for(int i=0; i<LastFrame.N; i++) // 지난 frame의 keypoint의 개수만큼 반복
    {
        MapPoint* pMP = LastFrame.mvpMapPoints[i]; // 지난 frame의 map points = pMP

        if(pMP) // pMP가 할당되어 있다면, Null 값이라면 False를 return
        {
            if(!LastFrame.mvbOutlier[i]) // 지난 frame의 해당 map point가 outlier가 아니라면,
            {
                // Project
                cv::Mat x3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position
                cv::Mat x3Dc = Rcw*x3Dw+tcw; // world to camera transformation x world 좌표계 상의 map point
                // world 좌표계 상의 map point -> camera 좌표계 상의 point

                const float xc = x3Dc.at<float>(0); // camera 좌표계 상의 point x 좌표
                const float yc = x3Dc.at<float>(1); // camera 좌표계 상의 point y 좌표
                const float invzc = 1.0/x3Dc.at<float>(2); // camera 좌표계 상의 point z 좌표의 inverse

                if(invzc<0) // z 좌표 < 0
                    continue; // 해당 for문을 빠져나가라.

                float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx; // xc*invzc : normalized plane 상의 x 좌표 -> pixel 좌표계 상의 x 좌표
                float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy; // yc*invzc : normalized plane 상의 y 좌표 -> pixel 좌표계 상의 y 좌표

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX) // 현재 frame의 image boundary를 벗어나면,
                    continue; // 해당 for문을 빠져나가라.
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY) // 현재 frame의 image boundary를 벗어나면,
                    continue; // 해당 for문을 빠져나가라.

                int nLastOctave = LastFrame.mvKeys[i].octave; // 지난 frame 상의 keypoint가 어떠한 scale에서 추출 되었는가

                // Search in a window. Size depends on scale
                float radius = th*CurrentFrame.mvScaleFactors[nLastOctave]; // Q.

                vector<size_t> vIndices2;

                // GetFeaturesInArea -> Grid Search를 통해 projected map point와 유사한 keypoint의 index를 구한다.
                if(bForward) // 전진하는 경우
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave); // minLevel = nLastOctave -> 전진하는 경우, 특정 feature에 더 가까워지기 때문
                else if(bBackward) // 후진하는 경우
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, 0, nLastOctave); // minLevel = 0, maxLevel = nLastOctave -> 후진하는 경우, 특정 feature에서 더 멀어지기 때문
                else
                    vIndices2 = CurrentFrame.GetFeaturesInArea(u,v, radius, nLastOctave-1, nLastOctave+1); // minLevel = nLastOctave-1, maxLevel = nLastOctave+1 -> feature들은 비슷한 scale을 가질 것이기 때문

                if(vIndices2.empty())
                    continue; // 해당 for문을 빠져나가라.

                const cv::Mat dMP = pMP->GetDescriptor(); // 특정 map point가 관측되는 keyframe의 keypoint descriptor

                int bestDist = 256; // 가장 큰 descriptor distance
                int bestIdx2 = -1;
                
                // Grid Search를 통해 구한, projected map point와 유사한 keypoint의 개수만큼 반복
                for(vector<size_t>::const_iterator vit=vIndices2.begin(), vend=vIndices2.end(); vit!=vend; vit++)
                {
                    const size_t i2 = *vit; // 반복자 pointer de-reference, i2 -> keypoint의 index
                    if(CurrentFrame.mvpMapPoints[i2]) // 해당 keypoint를 갖는 현재 frame의 map point가 존재한다면,
                        if(CurrentFrame.mvpMapPoints[i2]->Observations()>0) // 해당 map point와 일치하는 keyframe의 keypoint가 존재한다면,
                            continue; // 해당 for문을 빠져나가라.

                    if(CurrentFrame.mvuRight[i2]>0) // Q. RGB-D
                    {
                        const float ur = u - CurrentFrame.mbf*invzc;
                        const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
                        if(er>radius)
                            continue;
                    }

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2); // reprojected map points와 유사한 keypoint(같은 grid 내의 keypoint)의 descriptor

                    const int dist = DescriptorDistance(dMP,d); // map point의 descriptor - reprojected map points와 유사한 keypoint의 descriptor
                    
                    // 해당 map point의 descriptor에 가장 유사한, reprojected map points와 유사한 keypoint의 descriptor를 찾는다.
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

                    if(mbCheckOrientation) // mbCheckOrientation = true
                    {
                        // Q. rot의 의미?
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
        
        // Q. 빈도수가 높은 것이 아니라, rot이 가장 작은 경우만을 고려해야 하는 것이 아닌가?
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

    return nmatches; // 지난 frame의 map point와 일치하거나, 유사한 현재 frame의 map point의 개수
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
            // pMP->isBad() = false and sAlreadyFound.count(pMP) = false -> 해당 map point가 나쁘지 않다고 판단되고, 해당 map point가 이미 찾아지지 않았다면(중복 방지),
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 map point의 position
                cv::Mat x3Dc = Rcw*x3Dw+tcw; // world to camera coordinate x world 좌표계 상의 map point = camera 좌표계 상의 map point

                const float xc = x3Dc.at<float>(0); // camera 좌표계 상의 x 좌표
                const float yc = x3Dc.at<float>(1); // camera 좌표계 상의 y 좌표
                const float invzc = 1.0/x3Dc.at<float>(2); // 1 / camera 좌표계 상의 z 좌표

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

                const cv::Mat dMP = pMP->GetDescriptor(); // 해당 map point가 관측되는 keyframe의 keypoint descriptor

                int bestDist = 256; // descriptor의 최대 distance
                int bestIdx2 = -1;

                // Grid Search를 통해 찾은, i번째 relocalization candidate keyframe의 map points와 유사한 현재 frame의 keypoints index
                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit; // de-reference -> keypoint의 index
                    if(CurrentFrame.mvpMapPoints[i2]) // 해당 (frame 상의 keypoint에 해당하는)map point가 존재하면,
                        continue; // 해당 루프의 끝으로 이동한다.

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2); // 현재 frame의 i2번째 descriptor

                    const int dist = DescriptorDistance(dMP,d); // map point에 해당하는 keypoint descriptor - grid search로 찾은 keypoint descriptor
                    
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

                    if(mbCheckOrientation) // mbCheckOrientation = true
                    {
                        // Q. rot 값이 작을수록 correspondence의 확률이 더 커지는 것이 아닌가?
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
