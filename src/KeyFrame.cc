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

#include "KeyFrame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include<mutex>

namespace ORB_SLAM2
{

long unsigned int KeyFrame::nNextId=0;

KeyFrame::KeyFrame(Frame &F, Map *pMap, KeyFrameDatabase *pKFDB):
    mnFrameId(F.mnId),  mTimeStamp(F.mTimeStamp), mnGridCols(FRAME_GRID_COLS), mnGridRows(FRAME_GRID_ROWS),
    mfGridElementWidthInv(F.mfGridElementWidthInv), mfGridElementHeightInv(F.mfGridElementHeightInv),
    mnTrackReferenceForFrame(0), mnFuseTargetForKF(0), mnBALocalForKF(0), mnBAFixedForKF(0),
    mnLoopQuery(0), mnLoopWords(0), mnRelocQuery(0), mnRelocWords(0), mnBAGlobalForKF(0),
    fx(F.fx), fy(F.fy), cx(F.cx), cy(F.cy), invfx(F.invfx), invfy(F.invfy),
    mbf(F.mbf), mb(F.mb), mThDepth(F.mThDepth), N(F.N), mvKeys(F.mvKeys), mvKeysUn(F.mvKeysUn),
    mvuRight(F.mvuRight), mvDepth(F.mvDepth), mDescriptors(F.mDescriptors.clone()),
    mBowVec(F.mBowVec), mFeatVec(F.mFeatVec), mnScaleLevels(F.mnScaleLevels), mfScaleFactor(F.mfScaleFactor),
    mfLogScaleFactor(F.mfLogScaleFactor), mvScaleFactors(F.mvScaleFactors), mvLevelSigma2(F.mvLevelSigma2),
    mvInvLevelSigma2(F.mvInvLevelSigma2), mnMinX(F.mnMinX), mnMinY(F.mnMinY), mnMaxX(F.mnMaxX),
    mnMaxY(F.mnMaxY), mK(F.mK), mvpMapPoints(F.mvpMapPoints), mpKeyFrameDB(pKFDB),
    mpORBvocabulary(F.mpORBvocabulary), mbFirstConnection(true), mpParent(NULL), mbNotErase(false),
    mbToBeErased(false), mbBad(false), mHalfBaseline(F.mb/2), mpMap(pMap)
{
    mnId=nNextId++;

    mGrid.resize(mnGridCols);
    for(int i=0; i<mnGridCols;i++)
    {
        mGrid[i].resize(mnGridRows);
        for(int j=0; j<mnGridRows; j++)
            mGrid[i][j] = F.mGrid[i][j];
    }

    SetPose(F.mTcw);    
}

void KeyFrame::ComputeBoW()
{
    // BoW vector 혹은 Feature vector가 비어있다면,
    if(mBowVec.empty() || mFeatVec.empty()) // mBowVec.empty() = true or mFeatVec.empty() = true
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors); // 2차원 matrix -> 1차원 vector
        // keyframe의 mDescriptors -> keyframe이 가지고 있는 모든 keypoint의 descriptor
        // Feature vector associate features with nodes in the 4th level (from leaves up)
        // We assume the vocabulary tree has 6 levels, change the 4 otherwise
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4); // 현재 keyframe의 모든 keypoint의 descriptor -> BoW vector, Feature vector
    }
}

void KeyFrame::SetPose(const cv::Mat &Tcw_)
{
    unique_lock<mutex> lock(mMutexPose);
    Tcw_.copyTo(Tcw);
    cv::Mat Rcw = Tcw.rowRange(0,3).colRange(0,3);
    cv::Mat tcw = Tcw.rowRange(0,3).col(3);
    cv::Mat Rwc = Rcw.t();
    Ow = -Rwc*tcw; // camera to world coordinate의 translation

    Twc = cv::Mat::eye(4,4,Tcw.type());
    Rwc.copyTo(Twc.rowRange(0,3).colRange(0,3));
    Ow.copyTo(Twc.rowRange(0,3).col(3));
    cv::Mat center = (cv::Mat_<float>(4,1) << mHalfBaseline, 0 , 0, 1);
    Cw = Twc*center;
}

cv::Mat KeyFrame::GetPose()
{
    unique_lock<mutex> lock(mMutexPose); // unique_lock 객체인 lock은 mutex 객체인 mMutexPose를 소유한다.
    return Tcw.clone(); // clone() : 깊은 복사, 새로운 메모리 주소를 할당 받아 값을 복사한다.
    // 현재 keyframe의 world to camera coordinate
}

cv::Mat KeyFrame::GetPoseInverse()
{
    unique_lock<mutex> lock(mMutexPose); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexPose를 소유한다.
    return Twc.clone(); // camera to world coordinate의 pose
    // clone() : 깊은 복사 -> 새로운 메모리 주소를 할당 받아 값을 복사한다.
}

cv::Mat KeyFrame::GetCameraCenter()
{
    unique_lock<mutex> lock(mMutexPose); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexPose를 소유한다.
    return Ow.clone(); // world 좌표계 상에서의 camera의 위치, Twc의 translation
}

cv::Mat KeyFrame::GetStereoCenter()
{
    unique_lock<mutex> lock(mMutexPose);
    return Cw.clone();
}


cv::Mat KeyFrame::GetRotation()
{
    unique_lock<mutex> lock(mMutexPose); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexPos를 소유한다.
    return Tcw.rowRange(0,3).colRange(0,3).clone(); // world to camera coordinate의 rotation
}

cv::Mat KeyFrame::GetTranslation()
{
    unique_lock<mutex> lock(mMutexPose); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexPose를 소유한다.
    return Tcw.rowRange(0,3).col(3).clone(); // world to camera coordinate의 translation
}

// (mit->first)->AddConnection(this,mit->second) : this -> 현재 keyframe(tracking thread -> local mapping thread), 
// mit->second -> weight : 현재 keyframe의 map points가 특정 keyframe에서 몇 개 관측되는가.
void KeyFrame::AddConnection(KeyFrame *pKF, const int &weight)
{
    // Critical Section
    {
        unique_lock<mutex> lock(mMutexConnections); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexConnections를 소유한다.
        if(!mConnectedKeyFrameWeights.count(pKF)) // mConnectedKeyFrameWeights에 현재 frame이 없다면,
            mConnectedKeyFrameWeights[pKF]=weight; // 현재 keyframe에 대한 weight 값을 함수의 input으로 들어온 weight 값으로 치환한다.
        else if(mConnectedKeyFrameWeights[pKF]!=weight) // 현재 keyframe에 대한 weight 값이 함수의 input으로 들어온 weight 값과 다르다면,
            mConnectedKeyFrameWeights[pKF]=weight; // 현재 keyframe에 대한 weight 값을 함수의 input으로 들어온 weight 값으로 치환한다.
        else
            return; // 함수 밖으로 빠져나가라.
    }

    UpdateBestCovisibles(); // 특정 keyframe의 mConnectedKeyFrameWeights를 weight 값의 오름차순으로 정렬
}

void KeyFrame::UpdateBestCovisibles()
{
    unique_lock<mutex> lock(mMutexConnections); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexConnections를 소유한다.
    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(mConnectedKeyFrameWeights.size());
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
       vPairs.push_back(make_pair(mit->second,mit->first)); // mit->second : weight, mit->first : pKF

    sort(vPairs.begin(),vPairs.end()); // mConnectedKeyFrameWeights를 weight 값의 오름차순으로 정렬한다.
    list<KeyFrame*> lKFs; // 특정 keyframe의 mConnectedKeyFrameWeights를 weight 값의 오름차순으로 정렬한 keyframe list
    list<int> lWs; // 특정 keyframe의 mConnectedKeyFrameWeights를 weight 값의 오름차순으로 정렬한 weight list
    for(size_t i=0, iend=vPairs.size(); i<iend;i++)
    {
        lKFs.push_front(vPairs[i].second); // weight
        lWs.push_front(vPairs[i].first); // pKF
    }

    mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end()); // 특정 keyframe의 covisibility graph 상의 neighbors를 weight 값으로 정렬한 keyframe vector
    mvOrderedWeights = vector<int>(lWs.begin(), lWs.end()); // 특정 keyframe의 covisibility graph 상의 neighbors를 weight 값으로 정렬한 weight vector
}

set<KeyFrame*> KeyFrame::GetConnectedKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    set<KeyFrame*> s;
    for(map<KeyFrame*,int>::iterator mit=mConnectedKeyFrameWeights.begin();mit!=mConnectedKeyFrameWeights.end();mit++)
        s.insert(mit->first);
    return s;
}

vector<KeyFrame*> KeyFrame::GetVectorCovisibleKeyFrames()
{
    unique_lock<mutex> lock(mMutexConnections);
    return mvpOrderedConnectedKeyFrames;
}

// 해당 keyframe의 covisibility graph 상에서의 neighbors를 찾는 함수
vector<KeyFrame*> KeyFrame::GetBestCovisibilityKeyFrames(const int &N)
{
    unique_lock<mutex> lock(mMutexConnections); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexConnections를 소유한다.
    if((int)mvpOrderedConnectedKeyFrames.size()<N) // mvpOrderedConnectedKeyFrames -> covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 ordering
        return mvpOrderedConnectedKeyFrames;
    else // covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 ordering한 list의 size >= N
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(),mvpOrderedConnectedKeyFrames.begin()+N);
        // covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 N개 추출
}

vector<KeyFrame*> KeyFrame::GetCovisiblesByWeight(const int &w)
{
    unique_lock<mutex> lock(mMutexConnections);

    if(mvpOrderedConnectedKeyFrames.empty())
        return vector<KeyFrame*>();

    vector<int>::iterator it = upper_bound(mvOrderedWeights.begin(),mvOrderedWeights.end(),w,KeyFrame::weightComp);
    if(it==mvOrderedWeights.end())
        return vector<KeyFrame*>();
    else
    {
        int n = it-mvOrderedWeights.begin();
        return vector<KeyFrame*>(mvpOrderedConnectedKeyFrames.begin(), mvpOrderedConnectedKeyFrames.begin()+n);
    }
}

int KeyFrame::GetWeight(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexConnections);
    if(mConnectedKeyFrameWeights.count(pKF))
        return mConnectedKeyFrameWeights[pKF];
    else
        return 0;
}

void KeyFrame::AddMapPoint(MapPoint *pMP, const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    mvpMapPoints[idx]=pMP; // 해당 map point를 mvpMapPoints vector에 귀속시킨다.
}

void KeyFrame::EraseMapPointMatch(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    mvpMapPoints[idx]=static_cast<MapPoint*>(NULL); // 해당 keyframe 상의 idx번째 keypoint와 association 관계를 갖는 map point를 Null 값으로 초기화한다.
}

void KeyFrame::EraseMapPointMatch(MapPoint* pMP)
{
    int idx = pMP->GetIndexInKeyFrame(this);
    if(idx>=0)
        mvpMapPoints[idx]=static_cast<MapPoint*>(NULL);
}


void KeyFrame::ReplaceMapPointMatch(const size_t &idx, MapPoint* pMP)
{
    mvpMapPoints[idx]=pMP;
}

set<MapPoint*> KeyFrame::GetMapPoints()
{
    unique_lock<mutex> lock(mMutexFeatures);
    set<MapPoint*> s;
    for(size_t i=0, iend=mvpMapPoints.size(); i<iend; i++)
    {
        if(!mvpMapPoints[i])
            continue;
        MapPoint* pMP = mvpMapPoints[i];
        if(!pMP->isBad())
            s.insert(pMP);
    }
    return s;
}

// input : minObs, output : nPoints -> minObs 이상의 keyframe에서 관측되는 현재 keyframe의 map points
int KeyFrame::TrackedMapPoints(const int &minObs)
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.

    int nPoints=0;
    const bool bCheckObs = minObs>0; // minObs > 0 -> bCheckObs = true, minObs <= 0 -> bCheckObs = false
    for(int i=0; i<N; i++) // keyframe의 keypoint 개수만큼 반복
    {
        MapPoint* pMP = mvpMapPoints[i]; // keyframe의 i번째 keypoint와 association 관계가 있는 map point
        if(pMP) // pMP가 존재한다면,
        // pMP가 Null 값이라면 false를 return 할 것이다.
        {
            if(!pMP->isBad()) // pMP->isBad() = false, 해당 map point가 나쁘지 않다고 판단되면,
            {
                if(bCheckObs) // bCheckObs = true -> minObs > 0
                {
                    if(mvpMapPoints[i]->Observations()>=minObs) // 해당 keyframe의 map points가 최소한의 개수의 keyframe에서 발견된다면,
                        nPoints++; // nPoints : minObs 이상의 keyframe에서 관측되는 현재 keyframe의 map points
                }
                else // bCheckObs = false -> minObs <= 0
                    nPoints++; // mvpMapPoints[i]->Observations()>=minObs 조건을 항상 만족하기 때문
            }
        }
    }
    
    return nPoints;
}

vector<MapPoint*> KeyFrame::GetMapPointMatches()
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    return mvpMapPoints; // keyframe 상의 keypoint와 association의 관계를 가지는 map point
}

MapPoint* KeyFrame::GetMapPoint(const size_t &idx)
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    return mvpMapPoints[idx]; // keyframe 상의 idx번째 keypoint와 association의 관계를 가지는 map point
}

void KeyFrame::UpdateConnections()
{
    map<KeyFrame*,int> KFcounter;
    // map은 각 노드가 key와 value 쌍으로 이루어진 트리이다. 특히, 중복을 허용하지 않는다.
    // first, second가 있는 pair 객체로 저장되는데 first-key, second-value로 저장된다.

    vector<MapPoint*> vpMP;
    
    // Critical Section
    {
        unique_lock<mutex> lockMPs(mMutexFeatures); // unique_lock class의 객체인 lockMPs는 mutex 객체인 mMutexFeatures를 소유한다.
        vpMP = mvpMapPoints; // 현재 keyframe의 keypoint에 해당하는 map points -> vpMP vector, Q. 얕은 복사?
        // Q. local mapping thread로 넘어오기 전, tracking thread에서 depth 정보를 이용하여 만든 unmatched keypoints에 대한 map points 포함?
    }

    //For all map points in keyframe check in which other keyframes are they seen
    //Increase counter for those keyframes
    for(vector<MapPoint*>::iterator vit=vpMP.begin(), vend=vpMP.end(); vit!=vend; vit++)
    {
        MapPoint* pMP = *vit; // de-reference -> map point

        if(!pMP) // 해당 map point가 존재하지 않는다면,
            continue; // 해당 루프의 끝으로 이동한다.

        if(pMP->isBad()) // 해당 map point가 나쁘다고 판단되면,
            continue; // 해당 루프의 끝으로 이동한다.

        map<KeyFrame*,size_t> observations = pMP->GetObservations(); // 해당 map point가 어떠한 keyframe의 몇 번째 keypoint에서 관측되는가에 대한 정보

        for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
        {
            if(mit->first->mnId==mnId) // mit->first : keyframe, 현재 keyframe에 대한 observation 정보는 제외한다.
                continue; // 해당 루프의 끝으로 이동한다.
            KFcounter[mit->first]++; // KFcounter[keyframe]++
        }
    }

    // This should not happen
    if(KFcounter.empty()) // KFcounter.empty() = true
        return; // 해당 함수를 빠져나가라.

    //If the counter is greater than threshold add connection
    //In case no keyframe counter is over threshold add the one with maximum counter
    int nmax=0;
    KeyFrame* pKFmax=NULL;
    int th = 15;

    vector<pair<int,KeyFrame*> > vPairs;
    vPairs.reserve(KFcounter.size());
    for(map<KeyFrame*,int>::iterator mit=KFcounter.begin(), mend=KFcounter.end(); mit!=mend; mit++)
    {
        if(mit->second>nmax) // mit->second : 현재 keyframe의 map point들이 특정 keyframe(현재 keyframe은 제외)에서 몇 번 관측 되었는가.
        {
            nmax=mit->second; // 현재 keyframe의 map point들이 특정 keyframe에서 가장 많이 관측된 횟수
            pKFmax=mit->first; // 현재 keyframe의 map point들이 가장 많이 관측된 keyframe
        }
        if(mit->second>=th) // 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수 >= 15
        {
            vPairs.push_back(make_pair(mit->second,mit->first)); // (현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수(>= 15) = weight, 특정 keyframe) -> vPairs
            (mit->first)->AddConnection(this,mit->second); // 현재 keyframe과, 현재 keyframe의 15개 이상의 map point들이 관측된 특정 keyframe과의 edge update
            // mit->first keyframe을 기준으로 update
        }
    }

    if(vPairs.empty()) // 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수가 15 이상인 경우가 없을 때
    {
        vPairs.push_back(make_pair(nmax,pKFmax)); // (현재 keyframe의 map point들이 특정 keyframe에서 가장 많이 관측된 횟수 = weight, 현재 keyframe의 map point들이 가장 많이 관측된 keyframe)
        pKFmax->AddConnection(this,nmax); // 현재 keyframe과, 현재 keyframe의 nmax개의 map point들이 관측된 pKFmax keyframe과의 edge update
        // pKFmax keyframe을 기준으로 update
    }

    sort(vPairs.begin(),vPairs.end()); // weight을 기준으로 오름차순 정렬
    list<KeyFrame*> lKFs; // 현재 keyframe의 15개 이상의 map point들이 관측된 keyframe list
    list<int> lWs; // 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수(>= 15) = weight list
    for(size_t i=0; i<vPairs.size();i++)
    {
        lKFs.push_front(vPairs[i].second); // keyframe
        lWs.push_front(vPairs[i].first); // weight
    }

    // Critical Section
    {
        unique_lock<mutex> lockCon(mMutexConnections); // unique_lock class의 객체인 lockCon은 mutex 객체인 mMutexConnections를 소유한다.

        // mspConnectedKeyFrames = spConnectedKeyFrames;
        mConnectedKeyFrameWeights = KFcounter; // KFcounter -> 현재 keyframe에서 관측되는 map point들이 어떠한 keyframe(현재 keyframe 제외)에서 몇 번 관측되는가.
        // KFcounter->first : 현재 keyframe의 map point들이 관측된 keyframe, KFcounter->second : 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수
        mvpOrderedConnectedKeyFrames = vector<KeyFrame*>(lKFs.begin(),lKFs.end()); // 현재 keyframe의 15개 이상의 map point들이 관측된 keyframe list -> vector
        mvOrderedWeights = vector<int>(lWs.begin(), lWs.end()); // 현재 keyframe의 map point들이 특정 keyframe에서 관측된 횟수(>= 15) = weight list -> vector
        // 현재 keyframe을 기준으로 update -> this->AddConnection

        if(mbFirstConnection && mnId!=0) // mbFirstConnection = true and mnId != 0
        {
            mpParent = mvpOrderedConnectedKeyFrames.front(); // 현재 keyframe의 15개 이상의 map point들이 관측된 keyframe 중 가장 큰 weight을 가진 keyframe -> parent keyframe
            // front() : 벡터의 첫 번째 요소를 반환한다.
            mpParent->AddChild(this); // 현재 keyframe을 parent keyframe의 child keyframe으로 삽입한다.
            mbFirstConnection = false;
        }

    }
}

void KeyFrame::AddChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections); // unique_lock class의 객체인 lockCon은 mutex 객체인 mMutexConnections를 소유한다.
    mspChildrens.insert(pKF);
}

void KeyFrame::EraseChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mspChildrens.erase(pKF);
}

void KeyFrame::ChangeParent(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mpParent = pKF;
    pKF->AddChild(this);
}

set<KeyFrame*> KeyFrame::GetChilds()
{
    unique_lock<mutex> lockCon(mMutexConnections); // unique_lock class의 객체인 lockCon은 mutex 객체인 mMutexConnections를 소유한다.
    return mspChildrens;
}

KeyFrame* KeyFrame::GetParent()
{
    unique_lock<mutex> lockCon(mMutexConnections); // unique_lock class의 객체인 lockCon은 mutex 객체인 mMutexConnections를 소유한다.
    return mpParent;
}

bool KeyFrame::hasChild(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspChildrens.count(pKF);
}

void KeyFrame::AddLoopEdge(KeyFrame *pKF)
{
    unique_lock<mutex> lockCon(mMutexConnections);
    mbNotErase = true;
    mspLoopEdges.insert(pKF);
}

set<KeyFrame*> KeyFrame::GetLoopEdges()
{
    unique_lock<mutex> lockCon(mMutexConnections);
    return mspLoopEdges;
}

void KeyFrame::SetNotErase()
{
    unique_lock<mutex> lock(mMutexConnections);
    mbNotErase = true;
}

void KeyFrame::SetErase()
{
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mspLoopEdges.empty())
        {
            mbNotErase = false;
        }
    }

    if(mbToBeErased)
    {
        SetBadFlag();
    }
}

void KeyFrame::SetBadFlag()
{   
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mnId==0)
            return;
        else if(mbNotErase)
        {
            mbToBeErased = true;
            return;
        }
    }

    for(map<KeyFrame*,int>::iterator mit = mConnectedKeyFrameWeights.begin(), mend=mConnectedKeyFrameWeights.end(); mit!=mend; mit++)
        mit->first->EraseConnection(this);

    for(size_t i=0; i<mvpMapPoints.size(); i++)
        if(mvpMapPoints[i])
            mvpMapPoints[i]->EraseObservation(this);
    {
        unique_lock<mutex> lock(mMutexConnections);
        unique_lock<mutex> lock1(mMutexFeatures);

        mConnectedKeyFrameWeights.clear();
        mvpOrderedConnectedKeyFrames.clear();

        // Update Spanning Tree
        set<KeyFrame*> sParentCandidates;
        sParentCandidates.insert(mpParent);

        // Assign at each iteration one children with a parent (the pair with highest covisibility weight)
        // Include that children as new parent candidate for the rest
        while(!mspChildrens.empty())
        {
            bool bContinue = false;

            int max = -1;
            KeyFrame* pC;
            KeyFrame* pP;

            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(), send=mspChildrens.end(); sit!=send; sit++)
            {
                KeyFrame* pKF = *sit;
                if(pKF->isBad())
                    continue;

                // Check if a parent candidate is connected to the keyframe
                vector<KeyFrame*> vpConnected = pKF->GetVectorCovisibleKeyFrames();
                for(size_t i=0, iend=vpConnected.size(); i<iend; i++)
                {
                    for(set<KeyFrame*>::iterator spcit=sParentCandidates.begin(), spcend=sParentCandidates.end(); spcit!=spcend; spcit++)
                    {
                        if(vpConnected[i]->mnId == (*spcit)->mnId)
                        {
                            int w = pKF->GetWeight(vpConnected[i]);
                            if(w>max)
                            {
                                pC = pKF;
                                pP = vpConnected[i];
                                max = w;
                                bContinue = true;
                            }
                        }
                    }
                }
            }

            if(bContinue)
            {
                pC->ChangeParent(pP);
                sParentCandidates.insert(pC);
                mspChildrens.erase(pC);
            }
            else
                break;
        }

        // If a children has no covisibility links with any parent candidate, assign to the original parent of this KF
        if(!mspChildrens.empty())
            for(set<KeyFrame*>::iterator sit=mspChildrens.begin(); sit!=mspChildrens.end(); sit++)
            {
                (*sit)->ChangeParent(mpParent);
            }

        mpParent->EraseChild(this);
        mTcp = Tcw*mpParent->GetPoseInverse();
        mbBad = true;
    }


    mpMap->EraseKeyFrame(this);
    mpKeyFrameDB->erase(this);
}

bool KeyFrame::isBad()
{
    unique_lock<mutex> lock(mMutexConnections); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexConnections를 소유한다.
    return mbBad; // mbBad flag를 return 
}

void KeyFrame::EraseConnection(KeyFrame* pKF)
{
    bool bUpdate = false;
    {
        unique_lock<mutex> lock(mMutexConnections);
        if(mConnectedKeyFrameWeights.count(pKF))
        {
            mConnectedKeyFrameWeights.erase(pKF);
            bUpdate=true;
        }
    }

    if(bUpdate)
        UpdateBestCovisibles();
}

vector<size_t> KeyFrame::GetFeaturesInArea(const float &x, const float &y, const float &r) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=mnGridCols)
        return vIndices;

    const int nMaxCellX = min((int)mnGridCols-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=mnGridRows)
        return vIndices;

    const int nMaxCellY = min((int)mnGridRows-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool KeyFrame::IsInImage(const float &x, const float &y) const
{
    return (x>=mnMinX && x<mnMaxX && y>=mnMinY && y<mnMaxY);
}

cv::Mat KeyFrame::UnprojectStereo(int i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeys[i].pt.x;
        const float v = mvKeys[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);

        unique_lock<mutex> lock(mMutexPose);
        return Twc.rowRange(0,3).colRange(0,3)*x3Dc+Twc.rowRange(0,3).col(3);
    }
    else
        return cv::Mat();
}

float KeyFrame::ComputeSceneMedianDepth(const int q)
{
    vector<MapPoint*> vpMapPoints;
    cv::Mat Tcw_;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPose);
        vpMapPoints = mvpMapPoints;
        Tcw_ = Tcw.clone();
    }

    vector<float> vDepths;
    vDepths.reserve(N);
    cv::Mat Rcw2 = Tcw_.row(2).colRange(0,3);
    Rcw2 = Rcw2.t();
    float zcw = Tcw_.at<float>(2,3);
    for(int i=0; i<N; i++)
    {
        if(mvpMapPoints[i])
        {
            MapPoint* pMP = mvpMapPoints[i];
            cv::Mat x3Dw = pMP->GetWorldPos();
            float z = Rcw2.dot(x3Dw)+zcw;
            vDepths.push_back(z);
        }
    }

    sort(vDepths.begin(),vDepths.end());

    return vDepths[(vDepths.size()-1)/q];
}

} //namespace ORB_SLAM