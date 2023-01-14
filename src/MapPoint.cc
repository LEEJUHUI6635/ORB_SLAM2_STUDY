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

#include "MapPoint.h"
#include "ORBmatcher.h"

#include<mutex>

namespace ORB_SLAM2
{

long unsigned int MapPoint::nNextId=0;
mutex MapPoint::mGlobalMutex;

MapPoint::MapPoint(const cv::Mat &Pos, KeyFrame *pRefKF, Map* pMap):
    mnFirstKFid(pRefKF->mnId), mnFirstFrame(pRefKF->mnFrameId), nObs(0), mnTrackReferenceForFrame(0),
    mnLastFrameSeen(0), mnBALocalForKF(0), mnFuseCandidateForKF(0), mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(pRefKF), mnVisible(1), mnFound(1), mbBad(false),
    mpReplaced(static_cast<MapPoint*>(NULL)), mfMinDistance(0), mfMaxDistance(0), mpMap(pMap)
{
    Pos.copyTo(mWorldPos); // Pos -> 절대 좌표계인 world 좌표계 상에서의 map point의 position
    mNormalVector = cv::Mat::zeros(3,1,CV_32F);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

MapPoint::MapPoint(const cv::Mat &Pos, Map* pMap, Frame* pFrame, const int &idxF):
    mnFirstKFid(-1), mnFirstFrame(pFrame->mnId), nObs(0), mnTrackReferenceForFrame(0), mnLastFrameSeen(0),
    mnBALocalForKF(0), mnFuseCandidateForKF(0),mnLoopPointForKF(0), mnCorrectedByKF(0),
    mnCorrectedReference(0), mnBAGlobalForKF(0), mpRefKF(static_cast<KeyFrame*>(NULL)), mnVisible(1),
    mnFound(1), mbBad(false), mpReplaced(NULL), mpMap(pMap)
{
    Pos.copyTo(mWorldPos);
    cv::Mat Ow = pFrame->GetCameraCenter();
    mNormalVector = mWorldPos - Ow;
    mNormalVector = mNormalVector/cv::norm(mNormalVector);

    cv::Mat PC = Pos - Ow;
    const float dist = cv::norm(PC);
    const int level = pFrame->mvKeysUn[idxF].octave;
    const float levelScaleFactor =  pFrame->mvScaleFactors[level];
    const int nLevels = pFrame->mnScaleLevels;

    mfMaxDistance = dist*levelScaleFactor;
    mfMinDistance = mfMaxDistance/pFrame->mvScaleFactors[nLevels-1];

    pFrame->mDescriptors.row(idxF).copyTo(mDescriptor);

    // MapPoints can be created from Tracking and Local Mapping. This mutex avoid conflicts with id.
    unique_lock<mutex> lock(mpMap->mMutexPointCreation);
    mnId=nNextId++;
}

void MapPoint::SetWorldPos(const cv::Mat &Pos)
{
    unique_lock<mutex> lock2(mGlobalMutex);
    unique_lock<mutex> lock(mMutexPos);
    Pos.copyTo(mWorldPos);
}

cv::Mat MapPoint::GetWorldPos()
{
    unique_lock<mutex> lock(mMutexPos); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexPos를 소유한다.
    return mWorldPos.clone(); // 절대 좌표계인 world 좌표계 상의 map point의 position
}

cv::Mat MapPoint::GetNormal()
{
    unique_lock<mutex> lock(mMutexPos);
    return mNormalVector.clone();
}

KeyFrame* MapPoint::GetReferenceKeyFrame()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mpRefKF;
}

// Data Association -> 새로운 map point가 어느 keyframe에서 발견된 몇 번째 keypoint인지 저장한다.
void MapPoint::AddObservation(KeyFrame* pKF, size_t idx)
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    if(mObservations.count(pKF)) // count() : vector나 set과 같은 container에서 특정 원소가 몇 개 존재하는지 혹은 특정 조건을 만족하는 원소가 몇 개 포함되어 있는지를 구할 수 있다.
    // 해당 원소가 존재하지 않을 때에는 0을 반환한다.
    // 현재 keyframe이 존재한다면 return
        return;
    mObservations[pKF]=idx; // 새로운 map point가 어느 keyframe에서 발견된 몇 번째 keypoint인지 저장한다.
    // MapPoint::mObservations의 key : keyframe, value : keypoint의 idx

    // nObs : 해당 frame에서 keypoint와 association 관계가 있는 map point의 개수
    if(pKF->mvuRight[idx]>=0) // stereo의 경우, mvuRight[idx] >= 0
        nObs+=2; // left image에서 추출한 keypoint + right image에서 추출한 keypoint
    else // monocular의 경우, mvuRight[idx] < 0 
        nObs++; // left image에서 추출한 keypoint
}

void MapPoint::EraseObservation(KeyFrame* pKF)
{
    bool bBad=false;
    {
        unique_lock<mutex> lock(mMutexFeatures);
        if(mObservations.count(pKF))
        {
            int idx = mObservations[pKF];
            if(pKF->mvuRight[idx]>=0)
                nObs-=2;
            else
                nObs--;

            mObservations.erase(pKF);

            if(mpRefKF==pKF)
                mpRefKF=mObservations.begin()->first;

            // If only 2 observations or less, discard point
            if(nObs<=2)
                bBad=true;
        }
    }

    if(bBad)
        SetBadFlag();
}

map<KeyFrame*, size_t> MapPoint::GetObservations()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mObservations;
}

int MapPoint::Observations()
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    return nObs;
}

void MapPoint::SetBadFlag()
{
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        mbBad=true;
        obs = mObservations;
        mObservations.clear();
    }
    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        KeyFrame* pKF = mit->first;
        pKF->EraseMapPointMatch(mit->second);
    }

    mpMap->EraseMapPoint(this);
}

MapPoint* MapPoint::GetReplaced()
{
    unique_lock<mutex> lock1(mMutexFeatures); // unique_lock class의 객체인 lock1은 mutex 객체인 mMutexFeatures를 소유한다.
    unique_lock<mutex> lock2(mMutexPos); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexPos를 소유한다.
    return mpReplaced;
}

void MapPoint::Replace(MapPoint* pMP)
{
    if(pMP->mnId==this->mnId)
        return;

    int nvisible, nfound;
    map<KeyFrame*,size_t> obs;
    {
        unique_lock<mutex> lock1(mMutexFeatures);
        unique_lock<mutex> lock2(mMutexPos);
        obs=mObservations;
        mObservations.clear();
        mbBad=true;
        nvisible = mnVisible;
        nfound = mnFound;
        mpReplaced = pMP;
    }

    for(map<KeyFrame*,size_t>::iterator mit=obs.begin(), mend=obs.end(); mit!=mend; mit++)
    {
        // Replace measurement in keyframe
        KeyFrame* pKF = mit->first;

        if(!pMP->IsInKeyFrame(pKF))
        {
            pKF->ReplaceMapPointMatch(mit->second, pMP);
            pMP->AddObservation(pKF,mit->second);
        }
        else
        {
            pKF->EraseMapPointMatch(mit->second);
        }
    }
    pMP->IncreaseFound(nfound);
    pMP->IncreaseVisible(nvisible);
    pMP->ComputeDistinctiveDescriptors();

    mpMap->EraseMapPoint(this);
}

bool MapPoint::isBad()
{
    unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
    unique_lock<mutex> lock2(mMutexPos); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexPos를 소유한다.
    return mbBad; // map point culling 관련 flag
}

void MapPoint::IncreaseVisible(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnVisible+=n;
}

void MapPoint::IncreaseFound(int n)
{
    unique_lock<mutex> lock(mMutexFeatures);
    mnFound+=n;
}

float MapPoint::GetFoundRatio()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return static_cast<float>(mnFound)/mnVisible;
}

// 해당 map point가 representative descriptor(hamming distance가 가장 작은 descriptor)를 저장한다.
void MapPoint::ComputeDistinctiveDescriptors()
{
    // Retrieve all observed descriptors
    vector<cv::Mat> vDescriptors; // descriptor를 저장할 vector 생성

    // map class는 템플릿 인자로 2가지를 받는다. 첫 번째 인자는 key라 부르고, 두 번째 인자는 value라고 부른다. 
    // map class는 key가 있다면 vector처럼 랜덤 엑세스가 가능하며, 실행 중에 동적으로 크기가 확장된다. 
    map<KeyFrame*,size_t> observations;

    // Critical Section
    {
        unique_lock<mutex> lock1(mMutexFeatures); // unique_lock class인 lock1은 mutex 객체인 mMutexFeatures를 소유한다.
        if(mbBad) // mbBad = true -> map point 판별
            return;
        observations=mObservations; // mObservations : 해당 map point가 어떠한 keyframe의 몇 번째 keypoint에 해당하는지에 대한 정보를 담고 있다.
    }

    if(observations.empty()) // 해당 map point가 어떠한 keyframe의 몇 번째 keypoint에 해당하는지에 대한 정보를 담고 있지 않다면, 
        return;

    vDescriptors.reserve(observations.size()); // vDescriptors vector는 observations의 size만큼의 capacity를 확보해 놓는다.
    // reserve() : vector는 push_back()을 통해 배열의 원소를 계속 늘릴 수 있다. 그러나, vector가 처음 선언될 때 예약되어 있던 capacity를 초과하면, 그보다 더 큰 용량의 메모리를 
    // 할당한 후, 기존의 원소를 모두 복사하고 기존의 메모리는 해제하는 작업을 거치는데, 이 과정이 많이 일어나면 성능이 매우 떨어지게 된다. 
    // 따라서, 위의 문제를 해결하기 위해서 reserve(), resize() 함수를 통해 capacity를 미리 확보해 놓을 수 있는데, 두 함수의 차이는 용량 확보 후 그 공간을 초기화하느냐에 대한 여부이다.

    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        // observations의 key : keyframe, value : keypoint의 idx
        KeyFrame* pKF = mit->first; // iter->first : key, iter->second : value
        // KeyFrame class의 포인터 객체인 pKF에 각 keyframe의 index를 넣어준다.

        if(!pKF->isBad()) // mbBad = false -> keyframe을 판별했을 때, 나쁘지 않다면
            vDescriptors.push_back(pKF->mDescriptors.row(mit->second)); // mit->second : keypoint의 idx, pKF->mDescriptors의 keypoint의 idx의 한 행을 vDescriptors vector에 넣는다.
            // vDescriptors -> 하나의 map point에 해당하는 여러 keyframe 상의 keypoint의 descriptor, mDescriptors -> 하나의 keyframe이 가지는 모든 keypoint의 descriptor
    }

    if(vDescriptors.empty()) // vDescriptors가 비어 있다면,
        return;

    // Compute distances between them
    const size_t N = vDescriptors.size(); // vDescriptors의 size
    // size_t는 어떤 타입의 사이즈든지 나타낼 수 있는 충분한 bytes를 가진 unsigned integer

    float Distances[N][N]; // 하나의 keyframe 내, N개의 descriptor 간의 거리를 포함하고 있는 정적 할당 배열

    for(size_t i=0;i<N;i++)
    {
        Distances[i][i]=0; // 자기 자신과의 거리는 0
        for(size_t j=i+1;j<N;j++)
        {
            int distij = ORBmatcher::DescriptorDistance(vDescriptors[i],vDescriptors[j]); // 하나의 keyframe의 descriptor 간의 거리
            Distances[i][j]=distij;
            Distances[j][i]=distij;
        }
    }

    // Take the descriptor with least median distance to the rest
    int BestMedian = INT_MAX; // int의 최대값
    int BestIdx = 0;
    for(size_t i=0;i<N;i++)
    {   // Distances -> NxN matrix, Distances[i] -> i 번째 vector(i 번째 descriptor와 다른 descriptor 간의 distance)
        vector<int> vDists(Distances[i],Distances[i]+N); // i 번째 vector 그 자체를 나타낸다.
        // Distances -> 2차원 배열, Distances[i] -> 1차원 배열, Distances[i] -> 배열의 첫 번째 원소가 가지는 주소
        sort(vDists.begin(),vDists.end()); 
        int median = vDists[0.5*(N-1)]; // vDists의 median 값 -> 보다 신뢰도 있는 정보 이용(salt and pepper noise 제거)

        if(median<BestMedian) // N개의 descriptor 중 각각의 median 값을 구하고, median 값이 가장 작은 descriptor를 best descriptor로 추출한다.
        {
            BestMedian = median;
            BestIdx = i;
        }
    }

    // Critical Section
    {
        unique_lock<mutex> lock(mMutexFeatures); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexFeatures를 소유한다.
        mDescriptor = vDescriptors[BestIdx].clone(); // best index를 가지는 descriptor를 추출한다. -> MapPoint::mDescriptor
    }
}

cv::Mat MapPoint::GetDescriptor()
{
    unique_lock<mutex> lock(mMutexFeatures);
    return mDescriptor.clone();
}

int MapPoint::GetIndexInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    if(mObservations.count(pKF))
        return mObservations[pKF];
    else
        return -1;
}

bool MapPoint::IsInKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexFeatures);
    return (mObservations.count(pKF));
}

void MapPoint::UpdateNormalAndDepth()
{
    map<KeyFrame*,size_t> observations; // map은 각 노드가 key와 value 쌍으로 이루어진 트리이다. 따라서 map은 first, second가 있는 pair 객체로 저장되는데,
    // first-key, second-value로 저장되며, 중복은 허용되지 않는다.
    // map 기본 형태 : map<key, value> map1;
    // observations -> 하나의 map point가 발견될 수 있는 모든 keyframe과 keypoint와의 association 정보를 담고 있다.
    KeyFrame* pRefKF;
    cv::Mat Pos;
    // Critical Section
    {
        unique_lock<mutex> lock1(mMutexFeatures); // unique_lock class의 객체인 lock1은 mutex 객체인 mMutexFeatures를 소유한다.
        unique_lock<mutex> lock2(mMutexPos); // unique_lock class의 객체인 lock2는 mutex 객체인 mMutexPos를 소유한다.
        if(mbBad) // map point를 지울 것인지에 대한 flag
            return;
        // map point가 담고 있는 정보
        observations=mObservations; // 해당 map point가 어떠한 keyframe에서 발견되고, 해당 keyframe에서 몇 번째 keypoint인지에 대한 정보를 담고 있다.
        pRefKF=mpRefKF; // Reference Keyframe
        Pos = mWorldPos.clone(); // 절대 좌표계인 world 좌표계 상에서 map point의 position
    }

    if(observations.empty())
        return;
    
    // normal vector
    cv::Mat normal = cv::Mat::zeros(3,1,CV_32F); // [3, 1] size의 CV_32F type의 0으로 이루어진 matrix, CV_32F -> 4 byte의 float형
    int n=0;
    for(map<KeyFrame*,size_t>::iterator mit=observations.begin(), mend=observations.end(); mit!=mend; mit++)
    {
        // first : key -> keyframe의 index(어떠한 keyframe인지), second : value -> 1개의 keyframe 내의 keypoint의 index
        // keyframe은 계속 생성된다. -> 해당 map point가 관측되는 keyframe은 여러 개일 것이기 때문에
        KeyFrame* pKF = mit->first; // 해당 map point가 관측되는 keyframe -> pKF
        cv::Mat Owi = pKF->GetCameraCenter(); // world 좌표계 상에서의 camera의 위치, Twc의 translation, 해당하는 keyframe의 위치 in world coordinate
        cv::Mat normali = mWorldPos - Owi; // map point와 해당 map point가 관측되는 keyframe의 optical center
        normal = normal + normali/cv::norm(normali); // [3, 1], map point와 해당하는 map point가 관측되는 모든 keyframe의 optical center와의 벡터의 합
        n++; // n = observations의 개수
    }

    cv::Mat PC = Pos - pRefKF->GetCameraCenter(); // map point와 reference keyframe의 optical center와의 벡터
    const float dist = cv::norm(PC);
    // reference keyframe에 대한 scale parameter 계산
    const int level = pRefKF->mvKeysUn[observations[pRefKF]].octave; // observations[pRefKF] -> 해당 map point가 reference keyframe의 몇 번째 keypoint인지
    // cv::KeyPoint::octave() -> 특징점이 추출된 옥타브(피라미드 단계)를 나타낸다.
    const float levelScaleFactor =  pRefKF->mvScaleFactors[level]; // 특징점이 추출된 ocatave에 해당하는 scale factor
    const int nLevels = pRefKF->mnScaleLevels; // 전체 level의 개수 -> nLevels = 8

    // Critical Section
    {
        // map point는 max distance와 min distance, mean normal vector를 가지고 있다.
        unique_lock<mutex> lock3(mMutexPos); // unique_lock class의 객체인 lock3는 mutex 객체인 mMutexPos를 소유한다.
        mfMaxDistance = dist*levelScaleFactor; // (map point - reference keyframe) * 특징점이 추출된 octave에 해당하는 scale factor
        mfMinDistance = mfMaxDistance/pRefKF->mvScaleFactors[nLevels-1];
        mNormalVector = normal/n; // normal -> 하나의 map point와 해당하는 map point가 관측되는 모든 keyframe과의 벡터의 합, mNormalVector -> mean normal vector
    }
}

float MapPoint::GetMinDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 0.8f*mfMinDistance;
}

float MapPoint::GetMaxDistanceInvariance()
{
    unique_lock<mutex> lock(mMutexPos);
    return 1.2f*mfMaxDistance;
}

int MapPoint::PredictScale(const float &currentDist, KeyFrame* pKF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pKF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pKF->mnScaleLevels)
        nScale = pKF->mnScaleLevels-1;

    return nScale;
}

int MapPoint::PredictScale(const float &currentDist, Frame* pF)
{
    float ratio;
    {
        unique_lock<mutex> lock(mMutexPos);
        ratio = mfMaxDistance/currentDist;
    }

    int nScale = ceil(log(ratio)/pF->mfLogScaleFactor);
    if(nScale<0)
        nScale = 0;
    else if(nScale>=pF->mnScaleLevels)
        nScale = pF->mnScaleLevels-1;

    return nScale;
}



} //namespace ORB_SLAM
