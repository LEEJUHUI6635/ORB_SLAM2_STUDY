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

#include "Map.h"

#include<mutex>

namespace ORB_SLAM2
{

Map::Map():mnMaxKFid(0),mnBigChangeIdx(0)
{
}

// 현재의 keyframe을 mspKeyFrames에 삽입 한다.
void Map::AddKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    // 현재 keyframe을 mspKeyframes set에 삽입한다.
    mspKeyFrames.insert(pKF); // Map::mspKeyFrames -> set
    // set container : 노드 기반 컨테이너이며 균형 이진트리로 구현되어 있다. key라 불리는 원소들의 집합으로 이루어져 있다.
    // 원소가 insert() 함수에 의해 삽입이 되면, 원소는 자동으로 정렬된다.
    if(pKF->mnId>mnMaxKFid) // 현재 keyframe의 ID(KeyFrame) > 가장 큰 keyframe의 ID(Map)
        mnMaxKFid=pKF->mnId; // 가장 큰 keyframe의 ID = 현재 keyframe의 ID
}

void Map::AddMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    mspMapPoints.insert(pMP); // pMP -> Map::mspMapPoints
}

// 전체 map에서 해당 map point를 삭제
void Map::EraseMapPoint(MapPoint *pMP)
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    mspMapPoints.erase(pMP); // mspMapPoints -> map에 포함되는 map points
    // erase() : 벡터 v에서 i번째 원소를 삭제, erase 함수의 인자는 지우고 싶은 원소의 주소이다.

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::EraseKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    mspKeyFrames.erase(pKF); // 전체 map에 포함되는 모든 keyframe set에서 해당 keyframe을 삭제한다.

    // TODO: This only erase the pointer.
    // Delete the MapPoint
}

void Map::SetReferenceMapPoints(const vector<MapPoint *> &vpMPs)
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    mvpReferenceMapPoints = vpMPs; // 현재 local map points -> reference map points vector
}

void Map::InformNewBigChange()
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    mnBigChangeIdx++;
}

int Map::GetLastBigChangeIdx()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnBigChangeIdx;
}

vector<KeyFrame*> Map::GetAllKeyFrames()
{
    unique_lock<mutex> lock(mMutexMap);
    return vector<KeyFrame*>(mspKeyFrames.begin(),mspKeyFrames.end());
}

vector<MapPoint*> Map::GetAllMapPoints()
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    return vector<MapPoint*>(mspMapPoints.begin(),mspMapPoints.end()); // Q. overloading -> mspMapPoints의 처음부터 끝까지 vector 형태로 받아온다.
}

// map에 저장되어 있는 map point를 출력하는 함수 -> Q. 전체 map? local map? A. 전체 map
long unsigned int Map::MapPointsInMap()
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    return mspMapPoints.size(); // map에 저장되어 있는 map point를 출력
}

long unsigned int Map::KeyFramesInMap()
{
    unique_lock<mutex> lock(mMutexMap); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexMap을 소유한다.
    return mspKeyFrames.size(); // map에 포함되는 keyframe의 개수
}

vector<MapPoint*> Map::GetReferenceMapPoints()
{
    unique_lock<mutex> lock(mMutexMap);
    return mvpReferenceMapPoints;
}

long unsigned int Map::GetMaxKFid()
{
    unique_lock<mutex> lock(mMutexMap);
    return mnMaxKFid;
}

void Map::clear()
{
    // set container : 노드 기반 컨테이너이며 균형 이진트리로 구현되어 있다. key라 불리는 원소들의 집합으로 이루어진 컨테이너이다. key 값은 중복이 허용되지 않는다.
    // 원소가 insert 함수에 의하여 삽입이 되면, 원소는 자동으로 정렬(오름차순 정렬)된다. 
    // 기본 선언 방법은 set<[Data Type]> [변수 이름]
    // set<int>::iterator -> 반복자 선언
    // s.begin() : 맨 첫 번째 원소를 가리키는 반복자를 리턴한다. s.end() : 맨 마지막 원소를 가리키는 원소의 끝 부분을 알 때 사용한다. 
    // map point를 삭제한다.
    for(set<MapPoint*>::iterator sit=mspMapPoints.begin(), send=mspMapPoints.end(); sit!=send; sit++)
        delete *sit; // sit은 pointer이기 때문에 sit을 de-reference 하여 값을 삭제한다.

    // keyframe을 삭제한다.
    for(set<KeyFrame*>::iterator sit=mspKeyFrames.begin(), send=mspKeyFrames.end(); sit!=send; sit++)
        delete *sit;

    mspMapPoints.clear(); // mspMapPoints는 pointer이기 때문에, 해당하는 메모리 삭제
    mspKeyFrames.clear(); // mspKeyFrames는 pointer이기 때문에, 해당하는 메모리 삭제
    mnMaxKFid = 0; // Q.
    mvpReferenceMapPoints.clear(); // mvpReferenceMapPoints는 pointer이기 때문에, 해당하는 메모리 삭제
    mvpKeyFrameOrigins.clear(); // mvpKeyFrameOrigins는 pointer이기 때문에, 해당하는 메모리 삭제
}

} //namespace ORB_SLAM