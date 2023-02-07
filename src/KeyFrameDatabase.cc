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

#include "KeyFrameDatabase.h"

#include "KeyFrame.h"
#include "Thirdparty/DBoW2/DBoW2/BowVector.h"

#include<mutex>

using namespace std;

namespace ORB_SLAM2
{

KeyFrameDatabase::KeyFrameDatabase (const ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}


void KeyFrameDatabase::add(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutex); // unique_lock class의 객체인 lock은 mutex 객체인 mMutex를 소유한다.

    // vit->first : word id, vit->second : word value
    for(DBoW2::BowVector::const_iterator vit= pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(pKF); // mvInvertedFile[word id] = pKF
}

void KeyFrameDatabase::erase(KeyFrame* pKF)
{
    unique_lock<mutex> lock(mMutex); // unique_lock class의 객체인 lock은 mutex 객체인 mMutex를 소유한다.

    // Erase elements in the Inverse File for the entry
    // mBowVec->first : word id, mBowVec->second : word value
    for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // mvInvertedFile[word id] -> 해당 word에 속하는 keyframe list

        for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++) // 해당 word에 속하는 keyframe 개수만큼 반복
        {
            if(pKF==*lit) // 해당 keyframe이 해당 word에 속하는 keyframe이라면,
            {
                lKFs.erase(lit); // 해당 word에 속하는 keyframe list에서 해당 keyframe을 삭제한다.
                break; // 해당 루프를 종료한다.
            }
        }
    }
}

void KeyFrameDatabase::clear()
{
    // inverse index -> 모든 keyframe에 대해서, 각 word를 가지고 있는 image id를 저장하고 있는 database
    mvInvertedFile.clear(); // clear() : 리스트의 모든 요소를 제거한다.
    mvInvertedFile.resize(mpVoc->size()); // vocabulary(word) 크기만큼 resize
}

// mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore)
// 현재 keyframe과 map point가 아닌, word를 공유하는 keyframe 중 candidates search
vector<KeyFrame*> KeyFrameDatabase::DetectLoopCandidates(KeyFrame* pKF, float minScore)
{
    set<KeyFrame*> spConnectedKeyFrames = pKF->GetConnectedKeyFrames(); // 현재 keyframe의 map point들이 관측되는 keyframe들 = 현재 keyframe과 map point를 공유하는 keyframe set
    list<KeyFrame*> lKFsSharingWords;

    // Search all keyframes that share a word with current keyframes
    // Discard keyframes connected to the query keyframe
    // Critical Section
    {
        unique_lock<mutex> lock(mMutex); // unique_lock class의 객체인 lock은 mutex 객체인 mMutex를 소유한다.
        
        // vit->first : word id, vit->second : word value
        for(DBoW2::BowVector::const_iterator vit=pKF->mBowVec.begin(), vend=pKF->mBowVec.end(); vit != vend; vit++) // 현재 keyframe의 모든 word의 개수만큼 반복
        {
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // mvInvertedFile[word id] -> 특정 word id에 속한 keyframe list

            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++) // 특정 word id에 속한 keyframe 개수만큼 반복
            {
                KeyFrame* pKFi=*lit; // de-reference -> 특정 word id에 속한 keyframe
                if(pKFi->mnLoopQuery!=pKF->mnId) // 특정 word id에 속한 keyframe의 mnLoopQuery가 현재 keyframe의 id가 아니라면, -> 중복 방지
                {
                    pKFi->mnLoopWords=0;
                    if(!spConnectedKeyFrames.count(pKFi)) // 현재 keyframe과 map point를 공유하는 keyframe set에, 현재 keyframe과 word를 공유하는 keyframe이 없다면,
                    {
                        pKFi->mnLoopQuery=pKF->mnId; // 중복 방지
                        lKFsSharingWords.push_back(pKFi); // 현재 keyframe과 map point를 공유하지는 않지만, word를 공유하는 keyframe만을 포함
                    }
                }
                pKFi->mnLoopWords++; // 특정 keyframe이 가지고 있는 word의 개수
            }
        }
    }

    if(lKFsSharingWords.empty()) // lKFsSharingWords.empty() = true -> spConnectedKeyFrames.count(pKFi) = true -> 현재 keyframe과 map point를 공유하는 keyframe set에, 현재 keyframe과 word를 공유하는 keyframe이 있다면,
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lScoreAndMatch;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    // lKFsSharingWords -> 현재 keyframe과 map point를 공유하지는 않지만, word를 공유하는 keyframe만을 포함
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        if((*lit)->mnLoopWords>maxCommonWords) // 현재 keyframe과 map point를 공유하지는 않지만, word를 공유하는 keyframe이 가지고 있는 word의 개수
            maxCommonWords=(*lit)->mnLoopWords;
    }

    int minCommonWords = maxCommonWords*0.8f; // threshold

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit; // de-reference -> 현재 keyframe과 map point를 공유하지는 않지만, word를 공유하는 keyframe이 가지고 있는 word의 개수

        if(pKFi->mnLoopWords>minCommonWords) // 현재 keyframe과 공유하는 최소한의 word를 가지고 있는 모든 keyframe에 대하여
        {
            nscores++;

            float si = mpVoc->score(pKF->mBowVec,pKFi->mBowVec); // 현재 keyframe의 BoW vector - 현재 keyframe과 word를 공유하는 keyframe의 BoW vector

            pKFi->mLoopScore = si;
            if(si>=minScore) // minScore -> 현재 keyframe의 neighbor keyframes 중, 현재 keyframe과 가장 유사하지 않은 것의 score, threshold
                lScoreAndMatch.push_back(make_pair(si,pKFi)); // 현재 keyframe과 word를 공유하는 keyframe이 얼마나 유사한가에 대한 score, 현재 keyframe과 word를 공유하는 keyframe
        }
    }

    if(lScoreAndMatch.empty()) // lScoreAndMatch.empty() = true
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = minScore; // 현재 keyframe의 neighbor keyframes 중, 현재 keyframe과 가장 유사하지 않은 것의 score, threshold

    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        KeyFrame* pKFi = it->second; // 현재 keyframe과 word를 공유하는 keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10); // 현재 keyframe과 word를 공유하는 keyframe의 10개 이하의 neighbor keyframes

        float bestScore = it->first;
        float accScore = it->first;
        KeyFrame* pBestKF = pKFi;
        // 현재 keyframe과 word를 공유하는 keyframe 뿐만 아니라, 해당 keyframe의 neighbor keyframe까지 candidates로 고려한다.
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++) // 현재 keyframe과 word를 공유하는 keyframe의 neighbor keyframe의 개수만큼 반복
        {
            KeyFrame* pKF2 = *vit; // de-reference -> 현재 keyframe과 word를 공유하는 keyframe의 neighbor keyframe
            // pKF2->mnLoopQuery == pKF->mnId -> 현재 keyframe과 map point를 공유하지는 않지만, word를 공유하는 keyframe에 한해서,
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords) // pKF2->mnLoopWords > minCommonWords -> 현재 keyframe과 공유하는 최소한의 word를 가지고 있는 keyframe에 한해서,
            {
                accScore+=pKF2->mLoopScore; // 현재 keyframe과 word를 공유하는 keyframe에 대하여, 모든 neighbor keyframe의 accumulated score
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        // lAccScoreAndMatch -> 중복 가능
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF)); // 현재 keyframe과 word를 공유하는 keyframe에 대하여, 모든 neighbor keyframe의 accumulated score, 현재 keyframe과 가장 많은 word를 공유하는 keyframe
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore; // threshold

    set<KeyFrame*> spAlreadyAddedKF;
    vector<KeyFrame*> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size()); // 현재 keyframe과 word를 공유하는 keyframe의 개수만큼 메모리 할당

    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        // it->first : 현재 keyframe과 word를 공유하는 keyframe에 대하여, 모든 neighbor keyframe의 accumulated score, it->second : 현재 keyframe과 가장 많은 word를 공유하는 keyframe
        if(it->first>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second; // 현재 keyframe과 가장 많은 word를 공유하는 keyframe
            if(!spAlreadyAddedKF.count(pKFi)) // 중복 방지 -> spAlreadyAddedKF에 현재 keyframe과 가장 많은 word를 공유하는 keyframe이 존재한다면,
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }


    return vpLoopCandidates;
}

// DetectRelocalizationCandidates(&mCurrentFrame)
// 현재 frame과 word를 공유하는 모든 keyframe을 찾는 함수 -> BoW의 첫 번째 기능(BoW vector 이용)
vector<KeyFrame*> KeyFrameDatabase::DetectRelocalizationCandidates(Frame *F)
{
    list<KeyFrame*> lKFsSharingWords; // 해당 frame과 word를 공유하는 모든 keyframes

    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex); // unique_lock class의 객체인 lock은 mutex 객체인 mMutex를 소유한다.

        // DBoW2::BowVector -> std::map<WordId, WordValue>, word value -> 한 이미지 내에서 word의 빈도수
        for(DBoW2::BowVector::const_iterator vit=F->mBowVec.begin(), vend=F->mBowVec.end(); vit != vend; vit++)
        {
            // mnInvertedFile = inverse index -> 특정한 word를 가지고 있는 모든 keyframe
            list<KeyFrame*> &lKFs =   mvInvertedFile[vit->first]; // 특정한 word id에 해당하는 keyframes -> lKFs
            // 1개의 word id -> 여러 개의 keyframe
            for(list<KeyFrame*>::iterator lit=lKFs.begin(), lend= lKFs.end(); lit!=lend; lit++) // 특정한 word id에 속하는 keyframe의 개수만큼 반복
            {
                KeyFrame* pKFi=*lit; // de-reference -> 특정한 word id에 해당하는 keyframe
                // 특정 frame이 처음으로 BoW에 접근하였다면,
                if(pKFi->mnRelocQuery!=F->mnId) // 특정한 word id에 해당하는 keyframe의 mnRelocQuery가 frame의 id와 일치하지 않는다면,
                {
                    pKFi->mnRelocWords=0; // 특정한 word id에 해당하는 keyframe의 mnRelocWords를 초기화
                    pKFi->mnRelocQuery=F->mnId; // frame의 id -> 특정한 word id에 해당하는 keyframe의 mnRelocQuery
                    lKFsSharingWords.push_back(pKFi);
                    // 중복 방지 -> 여러 개의 word id에는 여러 개의 keyframe들이 중복되어 있을 것이다.
                }
                pKFi->mnRelocWords++; // mnRelocWords -> 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수
                // 여러 word id에 대하여 반복문을 수행하면, 이전에 나왔던 keyframe이 반복하여 나오는데 이를 반영하기 위함
                // 해당 keyframe이 현재 frame과 중복되는 relocalization words의 개수
            }
        }
    }
    if(lKFsSharingWords.empty())
        return vector<KeyFrame*>();

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0; // 현재 frame과 가장 많이 공유하고 있는 keyframe의 words 개수
    // lKFsSharingWords list -> 현재 frame과 word를 공유하고 있는 모든 keyframe
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++) // 현재 frame과 word를 공유하고 있는 모든 keyframe의 개수만큼 반복
    {
        // *lit : de-reference -> keyframe
        if((*lit)->mnRelocWords>maxCommonWords) // keyframe->mnRelocWords : 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수
            maxCommonWords=(*lit)->mnRelocWords; // keyframe->mnRelocWords -> maxCommonWords : 각 keyframe과 현재 frame이 공유하는 가장 많은 relocalization words 개수
    }

    int minCommonWords = maxCommonWords*0.8f; // threshold

    // pair class는 사용자가 지정한 2개 타입의 데이터를 저장하는데 사용한다.
    list<pair<float,KeyFrame*> > lScoreAndMatch;

    int nscores=0;

    // Compute similarity score.
    // lKFsSharingWords list -> 특정한 word를 공유하는 keyframe
    for(list<KeyFrame*>::iterator lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        KeyFrame* pKFi = *lit; // de-reference -> 특정한 word를 공유하는 keyframe

        if(pKFi->mnRelocWords>minCommonWords) // 해당 keyframe과 현재 frame이 공유하는 relocalization words의 개수가 일정 threshold(maxCommonWords*0.8)를 넘으면,
        {
            nscores++;
            float si = mpVoc->score(F->mBowVec,pKFi->mBowVec); // 해당 keyframe과 현재 frame의 similarity score 계산
            pKFi->mRelocScore=si;
            lScoreAndMatch.push_back(make_pair(si,pKFi)); // 하한선 이상의 keyframe + 해당 keyframe의 similarity score
        }
    }

    if(lScoreAndMatch.empty())
        return vector<KeyFrame*>();

    list<pair<float,KeyFrame*> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Q. covisibility graph 상의 neighbors이면서 relocalization candidate keyframe과의 관계를 고려
    // Lets now accumulate score by covisibility
    for(list<pair<float,KeyFrame*> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++) // keyframe + 해당 keyframe의 similarity score의 개수만큼 반복
    {
        KeyFrame* pKFi = it->second; // it->second : keyframe
        vector<KeyFrame*> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10); // 해당 keyframe의 covisibility graph 상에서의 10개의 neighbors(weight가 높은 순서대로) 추출

        // 초기화
        float bestScore = it->first; // it->first : 해당 keyframe의 similarity score
        float accScore = bestScore;
        KeyFrame* pBestKF = pKFi;
        // 해당 keyframe의 covisibility graph 상에서 N개의 neighbors만큼 반복
        for(vector<KeyFrame*>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKF2 = *vit; // de-reference -> neighbors keyframe
            if(pKF2->mnRelocQuery!=F->mnId) // Q. 해당 keyframe의 neighbors 중에 relocalization candidate keyframe이 아니라면,
                continue; // 해당 for문의 끝으로 이동한다.

            // Q. pKF2->mnRelocWords <= minCommonWords?
            accScore+=pKF2->mRelocScore; // neighbors keyframe의 mRelocScore(위의 코드에서 수행)까지 고려
            // 현재 frame과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe을 찾는다.
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2; // 현재 frame과 words를 공유하는 keyframe의 neighbors 중 가장 similarity score가 높은 keyframe
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF)); // 현재 keyframe(relocalization candidates)과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe
        // Q. 같은 keyframe이 중복될 수 있다.
        if(accScore>bestAccScore)
            bestAccScore=accScore; // 모든 keyframe(relocalization candidates)를 포함한 그의 neighbors 중 가장 많은 word를 공유하는 keyframe의 similarity score
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore; // 하한선
    set<KeyFrame*> spAlreadyAddedKF;
    // set container : set container는 노드 기반 컨테이너이며 균형 이진트리로 구현되어 있다. key라 불리는 원소들의 집합으로 이루어진 container이다. 
    // key 값은 중복이 허용되지 않는다. 원소가 insert 멤버 함수에 의해 삽입이 되면, 원소는 자동으로 정렬된다.
    vector<KeyFrame*> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    // 현재 frame과 word를 공유하는 keyframe의 neighbors 중, 가장 많은 word를 공유하는 keyframe + 해당 keyframe의 similarity score
    for(list<pair<float,KeyFrame*> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first; // it->first : accumulated similarity score(해당 keyframe의 neighbors까지 고려)
        if(si>minScoreToRetain)
        {
            KeyFrame* pKFi = it->second; // it->second : neighbors keyframe 중, 가장 많은 word를 공유하는 keyframe
            if(!spAlreadyAddedKF.count(pKFi)) // 해당 keyframe이 spAlreadyAddedKF에 존재하지 않는다면 -> 중복 허용 x
            {
                vpRelocCandidates.push_back(pKFi); // pKFi(neighbors keyframe) -> vpRelocCandidates
                spAlreadyAddedKF.insert(pKFi); // pKFi(neighbors) -> spAlreadyAddedKF
            }
        }
    }

    return vpRelocCandidates;
}

} //namespace ORB_SLAM