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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames()) // Local mapping thread에서 새로운 keyframe이 들어오면 return true, 새로운 keyframe이 들어오지 않으면 return false
        // CheckNewKeyFrames() = true -> Local mapping thread에서 새로운 keyframe이 들어오면,
        {
            // Detect loop candidates and check covisibility consistency
            // 현재 keyframe과 현재 keyframe의 neighbor keyframe들의 BoW vector의 similarity score를 구하여, 가장 낮은 score를 threshold로 설정한다. 
            // 현재 keyframe과 word를 공유하는 keyframe들 중 해당 threshold를 넘는 keyframe을 loop detection candidate keyframe으로 설정한다.
            // 후보 keyframe들 중, 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수가 3 이상이라면, 
            // 최종 loop detection candidate keyframe으로 설정한다.
            if(DetectLoop()) // DetectLoop() = true
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               // BoW의 두 번째 기능(Data Association)으로 현재 keyframe과 loop detection candidate keyframe과의 correspondence를 구하고, Sim3Solver로 두 keyframe의 relative similarity transformation을 계산한다.
               // 충분한 inlier를 가지고 similarity transformation을 최적화하였다면, 계산한 relative pose를 loop detection keyframe에 곱하여 현재 keyframe의 world to camera coordinate을 구한다. 
               // 확실한 검증을 위하여, loop detection keyframe과 neighbor keyframes의 map point를 현재 keyframe에 projection 하였을 때, 충분한 inlier를 가진다면 loop detection keyframe으로 확정한다.
               if(ComputeSim3()) // ComputeSim3() = true
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexLoopQueue를 소유한다.
    if(pKF->mnId!=0) // keyframe이 가장 처음의 keyframe이 아니라면,
        mlpLoopKeyFrameQueue.push_back(pKF); // 현재 keyframe -> mlpLoopKeyFrameQueue
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexLoopQueue를 소유한다.
    return(!mlpLoopKeyFrameQueue.empty()); // mlpLoopKeyFrameQueue.empty() = true -> return false, mlpLoopKeyFrameQueue.empty() = false -> return true
    // Local mapping thread에서 새로운 keyframe이 들어오면 return true, 새로운 keyframe이 들어오지 않으면 return false
}

// 현재 keyframe과 현재 keyframe의 neighbor keyframe들의 BoW vector의 similarity score를 구하여, 가장 낮은 score를 threshold로 설정한다. 
// 현재 keyframe과 word를 공유하는 keyframe들 중 해당 threshold를 넘는 keyframe을 loop detection candidate keyframe으로 설정한다.
// 후보 keyframe들 중, 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수가 3 이상이라면, 최종 loop detection candidate keyframe으로 설정한다.
bool LoopClosing::DetectLoop()
{
    // Critical Section
    {
        unique_lock<mutex> lock(mMutexLoopQueue); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexLoopQueue를 소유한다.
        mpCurrentKF = mlpLoopKeyFrameQueue.front(); // Local mapping thread에서 Loop closing thread로 넘겨준 keyframe들 중, 가장 처음 들어온 keyframe
        mlpLoopKeyFrameQueue.pop_front();
        // pop_front() : 리스트 제일 앞의 원소 삭제
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase(); // mbNotErase = true
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10) // 현재 keyframe이 최근의 loop detection을 수행한 keyframe에서 10 frame도 지나지 않았다면, -> Q. loop가 아닐 것이라고 판단하기 때문
    {
        // 현재 keyframe으로 BoW vocabulary를 계산하고, 현재 keyframe을 삭제한다.
        mpKeyFrameDB->add(mpCurrentKF); // mvInvertedFile[vit->first].push_back(mpCurrentKF)
        mpCurrentKF->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        return false; // DetectLoop() = false
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    // 현재 keyframe과 map point를 공유하는 keyframes 중 가장 유사하지 않은 keyframe과의 score -> minScore
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 정렬
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec; // 현재 keyframe의 BoW vector
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++) // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe의 개수만큼 반복
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i]; // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 i번째 keyframe
        if(pKF->isBad()) // 현재 keyframe의 neighbor keyframe이 나쁘다고 판단하면,
            continue; // 해당 루프의 끝으로 이동한다.
        const DBoW2::BowVector &BowVec = pKF->mBowVec; // 현재 keyframe의 neighbor keyframe의 BoW vector = histogram

        // 현재 keyframe의 BoW vector와 covisibility graph 상에서 연결된 keyframe들의 Bow vector와의 similarity score를 계산하여, 가장 유사하지 않은 similarity score를 가져와, 하한선으로 설정한다.
        float score = mpORBVocabulary->score(CurrentBowVec, BowVec); // 현재 keyframe의 BoW vector(histogram)과 neighbor keyframe의 BoW vector(histogram)과의 score를 계산한다.

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    // 현재 keyframe과 map point가 아닌, word를 공유하는 keyframe 중 candidates search
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty()) // vpCandidateKFs.empty() = true
    {
        mpKeyFrameDB->add(mpCurrentKF); // mvInvertedFile[word id] = mpCurrentKF -> 현재 keyframe의 모든 word에 대하여, inverted file을 갱신한다.
        mvConsistentGroups.clear(); // mvConsistentGroups 삭제
        mpCurrentKF->SetErase(); // Keyframe Culling : 현재 keyframe과 관련있는 모든 것에서 현재 keyframe에 대한 정보 삭제 + 현재 keyframe의 children keyframe의 parent keyframe을 할당
        return false; // DetectLoop() = false
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    // mvpEnoughConsistentCandidates -> 최종 candidate keyframes
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    // candidate keyframe + candidate keyframe의 covisibility graph 상에서 neighbor keyframes, candidate keyframe 시점에서 covisibility graph 상에서 연이은 candidate keyframe의 개수
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false); // mvConsistentGroups의 size만큼 false로 초기화
    // mvConsistentGroups : 지난 loop detection에서의 candidate keyframe에 대한 neighbor keyframes, 지난 candidate keyframe 시점에서 covisibility graph 상에서 연이은 candidate keyframe의 개수
    // mvConsistentGroups->first : 현재 keyframe과 word를 공유하는 i번째 candidate keyframe + 현재 keyframe과 word를 공유하는 i번째 candidate keyframe의 map point들이 관측되는 keyframe
    // mvConsistentGroups->second : 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++) // 현재 keyframe과 word를 공유하는 candidate keyframe의 개수만큼 반복
    {
        // 이번 loop detection 시점에서의 candidate keyframe에 대한 고려

        KeyFrame* pCandidateKF = vpCandidateKFs[i]; // 현재 keyframe과 word를 공유하는 i번째 candidate keyframe

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames(); // 현재 keyframe과 word를 공유하는 i번째 candidate keyframe의 map point들이 관측되는 keyframe들
        spCandidateGroup.insert(pCandidateKF); // spCandidateGroup = 현재 keyframe과 word를 공유하는 i번째 candidate keyframe + 현재 keyframe과 word를 공유하는 i번째 candidate keyframe의 map point들이 관측되는 keyframe들

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++) // 지난 loop detection에서의 여러 candidate keyframes과 이에 대한 neighbor keyframes
        {
            // 과거의 loop detection 시점에서의 candidate keyframe에 대한 고려 -> 이번 loop detection 시점에서의 candidate keyframe의 연속성을 찾기 위해

            // Q. iG -> 과거 loop detection에서의 iG개의 시점
            // mvConsistentGroups.size() -> 과거 loop detection에서의 시점의 개수
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first; // 지난 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes

            bool bConsistent = false;
            // 이번 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                // 지난 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes set에서, 이번 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes 중 하나 이상이 겹친다면,
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break; // 해당 루프를 종료한다.
                }
            }

            if(bConsistent) // bConsistent = true <- 지난 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes set에서, 이번 loop detection에서의 하나의 candidate keyframe과 이에 대한 neighbor keyframes 중 하나 이상이 겹친다면,
            {
                // mvConsistentGroups[iG].second : iG번째 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                // 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수
                // = iG번째 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수 + 1
                if(!vbConsistentGroup[iG]) // vbConsistentGroup[iG] = false
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    // spCandidateGroup -> 현재 keyframe과 word를 공유하는 i번째 candidate keyframe + 현재 keyframe과 word를 공유하는 i번째 candidate keyframe의 map point들이 관측되는 keyframe
                    // nCurrentConsistency -> 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                    // 중복 방지
                }
                // mnCovisibilityConsistencyTh = 3
                // nCurrentConsistency >= mnCovisibilityConsistencyTh -> 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수 >= 3
                // bEnoughConsistent = false
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    // consistency 조건을 만족하는 최종 candidate keyframe
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF); // 현재 keyframe과 word를 공유하는 i번째 candidate keyframe -> mvpEnoughConsistentCandidates
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup) // bConsistentForSomeGroup = false
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0); // nCurrentConsistency = 0
            // -> 현재 keyframe 시점에서 구한 candidate keyframe이 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결된 횟수 = 0
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    // vCurrentConsistentGroups -> 현재 시점, mvConsistentGroups -> 과거 시점
    mvConsistentGroups = vCurrentConsistentGroups; // mvConsistentGroups->first : spCandidateGroup, mvConsistentGroups->second : nCurrentConsistency
    // spCandidateGroup -> 현재 keyframe과 word를 공유하는 i번째 candidate keyframe + 현재 keyframe과 word를 공유하는 i번째 candidate keyframe의 map point들이 관측되는 keyframe
    // nCurrentConsistency -> 현재 keyframe 시점에서 구한 candidate keyframe은 지난 keyframe 시점에서 구한 candidate keyframe과 covisibility graph 상에서 연결되어 있어야 한다.


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF); // mvInvertedFile[word id] = mpCurrentKF

    if(mvpEnoughConsistentCandidates.empty()) // mvpEnoughConsistentCandidates.empty() = true
    {
        mpCurrentKF->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        return false;
    }
    else // mvpEnoughConsistentCandidates.empty() = false
    {
        return true;
    }

    // Q. 
    mpCurrentKF->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
    return false;
}

// BoW의 두 번째 기능(Data Association)으로 현재 keyframe과 loop detection candidate keyframe과의 correspondence를 구하고, Sim3Solver로 두 keyframe의 relative similarity transformation을 계산한다.
// 충분한 inlier를 가지고 similarity transformation을 최적화하였다면, 계산한 relative pose를 loop detection keyframe에 곱하여 현재 keyframe의 world to camera coordinate을 구한다. 
// 확실한 검증을 위하여, loop detection keyframe과 neighbor keyframes의 map point를 현재 keyframe에 projection 하였을 때, 충분한 inlier를 가진다면 loop detection keyframe으로 확정한다.
bool LoopClosing::ComputeSim3()
{
    // For each consistent loop candidate we try to compute a Sim3

    const int nInitialCandidates = mvpEnoughConsistentCandidates.size(); // loop detection candidate keyframe의 개수

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates); // 각 loop detection candidate keyframe에 따른 vpSim3Solvers

    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

    int nCandidates=0; //candidates with enough matches

    for(int i=0; i<nInitialCandidates; i++) // loop detection candidate keyframe의 개수만큼 반복
    {
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i]; // i번째 loop detection candidate keyframe

        // avoid that local mapping erase it while it is being processed in this thread
        pKF->SetNotErase(); // mbNotErase = true

        if(pKF->isBad()) // i번째 loop detection candidate keyframe이 나쁘다고 판단하면,
        {
            vbDiscarded[i] = true; // i번째 loop detection candidate keyframe을 고려하지 않는다.
            continue; // 해당 루프의 끝으로 이동한다.
        }
        
        // 현재 keyframe의 idx1번째 map point - i번째 loop detection candidate keyframe의 bestIdx2번째 map point correspondences
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

        if(nmatches<20) // 현재 keyframe과 i번째 loop detection candidate keyframe의 correspondence의 개수가 20 미만이라면,
        {
            vbDiscarded[i] = true; // i번째 loop detection candidate keyframe은 고려 x
            continue; // 해당 루프의 끝으로 이동한다.
        }
        else // nmatches >= 20 -> 현재 keyframe과 i번째 loop detection candidate keyframe의 correspondecne의 개수가 20 이상이라면,
        {
            // i번째 loop detection candidate keyframe에 대한 Sim3Solver 설정
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale); // mbFixScale -> Fix scale in the stereo/RGB-D case
            pSolver->SetRansacParameters(0.99,20,300);
            vpSim3Solvers[i] = pSolver;
        }

        nCandidates++;
    }

    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
    while(nCandidates>0 && !bMatch) // nCandidates > 0 and bMatch = false
    {
        for(int i=0; i<nInitialCandidates; i++) // loop detection candidate keyframe의 개수만큼 반복
        {
            if(vbDiscarded[i]) // vbDiscarded[i] = true
                continue; // 해당 루프의 끝으로 이동한다.

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i]; // i번째 loop detection candidate keyframe

            // Perform 5 Ransac Iterations
            vector<bool> vbInliers;
            int nInliers;
            bool bNoMore;

            Sim3Solver* pSolver = vpSim3Solvers[i];
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers); // loop detection candidate keyframe to current keyframe coordinate의 similarity transformation

            // If Ransac reachs max. iterations discard keyframe
            if(bNoMore) // bNoMore = true
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
            if(!Scm.empty()) // Scm.empty() = false
            {
                // 현재 keyframe과 i번째 loop detection candidate keyframe의 correspondence의 개수만큼 NULL값으로 초기화
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++) // 현재 keyframe의 keypoint 개수만큼 반복
                {
                    if(vbInliers[j]) // vbInliers[j] = true
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j]; // i번째 loop detection candidate keyframe의 j번째 map point correspondence
                }

                cv::Mat R = pSolver->GetEstimatedRotation(); // loop detection candidate keyframe to current keyframe coordinate의 rotation
                cv::Mat t = pSolver->GetEstimatedTranslation(); // loop detection candidate keyframe to current keyframe coordinate의 translation
                const float s = pSolver->GetEstimatedScale(); // loop detection candidate keyframe to current keyframe coordinate의 scale
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);
                // Sim3Solver를 통해 projection하여, 현재 keyframe의 map point와 loop detection candidate keyframe의 map point와의 더 많은 correspondence를 새롭게 찾는다.

                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale); // 보류
                // Q. pSolver->iterate() : Similarity transformation을 계산, OptimizeSim3() : 이미 계산한 Similarity transformation을 최적화

                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
                    bMatch = true; // while문 종료
                    mpMatchedKF = pKF; // Sim3 optimization을 통해 도출한 inlier가 20개 이상이라면, 해당 loop detection candidate keyframe을 loop detection keyframe으로 선정한다.
                    // Smw이 Scw보다 조금 더 정확한 pose일 것이다. 따라서, Sim3Solver를 통해 구한 relative pose를 통해 Scw를 보정한다.
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
                    // pKF->GetRotation() : loop detection keyframe의 world to camera coordinate의 rotation, pKF->GetTranslation() : loop detection keyframe의 world to camera coordinate의 translation
                    mg2oScw = gScm*gSmw; // loop detection keyframe to current keyframe coordinate의 transformation x loop detection keyframe의 world to camera coordinate의 transformation
                    // = world to current keyframe coordinate의 transformation
                    mScw = Converter::toCvMat(mg2oScw); // scale을 고려한 world to current keyframe coordinate의 transformation 계산

                    mvpCurrentMatchedPoints = vpMapPointMatches;
                    break; // 해당 루프를 종료한다.
                }
            }
        }
    }

    if(!bMatch) // bMatch = false -> Sim3 optimization을 통해 도출한 inlier가 20개 이상이 나오지 않는 경우
    {
        // 현재 keyframe과 해당하는 loop detection candidate keyframe들을 모두 삭제한다.
        for(int i=0; i<nInitialCandidates; i++) // loop detection candidate keyframe의 개수만큼 반복
             mvpEnoughConsistentCandidates[i]->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        mpCurrentKF->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        return false; // ComputeSim3() = false
    }

    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames(); // loop detection keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 정렬
    vpLoopConnectedKFs.push_back(mpMatchedKF); // vpLoopConnectedKFs = loop detection keyframe + loop detection keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들
    mvpLoopMapPoints.clear();
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit; // de-reference -> loop detection keyframe + loop detection keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches(); // loop detection keyframe 또는 covisibility graph 상에서 연결되어 있는 keyframe 상의 keypoint와 association의 관계를 가지는 map point들
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i]; // loop detection keyframe 또는 covisibility graph 상에서 연결되어 있는 keyframe 상의 keypoint와 association의 관계를 가지는 i번째 map point
            if(pMP) // 해당 map point가 존재한다면,
            {
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId) // 해당 map point가 나쁘지 않다고 판단하고, 해당 map point의 mnLoopPointForKF가 현재 keyframe의 id와 일치하지 않는다면,
                {
                    mvpLoopMapPoints.push_back(pMP); // loop detection keyframe과 neighbor keyframe들의 map point들
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId; // 중복 방지
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
    // loop detection keyframe + neighbor keyframes의 map points와 현재 keyframe의 map points와의 더 많은 correspondences를 찾는다. -> 또 한 번의 검증
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);

    // If enough matches accept Loop
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++) // 현재 keyframe과 loop detection keyframe의 map point와의 correspondence + loop detection keyframe의 neighbor keyframe의 map point와의 correspondence
    {
        if(mvpCurrentMatchedPoints[i]) // mvpCurrentMatchedPoints[i] != NULL
            nTotalMatches++; // 현재 keyframe과 loop detection keyframe 또는 neighbor keyframe의 map point와의 correspondence의 개수
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++) // loop detection candidate keyframe의 개수만큼 반복
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF) // i번째 loop detection candidate keyframe이 loop detection keyframe과 일치하지 않는다면,
            // loop detection keyframe을 제외한 나머지 loop detection candidate keyframe을 모두 삭제한다.
                mvpEnoughConsistentCandidates[i]->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        return true; // ComputeSim3() = true
    }
    else // nTotalMatches < 40
    {
        // 현재 keyframe과 loop detection keyframe 또는 neighbor keyframe의 map point와의 correspondence의 개수가 40 미만인 경우, 현재 keyframe과 모든 loop detection candidate keyframe을 삭제한다.
        for(int i=0; i<nInitialCandidates; i++) // loop detection candidate keyframe의 개수만큼 반복
            mvpEnoughConsistentCandidates[i]->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        mpCurrentKF->SetErase(); // Keyframe Culling : 해당 keyframe과 관련있는 모든 것에서 해당 keyframe에 대한 정보 삭제 + 해당 keyframe의 children keyframe의 parent keyframe을 할당
        return false; // ComputeSim3() = false
    }

}

void LoopClosing::CorrectLoop()
{
    cout << "Loop detected!" << endl;

    // Send a stop signal to Local Mapping
    // Avoid new keyframes are inserted while correcting the loop
    mpLocalMapper->RequestStop();
    // mbStopRequested = true, mbAbortBA = true

    // If a Global Bundle Adjustment is running, abort it
    if(isRunningGBA()) // isRunningGBA() = true -> mbRunningGBA = true
    {
        unique_lock<mutex> lock(mMutexGBA); // unique_lock class의 객체인 lock은 mutex 객체인 mMutexGBA를 소유한다.
        mbStopGBA = true;

        mnFullBAIdx++;

        if(mpThreadGBA) // GBA thread가 돌아가고 있다면,
        {
            mpThreadGBA->detach(); // std::thread::detach() -> thread 개체에서 thread를 떼어 낸다.
            delete mpThreadGBA; // Q. 해당 thread에 할당된 메모리 삭제?
        }
    }

    // Wait until Local Mapping has effectively stopped
    // Local mapping이 중지되지 않았다면,
    while(!mpLocalMapper->isStopped()) // mpLocalMapper->isStopped() = false
    {
        usleep(1000); // 1000 마이크로 초 동안 호출 프로세스의 실행을 일시중지한다.
    }

    // Ensure current keyframe is updated
    mpCurrentKF->UpdateConnections();

    // Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
    mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames(); // 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 정렬

    mvpCurrentConnectedKFs.push_back(mpCurrentKF); // mvpCurrentConnectedKFs = 현재 keyframe + 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes

    KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
    CorrectedSim3[mpCurrentKF]=mg2oScw; // CorrectedSim3->first : 현재 keyframe, CorrectedSim3->second : similarity transformation으로 구한 Scm x Smw, 현재 keyframe의 보정된 world to camera coordinate의 pose
    cv::Mat Twc = mpCurrentKF->GetPoseInverse(); // 현재 keyframe의 camera to world coordinate의 pose -> similarity transformation으로 보정하기 전의 pose

    // Critical Section
    {
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate); // unique_lock class의 객체인 lock은 mutex 객체인 mpMap->mMutexMapUpdate를 소유한다.

        for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
        {
            KeyFrame* pKFi = *vit; // de-reference -> 현재 keyframe + 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes

            cv::Mat Tiw = pKFi->GetPose(); // 현재 keyframe 또는 neighbor keyframe의 world to camera coordinate -> similarity transformation으로 보정하기 전의 pose

            if(pKFi!=mpCurrentKF) // 현재 keyframe이 아니라면,
            {
                cv::Mat Tic = Tiw*Twc; // 현재 keyframe의 neighbor keyframe의 world to camera coordinate의 transformation x 현재 keyframe의 camera to world to coordinate의 transformation
                // = 현재 keyframe to neighbor keyframe coordinate의 transformation
                cv::Mat Ric = Tic.rowRange(0,3).colRange(0,3); // 현재 keyframe to neighbor keyframe coordinate의 rotation
                cv::Mat tic = Tic.rowRange(0,3).col(3); // 현재 keyframe to neighbor keyframe coordinate의 translation
                g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric),Converter::toVector3d(tic),1.0);
                g2o::Sim3 g2oCorrectedSiw = g2oSic*mg2oScw; // 현재 keyframe to neighbor keyframe coordinate의 relative pose x similarity transformation으로 보정된 world to camera coordinate의 pose
                //Pose corrected with the Sim3 of the loop closure
                CorrectedSim3[pKFi]=g2oCorrectedSiw; // CorrectedSim3->first : 현재 keyframe의 neighbor keyframe, CorrectedSim3->second : Sic x similarity transformation으로 구한 Scw,
                // neighbor keyframe의 보정된 world to camera coordinate의 pose
            }

            cv::Mat Riw = Tiw.rowRange(0,3).colRange(0,3); // 현재 keyframe 또는 neighbor keyframe의 world to camera coordinate의 rotation -> similarity transformation으로 보정하기 전의 pose
            cv::Mat tiw = Tiw.rowRange(0,3).col(3); // 현재 keyframe 또는 neighbor keyframe의 world to camera coordinate의 translation -> similarity transformation으로 보정하기 전의 pose
            g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw),Converter::toVector3d(tiw),1.0);
            //Pose without correction
            NonCorrectedSim3[pKFi]=g2oSiw; // NonCorrectedSim3->first : 현재 keyframe 또는 neighbor keyframe, NonCorrectedSim3->second : similarity transformation으로 보정하기 전의 world to camera coordinate의 pose
        }

        // Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
        for(KeyFrameAndPose::iterator mit=CorrectedSim3.begin(), mend=CorrectedSim3.end(); mit!=mend; mit++)
        {
            KeyFrame* pKFi = mit->first; // 현재 keyframe 또는 neighbor keyframe
            g2o::Sim3 g2oCorrectedSiw = mit->second; // 현재 keyframe 또는 neighbor keyframe의 similarity transformation으로 보정된 world to camera coordinate의 pose
            g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse(); // 현재 keyframe 혹은 neighbor keyframe의 similarity transformation으로 보정된 camera to world coordinate의 pose

            g2o::Sim3 g2oSiw =NonCorrectedSim3[pKFi]; // similarity transformation으로 보정하기 전의 world to camera coordinate의 pose
            
            // map point에 대한 보정 + 처리
            vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches(); // 현재 keyframe 또는 neighbor keyframe 상의 keypoint와 association의 관계를 가지는 map point
            for(size_t iMP=0, endMPi = vpMPsi.size(); iMP<endMPi; iMP++) // 현재 keyframe 또는 neighbor keyframe 상의 keypoint와 association의 관계를 가지는 모든 map point에 대하여 반복
            {
                MapPoint* pMPi = vpMPsi[iMP]; // 현재 keyframe 또는 neighbor keyframe의 iMP번째 map point
                if(!pMPi) // 해당 map point가 존재하지 않는다면,
                    continue; // 해당 루프의 끝으로 이동한다.
                if(pMPi->isBad()) // 해당 map point가 나쁘다고 판단하면,
                    continue; // 해당 루프의 끝으로 이동한다.
                if(pMPi->mnCorrectedByKF==mpCurrentKF->mnId) // 해당 map point의 mnCorrectedByKF가 현재 keyframe id이면, -> 중복 방지
                    continue; // 해당 루프의 끝으로 이동한다.

                // Project with non-corrected pose and project back with corrected pose
                cv::Mat P3Dw = pMPi->GetWorldPos(); // 절대 좌표계인 world 좌표계 상의 해당 map point의 position
                Eigen::Matrix<double,3,1> eigP3Dw = Converter::toVector3d(P3Dw);
                Eigen::Matrix<double,3,1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));
                // g2oSiw.map(eigP3Dw) -> 보정되기 전의 world to camera coordinate의 pose x world 좌표계 상의 해당 map point의 position = 보정되기 전의 camera 좌표계 상의 map point의 position
                // g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw)) -> 보정된 후의 camera to world coordinate의 pose x 보정되기 전의 camera 좌표계 상의 map point의 position = 보정된 후의 world 좌표계 상의 map point의 position

                cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
                pMPi->SetWorldPos(cvCorrectedP3Dw); // cvCorrectedP3Dw -> mWorldPos(절대 좌표계인 world 좌표계 상의 map point의 position)로의 깊은 복사
                pMPi->mnCorrectedByKF = mpCurrentKF->mnId; // 중복 방지
                pMPi->mnCorrectedReference = pKFi->mnId;
                pMPi->UpdateNormalAndDepth(); // 보정된 map point의 max distance와 min distance, mean normal vector를 계산한다.
            }

            // keyframe에 대한 처리
            // Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
            Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix(); // 보정된 후의 world to camera coordinate의 rotation, normalized -> determinant(eigR) = 1
            Eigen::Vector3d eigt = g2oCorrectedSiw.translation(); // 보정된 후의 world to camera coordinate의 translation
            double s = g2oCorrectedSiw.scale(); // 보정된 후의 world to camera coordinate의 scale

            eigt *=(1./s); //[R t/s;0 1] -> rotation이 normalize 되었기 때문에, translation에 scale을 나눠준다.

            cv::Mat correctedTiw = Converter::toCvSE3(eigR,eigt); // 보정된 후의 world to camera coordinate의 similarity transformation

            pKFi->SetPose(correctedTiw); // 해당 keyframe의 camera to world coordinate의 transformation

            // Make sure connections are updated
            pKFi->UpdateConnections();
        }

        // Start Loop Fusion
        // Update matched map points and replace if duplicated
        // mvpCurrentMatchedPoints -> 현재 keyframe과 loop detection keyframe(loop detection keyframe의 neighbor keyframe)의 map point와의 correspondence
        // 중복되는 map points fusion
        // Q. mvpCurrentMatchedPoints -> 보정된 map points?
        for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
        {
            if(mvpCurrentMatchedPoints[i]) // i번째 map point가 존재한다면,
            {
                MapPoint* pLoopMP = mvpCurrentMatchedPoints[i]; // 현재 keyframe과 loop detection keyframe(loop detection keyframe의 neighbor keyframe)의 i번째 map point와의 correspondence
                MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i); // 현재 keyframe 상의 i번째 keypoint와 association의 관계를 가지는 map point -> Q. 보정되지 않은 map point
                if(pCurMP) // 현재 keyframe의 i번째 map point가 존재한다면, 
                    pCurMP->Replace(pLoopMP); // 현재 keyframe의 i번째 map point를 지우고, loop detection keyframe의 map point로 대체한다.
                else // 현재 keyframe의 i번째 map point가 존재하지 않는다면, 
                {
                    mpCurrentKF->AddMapPoint(pLoopMP,i); // loop detection keyframe의 map point를 현재 keyframe의 mvpMapPoints vector에 귀속시킨다.
                    pLoopMP->AddObservation(mpCurrentKF,i); // Data Association -> loop detection keyframe의 map point가 현재 keyframe에서 발견된 i번째 keypoint임을 저장한다.
                    pLoopMP->ComputeDistinctiveDescriptors(); // loop detection keyframe의 map point의 representative descriptor(다른 descriptor와의 hamming distance가 가장 작은 descriptor)를 저장한다.
                }
            }
        }

    }

    // Project MapPoints observed in the neighborhood of the loop keyframe
    // into the current keyframe and neighbors using corrected poses.
    // Fuse duplications.
    // loop detection keyframe(neighbor keyframe)과 현재 keyframe(neighbor keyframe)의 map point correspondence를 찾고, 현재 keyframe의 map point를 loop detection keyframe의 map point로 대체한다.
    // Q. loop detection keyframe의 map point가 현재 keyframe의 map point보다 더 정확하다고 판단하기 때문
    SearchAndFuse(CorrectedSim3);


    // After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
    map<KeyFrame*, set<KeyFrame*> > LoopConnections;

    // mvpCurrentConnectedKFs = 현재 keyframe + 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes
    for(vector<KeyFrame*>::iterator vit=mvpCurrentConnectedKFs.begin(), vend=mvpCurrentConnectedKFs.end(); vit!=vend; vit++)
    {
        KeyFrame* pKFi = *vit; // de-reference -> 현재 keyframe 또는 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes
        vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames(); // covisibility graph 상에서 연결되어 있는 keyframe들을 weight 순으로 정렬

        // Update connections. Detect new links.
        pKFi->UpdateConnections(); // map point가 갱신되었기 때문에, keyframe과 keyframe 간의 edge 또한 갱신되었을 것이다.
        LoopConnections[pKFi]=pKFi->GetConnectedKeyFrames(); // 해당 keyframe의 map point들이 관측되는 keyframe들
        // LoopConnections->first : 현재 keyframe 또는 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes, LoopConnections->second : 해당 keyframe의 map point들이 관측되는 keyframe들
        for(vector<KeyFrame*>::iterator vit_prev=vpPreviousNeighbors.begin(), vend_prev=vpPreviousNeighbors.end(); vit_prev!=vend_prev; vit_prev++)
        {
            LoopConnections[pKFi].erase(*vit_prev); // 해당 keyframe의 map point들이 관측되는 keyframe들 set에서 이전에 해당 keyframe의 covisibility graph 상에서 연결되어 있는 keyframe들을 삭제한다.
        }
        for(vector<KeyFrame*>::iterator vit2=mvpCurrentConnectedKFs.begin(), vend2=mvpCurrentConnectedKFs.end(); vit2!=vend2; vit2++)
        {
            LoopConnections[pKFi].erase(*vit2); // 해당 keyframe의 map point들이 관측되는 keyframe들 set에서 해당 keyframe들(현재 keyframe + 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes)을 삭제한다.
        }
        // LoopConnections = new - old = 새롭게 갱신된 loop candidate keyframe과 현재 keyframe 간의 loop edge
        // LoopConnections->first : 현재 keyframe + 현재 keyframe과 covisibility graph 상에서 연결되어 있는 keyframes, LoopConnections->second : 새롭게 갱신된 loop candidate keyframe과 현재 keyframe 간의 loop edge
    }

    // Optimize graph
    Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale); // 보류

    mpMap->InformNewBigChange(); // mnBigChangeIdx++;

    // Add loop edge
    mpMatchedKF->AddLoopEdge(mpCurrentKF); // loop detection keyframe의 mspLoopEdges에 현재 keyframe을 삽입한다.
    mpCurrentKF->AddLoopEdge(mpMatchedKF); // 현재 keyframe의 mspLoopEdges에 loop detection keyframe을 삽입한다.

    // Launch a new thread to perform Global Bundle Adjustment
    mbRunningGBA = true;
    mbFinishedGBA = false;
    mbStopGBA = false;
    mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment,this,mpCurrentKF->mnId); // 보류

    // Loop closed. Release Local Mapping.
    mpLocalMapper->Release(); // local mapping thread의 처리 대상인 새로운 keyframe list를 삭제하고, local mapping을 재게한다.

    mLastLoopKFid = mpCurrentKF->mnId;
}

// loop detection keyframe(neighbor keyframe)과 현재 keyframe(neighbor keyframe)의 map point correspondence를 찾고, 현재 keyframe의 map point를 loop detection keyframe의 map point로 대체한다.
// Q. loop detection keyframe의 map point가 현재 keyframe의 map point보다 더 정확하다고 판단하기 때문
// SearchAndFuse(CorrectedSim3)
// CorrectedSim3->first : 현재 keyframe 또는 현재 keyframe의 neighbor keyframe
// CorrectedSim3->second : similarity transformation으로 구한 Scm x Smw, 현재 keyframe의 보정된 world to camera coordinate의 pose,
// Sic x similarity transformation으로 구한 Scw, neighbor keyframe의 보정된 world to camera coordinate의 pose 
void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first; // 현재 keyframe 또는 현재 keyframe의 neighbor keyframe

        g2o::Sim3 g2oScw = mit->second; // 현재 keyframe 또는 현재 keyframe의 neighbor keyframe의 보정된 world to camera coordinate의 pose
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        // mvpLoopMapPoints -> loop detection keyframe과 neighbor keyframe들의 map point들
        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL)); // loop detection keyframe과 neighbor keyframe들의 map point들의 크기만큼 NULL 값으로 초기화
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints); // 현재 keyframe의 map point와 겹치는 loop detection keyframe의 map point를 구하는 과정
        // vpReplacePoint->first : loop detection의 map point index, 
        // vpReplacePoint->second : loop detection keyframe과 correspondence의 관계를 갖는 현재 keyframe의 map point
        
        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate); // unique_lock class의 객체인 lock은 mutex 객체인 mpMap->mMutexMapUpdate를 소유한다.
        const int nLP = mvpLoopMapPoints.size(); // loop detection keyframe과 neighbor keyframe들의 map point들의 개수
        for(int i=0; i<nLP;i++) // loop detection keyframe과 neighbor keyframe들의 map point들의 개수만큼 반복
        {
            MapPoint* pRep = vpReplacePoints[i]; // loop detection keyframe(neighbor keyframe)과 correspondence의 관계를 갖는 현재 keyframe의 map point
            if(pRep) // 해당 map point가 존재한다면,
            {
                pRep->Replace(mvpLoopMapPoints[i]); // 현재 keyframe의 map point(Q. 보정 전의 map point일테니까?)를 loop detection keyframe의 map point로 대체한다.
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset); // unique_lock class의 객체인 lock이 mutex 객체인 mMutexReset를 소유한다.
        mbResetRequested = true; // mbResetRequested flag를 true로 설정한다.
    }

    // reset이 완료되기 전까지 대기하라.
    while(1)
    {
        // Critical Section -> Q. 위의 lock과 아래의 lock2는 mMutexReset이라는 같은 mutex 객체를 소유하는데, 왜 다른 unique_lock class 객체를 쓰는가?
        {
        unique_lock<mutex> lock2(mMutexReset); // unique_lock class의 객체인 lock이 mutex 객체인 mMutexReset를 소유한다.
        if(!mbResetRequested) // mbResetRequested = false
            break; // mbResetRequested = false라면, while문을 빠져나가라.
        }
        usleep(5000); // 5000 마이크로 초 동안 대기하라.
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
    cout << "Starting Global Bundle Adjustment" << endl;

    int idx =  mnFullBAIdx;
    Optimizer::GlobalBundleAdjustemnt(mpMap,10,&mbStopGBA,nLoopKF,false);

    // Update all MapPoints and KeyFrames
    // Local Mapping was active during BA, that means that there might be new keyframes
    // not included in the Global BA and they are not consistent with the updated map.
    // We need to propagate the correction through the spanning tree
    {
        unique_lock<mutex> lock(mMutexGBA);
        if(idx!=mnFullBAIdx)
            return;

        if(!mbStopGBA)
        {
            cout << "Global Bundle Adjustment finished" << endl;
            cout << "Updating map ..." << endl;
            mpLocalMapper->RequestStop();
            // Wait until Local Mapping has effectively stopped

            while(!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
            {
                usleep(1000);
            }

            // Get Map Mutex
            unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

            // Correct keyframes starting at map first keyframe
            list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(),mpMap->mvpKeyFrameOrigins.end());

            while(!lpKFtoCheck.empty())
            {
                KeyFrame* pKF = lpKFtoCheck.front();
                const set<KeyFrame*> sChilds = pKF->GetChilds();
                cv::Mat Twc = pKF->GetPoseInverse();
                for(set<KeyFrame*>::const_iterator sit=sChilds.begin();sit!=sChilds.end();sit++)
                {
                    KeyFrame* pChild = *sit;
                    if(pChild->mnBAGlobalForKF!=nLoopKF)
                    {
                        cv::Mat Tchildc = pChild->GetPose()*Twc;
                        pChild->mTcwGBA = Tchildc*pKF->mTcwGBA;//*Tcorc*pKF->mTcwGBA;
                        pChild->mnBAGlobalForKF=nLoopKF;

                    }
                    lpKFtoCheck.push_back(pChild);
                }

                pKF->mTcwBefGBA = pKF->GetPose();
                pKF->SetPose(pKF->mTcwGBA);
                lpKFtoCheck.pop_front();
            }

            // Correct MapPoints
            const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

            for(size_t i=0; i<vpMPs.size(); i++)
            {
                MapPoint* pMP = vpMPs[i];

                if(pMP->isBad())
                    continue;

                if(pMP->mnBAGlobalForKF==nLoopKF)
                {
                    // If optimized by Global BA, just update
                    pMP->SetWorldPos(pMP->mPosGBA);
                }
                else
                {
                    // Update according to the correction of its reference keyframe
                    KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

                    if(pRefKF->mnBAGlobalForKF!=nLoopKF)
                        continue;

                    // Map to non-corrected camera
                    cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0,3).colRange(0,3);
                    cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0,3).col(3);
                    cv::Mat Xc = Rcw*pMP->GetWorldPos()+tcw;

                    // Backproject using corrected camera
                    cv::Mat Twc = pRefKF->GetPoseInverse();
                    cv::Mat Rwc = Twc.rowRange(0,3).colRange(0,3);
                    cv::Mat twc = Twc.rowRange(0,3).col(3);

                    pMP->SetWorldPos(Rwc*Xc+twc);
                }
            }            

            mpMap->InformNewBigChange();

            mpLocalMapper->Release();

            cout << "Map updated!" << endl;
        }

        mbFinishedGBA = true;
        mbRunningGBA = false;
    }
}

void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
