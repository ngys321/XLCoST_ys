#!/bin/bash

# DOCS (sbatch, srun)
# https://github.com/dasandata/Open_HPC/blob/master/Document/User%20Guide/5_use_resource/5.2_Allocate_Resource.md

# HOW TO USE SBATCH
# gpu 4개, 사용시간은 48시간으로 지정하여 jop.sh 작업제출
# 참고 : --time=일-시간:분:초
# (base 파티션 & base_qos 인 경우)  sbatch --gres=gpu:4 --time=48:00:00 ./job.sh
# (big 파티션 & big_qos 인 경우)  sbatch -p big -q big_qos --gres=gpu:4 --time=48:00:00 ./job.sh

# HOW TO USE SRUN
# gpu 4개, 사용시간은 48시간으로 지정하여 bash 실행
# srun --gres=gpu:4 --time=48:00:00 --pty bash -i

## sbatch 돌아가고 있는 상태 확인
# squeuelong -u ysnamgoong42
#################################################################################################

PROBING_CASE=${1:-0} # run_new.py 의 args.probing_case 에 들어갈 값.    run_new.py 내에서 여러 코드 경우의 수를 실험해보기 위해서, 사용하는 probing 변수임.
                     #
                     #                                               run_new.py 내에서 if arg.probing_case == 1: (코드 경우1)
                     #                                               run_new.py 내에서 if arg.probing_case == 2: (코드 경우2)
                     #                                               run_new.py 내에서 if arg.probing_case == 3: (코드 경우3)
                     #                                               위 처럼 코드를 작성해두고, 나중에 터미널에서 이 스크립트 파일를 실행시킬때, 첫번째 argument 의 값을 1,2,3 으로 바꿔가며 여러개의 코드 경우(1,2,3)를 실험해보면 된다.
                     #                                               ex. $job_~.sh 1      (run_new.py 내의 코드 경우1 을 실험하기 위한 터미널 커맨드)
# 메모 출력
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
echo "probing_case 도입"
echo 
echo 
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"

# job script 이름 출력
echo 
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ RUNNING SCRIPT: $SLURM_JOB_NAME"
echo


# conda 환경 활성화.
source ~/.bashrc
conda activate xlcost

# cuda 11.7 환경 구성.
ml purge
ml load cuda/11.7

# GPU 체크
nvidia-smi
nvcc -V

######################################## 작업 준비 끝 ############################################
# 활성화된 환경에서 코드 실행.

echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START TRAIN @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
bash run_NL_PL_new.sh 0 desc python program plbart train 10 $PROBING_CASE   # .sh파일명 이후, argument 순서대로 : gpu지정번호(안쓰임)/source언어/target언어/program인지snippet인지/모델/train인지eval인지/epoch수/probing_case변수
echo "@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@ START EVAL @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
bash run_NL_PL_new.sh 0 desc python program plbart eval 10 $PROBING_CASE
