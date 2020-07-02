dlp submit -a clsi \
              -d scpn \
              -n scpn \
              -e demo.sh \
              -i reg.deeplearning.cn/ayers/nvidia-cuda:9.1-cudnn7-devel-centos7-py2 \
              --useGpu -g 1 -t PtJob \
              -k TeslaM40 \
              -l /ps2/intern/clsi/scpn/demo_trainDis_75.log \
              -o /ps2/intern/clsi/scpn/demo_trainDis_75.out
