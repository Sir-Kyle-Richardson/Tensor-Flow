Send-AU3Key -Key "{F5}";
clear;
python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m unittest $args[0]