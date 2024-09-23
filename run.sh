date

# 3-stream
python main.py pretrain_skeleton_3views --config config/three_streams/ntu60_cs/pretext_caca_3views_xsub_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/ntu60_cs/linear_eval_caca_3views_xsub_cross_2_10.yaml

python main.py pretrain_skeleton_3views --config config/three_streams/ntu60_cv/pretext_caca_3views_xview_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/ntu60_cv/linear_eval_caca_3views_xview_cross_2_10.yaml

python main.py pretrain_skeleton_3views --config config/three_streams/pkup1cs/pretext_caca_3views_xsub_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/pkup1cs/linear_eval_caca_3views_xsub_cross_2_10.yaml

python main.py pretrain_skeleton_3views --config config/three_streams/pkup2cs/pretext_caca_3views_xsub_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/pkup2cs/linear_eval_caca_3views_xsub_cross_2_10.yaml

python main.py pretrain_skeleton_3views --config config/three_streams/ntu120_cs/pretext_caca_3views_xsub_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/ntu120_cs/linear_eval_caca_3views_xsub_cross_2_10.yaml

python main.py pretrain_skeleton_3views --config config/three_streams/ntu120_cv/pretext_caca_3views_xview_cross_2_10.yaml
python main.py linear_evaluation_3views --config config/three_streams/ntu120_cv/linear_eval_caca_3views_xview_cross_2_10.yaml


# 1-stream
python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cs/pretext/pretext_caca_512_2048_512_2048_0.996_joint.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cs/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_joint.yaml

python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cs/pretext/pretext_caca_512_2048_512_2048_0.996_motion.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cs/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_motion.yaml

python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cs/pretext/pretext_caca_512_2048_512_2048_0.996_bone.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cs/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_bone.yaml

python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cv/pretext/pretext_caca_512_2048_512_2048_0.996_joint.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cv/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_joint.yaml

python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cv/pretext/pretext_caca_512_2048_512_2048_0.996_motion.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cv/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_motion.yaml

python main.py pretrain_skeleton --config config/single_stream/stgcn/ntu60_cv/pretext/pretext_caca_512_2048_512_2048_0.996_bone.yaml
python main.py linear_evaluation --config config/single_stream/stgcn/ntu60_cv/linear_eval/linear_eval_caca_512_2048_512_2048_0.996_bone.yaml
