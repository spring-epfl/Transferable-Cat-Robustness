Normal adversarial training:
python train.py --dataset bank_fraud --eps 10 --model_path ../test.pt --attack_iters 20 --batch_size 1024 --epochs 30 --target-class 0

Bilevel minimization:
python train.py --dataset bank_fraud --eps 10 --model_path ../test.pt --emb-only --emb-fine --emb-adv-iters 1  --attack_iters 2 --batch_size 1024 --epochs 30 --iter_lim 20 --target-class 0

Evaluating the model:
python eval.py --dataset bank_fraud --cost_bound 10 --attack greedy_delta --utility_type success_rate --model_path ../test.pt
